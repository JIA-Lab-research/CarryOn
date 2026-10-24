"""Registry visual memory, memory-conditioned recall, and on-object SoM selector."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

from .semantic_memory import (
    BASELINE_TEMPLATE, REAL_TEMPLATE, SIM_TEMPLATE, SemanticMemorySnapshot,
    array_sha256, canonical_json_sha256, deterministic_permutation,
    mask_sha256, sha256_bytes,
)


def _norm(name: str) -> str:
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisualObjectMemory:
    """Load the reference catalog used for object identity."""
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path).resolve()
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = {}
        for item in raw.get("objects", []):
            path = Path(item["canonical_rgb"])
            if not path.is_absolute():
                path = (self.manifest_path.parent / path).resolve()
            if not path.is_file():
                raise ValueError(f"Missing visual memory reference: {path}")
            forbidden = {"task_id", "episode_id", "init_state_index", "pair_id",
                         "xyz", "quat", "pose", "trajectory", "instance_id"}
            if forbidden.intersection(item):
                raise ValueError(f"instance information forbidden in visual memory: {path}")
            self.entries[_norm(item["object_name"])] = path
        if not self.entries:
            raise ValueError("visual memory has no objects")

    def target_name_from_instruction(self, instruction: str) -> str:
        text = _norm(instruction)
        hits = [name for name in self.entries if name == text or name in text]
        if len(hits) != 1:
            raise KeyError(
                f"instruction must identify exactly one catalog object; hits={hits!r}"
            )
        return hits[0]

    def reference_for(self, object_name: str) -> Path:
        key = self.target_name_from_instruction(object_name)
        return self.entries[key]


class VisualMemorySomSelector:
    """SAM3 candidates + readable SoM + one strict VLM candidate-id decision."""
    def __init__(
        self,
        memory: VisualObjectMemory,
        vlm,
        *,
        debug_dir: str = "",
        memory_conditioned_recovery: bool = False,
        target_pointer=None,
        molmo_target_directed: bool = False,
        semantic_memory: Optional[SemanticMemorySnapshot] = None,
        semantic_environment: str = "sim",
        semantic_formal_seed: Optional[int] = None,
        semantic_episode_key: str = "",
    ):
        self.memory, self.vlm = memory, vlm
        self.debug_dir = Path(debug_dir).resolve() if debug_dir else None
        self.memory_conditioned_recovery = bool(memory_conditioned_recovery)
        self.target_pointer = target_pointer
        self.molmo_target_directed = bool(molmo_target_directed)
        self.semantic_memory = semantic_memory
        self.semantic_environment = str(semantic_environment)
        self.semantic_formal_seed = semantic_formal_seed
        self.semantic_episode_key = str(semantic_episode_key)
        self._semantic_call_index = 0
        self.last_evidence = {}
        self.last_error = None
        self._loop = None

    @staticmethod
    def _rgb_array(rgb):
        import cv2
        import numpy as np
        if isinstance(rgb, (bytes, bytearray)):
            bgr = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cannot decode current RGB")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if isinstance(rgb, (str, Path)):
            bgr = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"cannot read current RGB: {rgb}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"current RGB must be HxWx3, got {arr.shape}")
        return arr

    @staticmethod
    def render_som(rgb, masks):
        import cv2
        import numpy as np
        canvas = VisualMemorySomSelector._rgb_array(rgb).copy()
        mapping = []
        palette = [(255,40,40),(20,220,255),(80,255,80),(255,80,230),
                   (255,180,20),(70,120,255),(240,240,40),(40,255,190)]
        for idx, raw in enumerate(masks, start=1):
            mask = np.asarray(raw, dtype=bool)
            if mask.shape != canvas.shape[:2] or not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            cx, cy = int(round(float(xs.mean()))), int(round(float(ys.mean())))
            color = palette[(idx - 1) % len(palette)]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 3)
            radius = 22
            cv2.circle(canvas, (cx, cy), radius, color, -1)
            text = str(idx)
            scale, thick = 0.9, 3
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(canvas, text, (cx - tw//2, cy + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2,
                        cv2.LINE_AA)
            cv2.putText(canvas, text, (cx - tw//2, cy + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick,
                        cv2.LINE_AA)
            mapping.append({"candidate_id": idx, "center_px": [cx, cy],
                            "mask_pixels": int(mask.sum())})
        return canvas, mapping

    @staticmethod
    def _b64_jpeg(rgb) -> str:
        import cv2
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise ValueError("jpeg encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _chat(self, messages):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        try:
            return self._loop.run_until_complete(
                self.vlm.plan_chat(messages, response_schema_hint=False, temperature=0.0)
            )
        except TypeError:
            return self._loop.run_until_complete(self.vlm.plan_chat(messages))

    def collect_candidates(self, rgb, segmenter, *, prompt="object"):
        import numpy as np
        current = self._rgb_array(rgb)
        prompts = []
        for value in (prompt, *self.memory.entries.keys()):
            if value and value not in prompts:
                prompts.append(value)
        valid, sources = [], []
        for value in prompts:
            raw_masks = segmenter.segment_candidates(current, prompt=value) or []
            retry_threshold = getattr(
                getattr(segmenter, "config", None), "retry_threshold", None
            )
            if not raw_masks and retry_threshold is not None:
                raw_masks = segmenter.segment_candidates(
                    current, prompt=value, threshold=float(retry_threshold)
                ) or []
            for raw in raw_masks:
                mask = np.asarray(raw, dtype=bool)
                if mask.shape != current.shape[:2] or not mask.any():
                    continue
                duplicate = False
                for prior in valid:
                    inter = int((mask & prior).sum())
                    union = int((mask | prior).sum())
                    if union and inter / union >= 0.80:
                        duplicate = True
                        break
                if not duplicate:
                    valid.append(mask)
                    sources.append(value)
        return current, valid, sources

    def _memory_conditioned_point_mask(self, target_name, current, segmenter):
        """Use reference+current RGB for coarse localization, then SAM3 point mask."""
        import numpy as np

        ref_path = self.memory.reference_for(target_name)
        reference = self._rgb_array(ref_path)
        height, width = current.shape[:2]
        messages = [
            {
                "role": "system",
                "content": (
                    "Locate the one object matching the frozen reference in the "
                    "current scene. First compare the reference's label, cap, colors, "
                    "body shape, and packaging against every visible object. Do not "
                    "choose an adjacent carton, box, or sauce merely because it is "
                    "nearby or similarly colored. Return JSON only: "
                    "{\"point_1000\":[x,y]}. Coordinates must be integers from 0 "
                    "through 999 and the point must lie inside the matching object."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Target: {target_name!r}. First image is its frozen "
                            f"reference. Second is the current {width}x{height} RGB."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + self._b64_jpeg(reference)
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + self._b64_jpeg(current)
                        },
                    },
                ],
            },
        ]
        started = time.time()
        response = self._chat(messages)
        ended = time.time()
        text = (
            getattr(response, "text", None)
            or getattr(response, "content", None)
            or str(response)
        )
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        try:
            payload = json.loads(match.group(0)) if match else None
            point = payload.get("point_1000") if isinstance(payload, dict) else None
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(isinstance(value, bool) for value in point)
                or not all(isinstance(value, (int, float)) for value in point)
            ):
                raise ValueError("point_1000 must contain two numeric values")
            x1000, y1000 = float(point[0]), float(point[1])
            if not (0.0 <= x1000 <= 999.0 and 0.0 <= y1000 <= 999.0):
                raise ValueError(f"point_1000 out of bounds: {[x1000, y1000]}")
            x = int(round(x1000 * (width - 1) / 999.0))
            y = int(round(y1000 * (height - 1) / 999.0))
        except Exception as exc:
            self.last_error = f"MEMORY_CONDITIONED_LOCALIZATION_FAILURE: {exc}"
            return None, None

        evidence = {
            "mode": "memory_conditioned_point",
            "reference_rgb": str(ref_path),
            "reference_sha256": _sha(ref_path),
            "point_1000": [x1000, y1000],
            "point_px": [x, y],
            "judge_request_id": getattr(response, "request_id", None),
            "judge_raw_response": text,
            "judge_latency_ms": round((ended - started) * 1000, 3),
        }
        mask = segmenter.segment_at_point(current, point_px=(x, y))
        evidence["sam3_result"] = getattr(segmenter, "last_result", None)
        if mask is None:
            self.last_error = (
                "CANDIDATE_RECALL_FAILURE: point-prompt SAM3 returned no mask"
            )
            return None, evidence
        mask = np.asarray(mask, dtype=bool)
        evidence["mask_pixels"] = int(mask.sum())
        evidence["point_contained"] = bool(
            mask.shape == current.shape[:2] and mask[y, x]
        )
        if mask.shape != current.shape[:2] or not mask.any() or not mask[y, x]:
            self.last_error = (
                "CANDIDATE_RECALL_FAILURE: point-prompt mask invalid or misses point"
            )
            return None, evidence
        return mask, evidence

    def _molmo_target_directed_point_mask(self, target_name, current, segmenter):
        """Parse reference-conditioned pointing coordinates."""
        import cv2
        import numpy as np

        pointer = self.target_pointer
        if pointer is None:
            self.last_error = "REFERENCE_CONDITIONED_POINTER_FAILURE: pointer unavailable"
            return None, None
        ref_path = self.memory.reference_for(target_name)
        reference = self._rgb_array(ref_path)
        height, width = current.shape[:2]
        reference_panel = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
        composite = np.concatenate([reference_panel, current], axis=1)
        request_id = f"molmo-{time.time_ns()}"
        phrase = (
            f"object named {target_name!r} in the RIGHT-HAND CURRENT SCENE panel "
            "that visually matches the LEFT-HAND REFERENCE panel; point inside "
            "the object in the right panel, never at the left reference"
        )
        started = time.time()
        composite_point = pointer.point_at(
            composite, phrase, request_id=request_id
        )
        ended = time.time()
        evidence = {
            "mode": "reference_conditioned_molmo_point",
            "coordinate_contract": "current_frame_pixel_xy_v1",
            "reference_rgb": str(ref_path),
            "reference_sha256": _sha(ref_path),
            "request_id": request_id,
            "molmo_latency_ms": round((ended - started) * 1000, 3),
            "molmo_raw_response": getattr(pointer, "last_result", None),
            "molmo_error": getattr(pointer, "last_error", None),
            "composite_size_wh": [int(width * 2), int(height)],
        }
        if composite_point is None:
            self.last_error = (
                "REFERENCE_CONDITIONED_POINTER_FAILURE: "
                + str(getattr(pointer, "last_error", "no point"))
            )
            return None, evidence
        cx, cy = int(composite_point[0]), int(composite_point[1])
        evidence["composite_point_px"] = [cx, cy]
        if not (width <= cx < width * 2 and 0 <= cy < height):
            self.last_error = (
                "REFERENCE_CONDITIONED_POINTER_FAILURE: Molmo point is not in "
                f"right current panel: {[cx, cy]}"
            )
            return None, evidence
        x, y = cx - width, cy
        evidence["point_px"] = [x, y]
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(self.debug_dir / "MOLMO_REFERENCE_CURRENT_COMPOSITE.png"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            )
            overlay = current.copy()
            cv2.circle(overlay, (x, y), 12, (255, 40, 40), 3)
            cv2.drawMarker(
                overlay, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3
            )
            cv2.imwrite(
                str(self.debug_dir / "MOLMO_POINT_OVERLAY.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
        point_sam3_started = time.perf_counter()
        mask = segmenter.segment_at_point(current, point_px=(x, y))
        evidence["point_sam3_latency_ms"] = round(
            (time.perf_counter() - point_sam3_started) * 1000, 3
        )
        evidence["sam3_result"] = getattr(segmenter, "last_result", None)
        if mask is None:
            self.last_error = "CANDIDATE_RECALL_FAILURE: Molmo point SAM3 returned no mask"
            return None, evidence
        mask = np.asarray(mask, dtype=bool)
        evidence["mask_pixels"] = int(mask.sum())
        evidence["point_contained"] = bool(
            mask.shape == current.shape[:2] and mask[y, x]
        )
        if mask.shape != current.shape[:2] or not mask.any() or not mask[y, x]:
            self.last_error = (
                "CANDIDATE_RECALL_FAILURE: Molmo point mask invalid or misses point"
            )
            return None, evidence
        return mask, evidence

    @staticmethod
    def _strict_candidate_id(text: str, count: int) -> int:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        payload = json.loads(match.group(0)) if match else None
        cid = payload.get("candidate_id") if isinstance(payload, dict) else None
        if isinstance(cid, bool) or not isinstance(cid, int) or not 1 <= cid <= count:
            raise ValueError("candidate_id missing/out of_range")
        return cid

    def _write_semantic_event(self, evidence: dict) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        (self.debug_dir / "SEMANTIC_SELECTION_EVIDENCE.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        sidecar = self.debug_dir / "SEMANTIC_RUNTIME_LEDGER.jsonl"
        old = sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""
        line = json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
        tmp = sidecar.with_name(sidecar.name + f".tmp.{time.time_ns()}")
        tmp.write_text(old + line, encoding="utf-8")
        tmp.replace(sidecar)

    def _formal_hold(self, reason: str, evidence: dict) -> None:
        self.last_error = f"SEMANTIC_FORMAL_HOLD: {reason}"
        evidence.update({"formal_hold": True, "formal_hold_reason": reason})
        self.last_evidence = evidence
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / "FORMAL_HOLD.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self._write_semantic_event(evidence)

    def mark_execution(self, *, physical_descent_started: bool, motion_evidence: dict) -> dict:
        selection = dict(self.last_evidence.get("semantic_memory", self.last_evidence) or {})
        verified_sha = selection.get("verified_candidate_mask_sha256")
        event = {
            "schema": "carryon-semantic-execution-event/v1",
            "recorded_unix_ns": time.time_ns(),
            "semantic_call_index": selection.get("semantic_call_index"),
            "matched_entry_id": selection.get("matched_entry_id"),
            "route_template_id": selection.get("route_template_id"),
            "verified_candidate_mask_sha256": verified_sha,
            "physical_descent_started": bool(physical_descent_started),
            "executed": bool(physical_descent_started and verified_sha),
            "consumed_candidate_mask_sha256": (
                verified_sha if physical_descent_started and verified_sha else None
            ),
            "motion_evidence": dict(motion_evidence or {}),
        }
        self._write_semantic_event(event)
        return event

    def _semantic_fallback(
        self, attempt: dict, object_name: str, rgb, segmenter, *, prompt: str,
        reason: str,
    ):
        attempt.update({
            "fallback_used": True,
            "fallback_reason": str(reason),
            "verified": False,
            "verified_candidate_mask_sha256": None,
        })
        chosen = self._select_mask_baseline(object_name, rgb, segmenter, prompt=prompt)
        combined = {
            "schema": "carryon-semantic-selection-with-fallback/v1",
            "semantic_memory": attempt,
            "baseline_selection": dict(self.last_evidence or {}),
        }
        self.last_evidence = combined
        self._write_semantic_event(combined)
        return chosen

    def select_mask(
        self, object_name: str, rgb, segmenter, *, prompt="object",
        scene_bundle_identity=None,
    ):
        memory = self.semantic_memory
        if memory is None:
            return self._select_mask_baseline(object_name, rgb, segmenter, prompt=prompt)

        target_name = self.memory.target_name_from_instruction(object_name)
        ref_path = self.memory.reference_for(target_name)
        ref_sha = _sha(ref_path)
        route = memory.route(
            environment=self.semantic_environment,
            target_object_key=target_name,
            reference_sha256=ref_sha,
        )
        if route.template_id == BASELINE_TEMPLATE:
            return self._select_mask_baseline(object_name, rgb, segmenter, prompt=prompt)

        self._semantic_call_index += 1
        attempt = {
            "schema": "carryon-semantic-selection-event/v1",
            "recorded_unix_ns": time.time_ns(),
            "semantic_snapshot_path": str(memory.path),
            "semantic_snapshot_sha256": memory.sha256,
            "template_registry_sha256": memory.template_registry_sha256,
            "semantic_call_index": self._semantic_call_index,
            "matched_entry_id": route.entry_id,
            "entry_match_count": route.match_count,
            "route_template_id": route.template_id,
            "route_reason": route.reason,
            "environment": self.semantic_environment,
            "target_object_key": target_name,
            "reference_path": str(ref_path),
            "reference_sha256": ref_sha,
            "use_hints": False,
            "state_privilege_used": False,
            "fallback_used": False,
            "formal_hold": False,
        }
        if route.template_id == REAL_TEMPLATE:
            self._formal_hold("REAL_TEMPLATE_DISABLED", attempt)
            return None
        if route.template_id != SIM_TEMPLATE:
            self._formal_hold("UNKNOWN_OR_DISABLED_TEMPLATE", attempt)
            return None
        if self.semantic_formal_seed is None or not self.semantic_episode_key:
            self._formal_hold("MISSING_FORMAL_SEED_OR_EPISODE_KEY", attempt)
            return None
        if not isinstance(scene_bundle_identity, dict) or not scene_bundle_identity.get(
            "bundle_sha256"
        ):
            self._formal_hold("MISSING_SCENE_BUNDLE_IDENTITY", attempt)
            return None
        attempt["scene_bundle"] = dict(scene_bundle_identity)

        current, valid, candidate_sources = self.collect_candidates(
            rgb, segmenter, prompt=prompt,
        )
        count = len(valid)
        candidate_shas = [mask_sha256(mask) for mask in valid]
        attempt.update({
            "candidate_count": count,
            "candidate_sources": candidate_sources,
            "candidate_mask_sha256": candidate_shas,
            "candidate_bundle_sha256": canonical_json_sha256(candidate_shas),
        })
        if count < 1 or count > 8:
            return self._semantic_fallback(
                attempt, object_name, rgb, segmenter, prompt=prompt,
                reason=f"CANDIDATE_COUNT_OUT_OF_RANGE:{count}",
            )

        prompt_schema = {
            "schema": "carryon-sim-forced-choice-prompt/v1",
            "system": (
                "Select the one numbered candidate in the current scene that "
                "matches the frozen reference and instruction. Return JSON only: "
                "{\"candidate_id\": 2}. You must choose one displayed candidate ID."
            ),
            "response": {"candidate_id": "strict_integer_1_to_K"},
            "absent_allowed": False,
        }
        prompt_schema_sha = canonical_json_sha256(prompt_schema)
        response_schema_sha = canonical_json_sha256(prompt_schema["response"])
        calls = []
        selected_physical_shas = []
        overlays = []
        reference = self._rgb_array(ref_path)
        for round_index in (1, 2):
            order, presentation_seed = deterministic_permutation(
                count,
                formal_seed=int(self.semantic_formal_seed),
                episode_key=self.semantic_episode_key,
                semantic_call_index=self._semantic_call_index,
                prompt_schema_sha256=prompt_schema_sha,
                round_index=round_index,
            )
            presented = [valid[index] for index in order]
            overlay, mapping = self.render_som(current, presented)
            overlays.append(overlay)
            physical_mapping = [
                {
                    **mapping[presented_index],
                    "physical_index": physical_index,
                    "candidate_mask_sha256": candidate_shas[physical_index],
                }
                for presented_index, physical_index in enumerate(order)
            ]
            messages = [
                {"role": "system", "content": prompt_schema["system"]},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Instruction target: {object_name!r}. First image is "
                            "the frozen reference; second is current candidates."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + self._b64_jpeg(reference)
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + self._b64_jpeg(overlay)
                        },
                    },
                ]},
            ]
            started = time.time()
            response = self._chat(messages)
            ended = time.time()
            raw = (
                getattr(response, "text", None)
                or getattr(response, "content", None)
                or str(response)
            )
            call = {
                "round_index": round_index,
                "model_identity": type(self.vlm).__name__,
                "prompt_schema_sha256": prompt_schema_sha,
                "response_schema_sha256": response_schema_sha,
                "presentation_seed": presentation_seed,
                "presented_to_physical_mapping": physical_mapping,
                "som_image_sha256": array_sha256(overlay),
                "request_id": getattr(response, "request_id", None),
                "raw_response_sha256": sha256_bytes(raw.encode("utf-8")),
                "latency_ms": round((ended - started) * 1000, 3),
            }
            try:
                cid = self._strict_candidate_id(raw, count)
                physical_index = order[cid - 1]
                selected_sha = candidate_shas[physical_index]
                call.update({
                    "valid": True,
                    "parsed_candidate_id": cid,
                    "selected_physical_index": physical_index,
                    "selected_candidate_mask_sha256": selected_sha,
                })
                selected_physical_shas.append(selected_sha)
            except Exception as exc:
                call.update({
                    "valid": False,
                    "parsed_candidate_id": None,
                    "selected_physical_index": None,
                    "selected_candidate_mask_sha256": None,
                    "schema_error": f"{type(exc).__name__}: {exc}",
                })
                calls.append(call)
                attempt["selection_calls"] = calls
                return self._semantic_fallback(
                    attempt, object_name, rgb, segmenter, prompt=prompt,
                    reason=f"INVALID_SELECTION_CALL_{round_index}",
                )
            calls.append(call)

        attempt["selection_calls"] = calls
        if selected_physical_shas[0] != selected_physical_shas[1]:
            return self._semantic_fallback(
                attempt, object_name, rgb, segmenter, prompt=prompt,
                reason="SOM_REVERIFY_MASK_DISAGREEMENT",
            )

        verified_sha = selected_physical_shas[0]
        physical_index = candidate_shas.index(verified_sha)
        attempt.update({
            "verified": True,
            "verified_candidate_mask_sha256": verified_sha,
            "fallback_used": False,
            "fallback_reason": None,
        })
        self.last_error = None
        self.last_evidence = attempt
        if self.debug_dir:
            import cv2
            import numpy as np

            self.debug_dir.mkdir(parents=True, exist_ok=True)
            for round_index, overlay in enumerate(overlays, start=1):
                cv2.imwrite(
                    str(self.debug_dir / f"SEMANTIC_SOM_PRESENTATION_{round_index}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
            for index, mask in enumerate(valid, start=1):
                np.save(self.debug_dir / f"semantic_candidate_{index:02d}_mask.npy", mask)
            chosen = valid[physical_index]
            selected = current.copy()
            selected[chosen] = (
                0.55 * selected[chosen] + 0.45 * np.array([40, 255, 40])
            ).astype(np.uint8)
            cv2.imwrite(
                str(self.debug_dir / "SEMANTIC_SELECTED_OVERLAY.png"),
                cv2.cvtColor(selected, cv2.COLOR_RGB2BGR),
            )
        self._write_semantic_event(attempt)
        return valid[physical_index]

    def _select_mask_baseline(self, object_name: str, rgb, segmenter, *, prompt="object"):
        import cv2
        import numpy as np
        self.last_error = None
        selector_started = time.perf_counter()
        target_name = self.memory.target_name_from_instruction(object_name)
        candidate_sam3_started = time.perf_counter()
        if self.molmo_target_directed:
            # Resolve online identity from the reference image, Molmo point and point-SAM3 mask.
            # The broader candidate collector is used for offline evidence collection.
            current = self._rgb_array(rgb)
            valid, candidate_sources = [], []
            candidate_collection_mode = "skipped_online_molmo_direct"
        else:
            current, valid, candidate_sources = self.collect_candidates(
                rgb, segmenter, prompt=prompt)
            candidate_collection_mode = "online_candidate_union"
        candidate_sam3_latency_ms = round(
            (time.perf_counter() - candidate_sam3_started) * 1000, 3
        )
        recovery_evidence = None
        molmo_evidence = None
        if self.memory_conditioned_recovery:
            recovered, recovery_evidence = self._memory_conditioned_point_mask(
                target_name, current, segmenter
            )
            if recovered is None:
                self.last_evidence = {
                    "candidate_count": len(valid),
                    "object_name": target_name,
                    "memory_conditioned_recovery": recovery_evidence or True,
                }
                return None
            duplicate_index = None
            for index, prior in enumerate(valid):
                inter = int((recovered & prior).sum())
                union = int((recovered | prior).sum())
                if union and inter / union >= 0.80:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                valid.append(recovered)
                candidate_sources.append("memory_conditioned_point")
                recovery_evidence["candidate_id"] = len(valid)
            else:
                candidate_sources[duplicate_index] += "+memory_conditioned_point"
                recovery_evidence["candidate_id"] = duplicate_index + 1
        if self.molmo_target_directed:
            recovered, molmo_evidence = self._molmo_target_directed_point_mask(
                target_name, current, segmenter
            )
            if recovered is None:
                self.last_evidence = {
                    "candidate_count": len(valid),
                    "object_name": target_name,
                    "molmo_target_directed_recall": molmo_evidence or True,
                }
                return None
            # Registry-memory LIBERO contract: Molmo decides target identity and
            # point-SAM3 supplies the exact mask. Preserve the candidate SoM as
            # visualization evidence, but never let a second Judge override it.
            valid.append(recovered)
            candidate_sources.append("molmo_target_directed_point")
            molmo_evidence["candidate_id"] = len(valid)
            overlay, mapping = self.render_som(current, valid)
            cid = len(valid)
            ref_path = self.memory.reference_for(target_name)
            evidence = {
                "instruction_or_target_hint": object_name,
                "object_name": target_name,
                "reference_rgb": str(ref_path),
                "reference_sha256": _sha(ref_path),
                "candidate_count": len(valid),
                "mapping": mapping,
                "candidate_sources": candidate_sources,
                "candidate_collection_mode": candidate_collection_mode,
                "memory_conditioned_recovery": recovery_evidence,
                "molmo_target_directed_recall": molmo_evidence,
                "selected_candidate_id": cid,
                "selection_mode": "molmo_point_sam3_direct_freeze",
                "som_role": "visualization_only",
                "som_judge_called": False,
                "judge_request_id": None,
                "judge_raw_response": None,
                "judge_latency_ms": 0.0,
                "latency_ms": {
                    "candidate_sam3": candidate_sam3_latency_ms,
                    "molmo": float(molmo_evidence.get("molmo_latency_ms", 0.0)),
                    "point_sam3": float(
                        molmo_evidence.get("point_sam3_latency_ms", 0.0)
                    ),
                    "selector_total": round(
                        (time.perf_counter() - selector_started) * 1000, 3
                    ),
                },
            }
            if self.debug_dir:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.debug_dir / "CURRENT_RGB.png"), cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(self.debug_dir / "SOM_PRESENTATION.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                for i, mask in enumerate(valid, 1):
                    np.save(self.debug_dir / f"candidate_{i:02d}_mask.npy", mask)
                    cv2.imwrite(str(self.debug_dir / f"candidate_{i:02d}_mask.png"), mask.astype(np.uint8) * 255)
                selected = current.copy()
                selected[recovered] = (0.55 * selected[recovered] + 0.45 * np.array([40, 255, 40])).astype(np.uint8)
                cv2.imwrite(str(self.debug_dir / "SELECTED_OVERLAY.png"), cv2.cvtColor(selected, cv2.COLOR_RGB2BGR))
                (self.debug_dir / "SELECTION_EVIDENCE.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (self.debug_dir / "MOLMO_TARGET_DIRECTED_RECALL.json").write_text(json.dumps(molmo_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.last_evidence = evidence
            return recovered
        if not valid:
            self.last_error = "CANDIDATE_RECALL_FAILURE: SAM3 returned K=0"
            self.last_evidence = {"candidate_count": 0, "object_name": object_name}
            return None
        overlay, mapping = self.render_som(current, valid)
        ref_path = self.memory.reference_for(target_name)
        reference = self._rgb_array(ref_path)
        messages = [
            {"role":"system", "content":
             "Select the one numbered candidate in the current scene that matches "
             "the reference object and instruction. Return JSON only: {\"candidate_id\": 2}. "
             "You must choose one displayed candidate ID."},
            {"role":"user", "content":[
                {"type":"text", "text":f"Instruction target: {object_name!r}. First image is the frozen reference; second is current candidates."},
                {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,"+self._b64_jpeg(reference)}},
                {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,"+self._b64_jpeg(overlay)}},
            ]},
        ]
        started = time.time()
        response = self._chat(messages)
        ended = time.time()
        text = getattr(response, "text", None) or getattr(response, "content", None) or str(response)
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        try:
            payload = json.loads(match.group(0)) if match else None
            cid = payload.get("candidate_id") if isinstance(payload, dict) else None
            if isinstance(cid, bool) or not isinstance(cid, int) or not 1 <= cid <= len(valid):
                raise ValueError("candidate_id missing/out of range")
        except Exception as exc:
            self.last_error = f"MEMORY_SOM_SELECTION_FAILURE: {exc}"
            return None
        evidence = {"instruction_or_target_hint": object_name,
                    "object_name": target_name, "reference_rgb": str(ref_path),
                    "reference_sha256": _sha(ref_path), "candidate_count": len(valid),
                    "mapping": mapping, "candidate_sources": candidate_sources,
                    "memory_conditioned_recovery": recovery_evidence,
                    "molmo_target_directed_recall": molmo_evidence,
                    "selected_candidate_id": cid,
                    "judge_request_id": getattr(response, "request_id", None),
                    "judge_raw_response": text, "judge_latency_ms": round((ended-started)*1000,3)}
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.debug_dir/"CURRENT_RGB.png"), cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self.debug_dir/"SOM_PRESENTATION.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            for i, mask in enumerate(valid, 1):
                np.save(self.debug_dir/f"candidate_{i:02d}_mask.npy", mask)
                cv2.imwrite(str(self.debug_dir/f"candidate_{i:02d}_mask.png"), mask.astype(np.uint8)*255)
            selected = current.copy()
            selected[valid[cid-1]] = (0.55*selected[valid[cid-1]] + 0.45*np.array([40,255,40])).astype(np.uint8)
            cv2.imwrite(str(self.debug_dir/"SELECTED_OVERLAY.png"), cv2.cvtColor(selected, cv2.COLOR_RGB2BGR))
            (self.debug_dir/"SELECTION_EVIDENCE.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True)+"\n", encoding="utf-8")
            if recovery_evidence is not None:
                (self.debug_dir/"MEMORY_CONDITIONED_RECALL.json").write_text(
                    json.dumps(recovery_evidence, indent=2, sort_keys=True)+"\n",
                    encoding="utf-8",
                )
            if molmo_evidence is not None:
                (self.debug_dir/"MOLMO_TARGET_DIRECTED_RECALL.json").write_text(
                    json.dumps(molmo_evidence, indent=2, sort_keys=True)+"\n",
                    encoding="utf-8",
                )
        self.last_evidence = evidence
        return valid[cid-1]
