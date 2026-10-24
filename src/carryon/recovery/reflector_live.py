"""Generate bounded recovery-policy proposals from failed-attempt evidence."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .reflection import StrategyReflectionProposal, _check_strategy_structure
from .schema import EpisodeRecord

_SYSTEM_PROMPT = """You are the recovery-strategy reflector of a robot manipulation system.
A Vision-Language-Action policy failed a tabletop pick-and-place task, a scripted
recovery attempt ALSO failed, and the failure judge has fired again. Your job:
read the failed attempt's step trace and the attached time-ordered frames, work
out WHY that strategy missed, and propose ONE alternative strategy from the
whitelisted combination space below for the next attempt.

You do not write code. You pick one primitive per dimension and bounded params.

grounding_plan (ordered list -- ORDER IS AUTHORITATIVE, the first primitive with
evidence wins; reorder to demote a source you distrust, e.g. put recent_contact
before judge_point when the judge pointed at the wrong object):
  scene_diff | judge_point | recent_contact | task_target | point_model | multiview
  (scene_diff = diff against the episode-start frame; finds the object that
   MOVED since the episode began -- include it FIRST for displaced-object
   failures, omit it to disable)
perception_plan: single_view_topdown | hybrid_auto | multiview_obb
grasp_plan: topdown | center_correct | yaw_sweep | side_grasp | topk_goalset
HARD CONSTRAINT: hybrid_auto is live GraspGen and MUST pair with topk_goalset;
never pair GraspGen candidates with center_correct, side_grasp, or a single goto_grasp.
verification_plan: lift_assert | retry_then_lift_assert
handback_mode: full_hold | pre_grasp | complete
  How much of the task recovery finishes before returning control (an
  intervention-ladder rung). Choose from what the trace shows:
  * full_hold (default): regrasp -> lift -> confirm -> rewind to the pre-failure
    pose -> hand back. Use when the grasp itself was the problem and, once the
    object is re-held, the VLA can finish from in-distribution.
  * complete: regrasp -> lift -> carry to the place destination -> release (no
    handback). Choose when the trace shows the grasp CHAIN SUCCEEDED (lift/hold
    ok) but the episode still failed -- i.e. the handback could not finish the
    place. The recovery finishes it. (dominant engaged-failure mode)
  * pre_grasp: park at the pre-grasp standoff over the object, open, hand back so
    the VLA closes the last cm. Choose when the grasp keeps MISSING but the arm
    can clearly REACH the object -- let the in-distribution VLA do the fine pick.

params (all optional, all bounded -- proposals outside the bounds are rejected):
  top_k: int 1..30                     (candidates for goalset/center paths)
  prefer_low_tilt: bool                 max_tilt_deg: 5..65
  failopen_tilt_band_deg: 0..15         allow_tilt_failopen: bool
  depth_frac: 0..0.6   (fraction of the object's own height to descend below its top)
  yaw_mode: "pca" | "pca_perpendicular" | "topdown"
    (pca_perpendicular = jaws straddle ACROSS the mask's principal axis --
     required for side-lying bottles/cylinders whose long axis is the roll axis)
  yaw_offset_deg: -180..180 (yaw_sweep only)
  z_offset / x_offset / y_offset: -0.03..0.03 m
  lift_dz: 0.02..0.10                   close_width: 0..0.08
  place_release_bias_m: 0..0.08 (complete only; release height above the
    visually perceived receptacle rim; lower values reduce drop/bounce)
  sam_text_prompts: up to 3 short noun phrases (grounding text fallbacks)
  max_attempts: 1..3 (retry_then_lift_assert)
  target_point: [x, y]  (normalized image fractions, each 0..1, origin top-left)
    A failure-informed re-grounding point: when the trace/frames show the last
    attempt grounded or grasped the WRONG object (a neighbour, the basket), give
    the image-space point of the CORRECT object so grounding re-anchors there via
    the point-prompt segmenter -- steer AWAY from the object that was mis-grasped.
    Only emit it when you can localise the right object in the frames.

Reading the trace diagnostics: each grasp/lift/hold step may carry a
"diagnostics:" line with physical widths (metres) and heights. Use it to name
the failure precisely:
  * hold(...) "jaws closed to ~empty" == the gripper shut to its empty width with
    nothing between the fingers -> a closed-past / no-contact MISS. If the grasp
    reached the object but still closed empty, the grasp GEOMETRY is wrong
    (cylinder pinched along its axis, box top-edge pinched): switch grasp_plan
    (side_grasp / yaw_mode=pca_perpendicular / depth_frac), not grounding.
  * close(jaw_after > empty) then lift(jaw_after ~empty / ee_z rose) == the object
    was between the fingers at close but SLIPPED on the lift -> tighten close_width
    or lower the lift, keep the grounding.
  * if grounding never found the object (perceive/ground FAIL, or grasp landed on
    a neighbour) -> reorder grounding_plan (scene_diff first) and/or set
    target_point at the correct object.

trigger/validation buckets: diagnosis in {generic, wrong_object, empty_grasp,
knockdown, rewind_fail, perception_fail}; geometry in {unknown, tall_box,
cylinder, low_object, clutter}; scene_relation "tabletop".

Physical context: Franka Panda, 8 cm max jaw opening, top-down-ish overhead
camera, objects on a wood table. Side-lying bottles need the jaws to straddle
ACROSS the cylinder axis and descend below the equator; thin boxes need a deep
wrap (depth_frac) so the fingers do not pinch the top edge; goalset strategies
let the motion planner pick among many candidates while single-grasp paths
(center_correct/topdown) trust the mask centroid.

Respond with ONE JSON object, no markdown fences:
{"rationale": "<what the trace+frames show and why this change>",
 "selected_category": "<geometry bucket you believe is correct>",
 "override_default": true,
 "strategy": {
   "strategy_id": "reflector-<short-slug>",
   "trigger": {"diagnosis": "...", "geometry": "...", "scene_relation": "tabletop"},
   "validation_buckets": [{"diagnosis": "...", "geometry": "...", "scene_relation": "tabletop"}],
   "grounding_plan": ["..."],
   "perception_plan": "...",
   "grasp_plan": "...",
   "verification_plan": "...",
   "handback_mode": "full_hold | pre_grasp | complete",
   "params": {...},
   "description": "<one sentence>"}}
If you believe the library default should stand, respond with
{"rationale": "...", "override_default": false}."""


def _fmt_num(v: Any) -> str:
    """Compact numeric formatter that tolerates None / non-numbers."""
    if v is None:
        return "?"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _data_digest(data: Any) -> str:
    """Collect compact diagnostics for a recovery attempt."""
    if not isinstance(data, dict) or not data:
        return ""
    bits: list[str] = []

    def _width(state: Any) -> Optional[Any]:
        return state.get("w_meas") if isinstance(state, dict) else None

    candidates = data.get("grasp_candidates")
    debug = data.get("perceive_debug")
    if isinstance(candidates, list) or isinstance(debug, dict):
        candidates = candidates if isinstance(candidates, list) else []
        tilts = []
        for row in candidates:
            if not isinstance(row, dict) or row.get("tilt_deg") is None:
                continue
            try:
                tilts.append(float(row["tilt_deg"]))
            except (TypeError, ValueError):
                continue
        pc = debug.get("point_cloud") if isinstance(debug, dict) else None
        lo = pc.get("xyz_min") if isinstance(pc, dict) else None
        hi = pc.get("xyz_max") if isinstance(pc, dict) else None
        posture, extent = "unobserved", None
        if (isinstance(lo, (list, tuple)) and isinstance(hi, (list, tuple))
                and len(lo) == len(hi) == 3):
            try:
                extent = [float(b) - float(a) for a, b in zip(lo, hi)]
                if extent[2] < 0.025:
                    posture = "flat"
                elif extent[2] >= 1.5 * max(extent[0], extent[1]):
                    posture = "upright"
                else:
                    posture = "ambiguous_or_contaminated"
            except (TypeError, ValueError):
                extent = None
        tilt_text = "/".join(_fmt_num(v) for v in (min(tilts), max(tilts))) if tilts else "?"
        extent_text = ",".join(_fmt_num(v) for v in extent) if extent else "?"
        bits.append(f"perception(n_candidates={len(candidates)},tilt_min/max={tilt_text},"
                    f"posture={posture},extent_xyz=[{extent_text}])")

    g = data.get("gripper")
    if isinstance(g, dict):
        bits.append(
            f"close(cmd={_fmt_num(g.get('command_width'))},"
            f"jaw_after={_fmt_num(_width(g.get('state_after')))})"
        )
    lift = data.get("lift")
    if isinstance(lift, dict):
        st = lift.get("state_after") if isinstance(lift.get("state_after"), dict) else {}
        ee = st.get("ee_pos") if isinstance(st, dict) else None
        z = ee[2] if isinstance(ee, (list, tuple)) and len(ee) >= 3 else None
        bits.append(
            f"lift(dz={_fmt_num(lift.get('dz'))},"
            f"jaw_after={_fmt_num(_width(st))},ee_z={_fmt_num(z)})"
        )
    hc = data.get("holding_check")
    if isinstance(hc, dict):
        measured = hc.get("measured_width")
        empty = hc.get("empty_width")
        cmd = hc.get("closed_cmd_width")
        thr = hc.get("command_relative_threshold")
        contact = hc.get("grasp_contact")
        verdict = ""
        try:
            if contact is False:
                verdict = " -> grasp_contact=false (no object contact: width-only hold/slip inference forbidden)"
            elif measured is not None and empty is not None and float(measured) <= float(empty) + float(
                hc.get("hold_margin", 0.0) or 0.0
            ):
                verdict = " -> jaws closed to ~empty (no object between fingers: closed-past/no-contact miss)"
            elif measured is not None and cmd is not None and float(measured) > float(cmd):
                verdict = " -> jaws held wider than the close command (object present but grasp lost/slipped before this check)"
        except (TypeError, ValueError):
            verdict = ""
        bits.append(
            f"hold(measured={_fmt_num(measured)},closed_cmd={_fmt_num(cmd)},"
            f"empty={_fmt_num(empty)},thr={_fmt_num(thr)},"
            f"grasp_contact={contact}){verdict}"
        )
    sg = data.get("selected_grasp")
    if isinstance(sg, dict):
        fp = sg.get("fingertip")
        z = fp[2] if isinstance(fp, (list, tuple)) and len(fp) >= 3 else None
        bits.append(
            f"grasp(source={sg.get('source')},score={_fmt_num(sg.get('score'))},"
            f"approach={sg.get('approach')},center_correct={sg.get('center_correct')},"
            f"tip_z={_fmt_num(z)})"
        )
    place = data.get("place")
    if isinstance(place, dict):
        st = place.get("state_after") if isinstance(place.get("state_after"), dict) else {}
        ee = st.get("ee_pos") if isinstance(st, dict) else None
        ee_text = (
            ",".join(_fmt_num(v) for v in ee[:3])
            if isinstance(ee, (list, tuple)) and len(ee) >= 3 else "?"
        )
        bits.append(
            f"place(clearance={_fmt_num(place.get('clearance_m'))},"
            f"release_bias={_fmt_num(place.get('release_bias_m'))},ee_xyz=[{ee_text}])"
        )
    return "  ".join(bits)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON object extraction from an LLM reply."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class OnlineStrategyReflector:
    """Make one bounded reflection request per re-triggered job."""

    def __init__(
        self,
        client: Any,
        *,
        max_frames: int = 6,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.client = client
        self.max_frames = max(0, int(max_frames))
        self._log = log or (lambda _msg: None)

    # ---- evidence assembly ------------------------------------------------ #
    def _frame_parts(self, record: Optional[EpisodeRecord]) -> list[dict]:
        frames = list(getattr(record, "frames", None) or [])
        if not frames or self.max_frames == 0:
            return []
        frames.sort(key=lambda f: getattr(f, "ts", 0.0))
        step = max(1, len(frames) // self.max_frames)
        picked = frames[::step][: self.max_frames]
        parts: list[dict] = []
        t0 = getattr(picked[0], "ts", 0.0)
        for f in picked:
            p = getattr(f, "rgb_path", None)
            if not p:
                continue
            try:
                raw = Path(p).read_bytes()
            except OSError:
                continue
            b64 = base64.b64encode(raw).decode("ascii")
            parts.append({"type": "text", "text": f"[frame t+{getattr(f, 'ts', 0.0) - t0:.1f}s]"})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return parts

    @staticmethod
    def _trace_text(
        object_hint: str,
        geometry: str,
        failed_invocation: Any,
        library_ids: Sequence[str],
    ) -> str:
        lines = [
            f"task object: {object_hint!r}   perceived geometry: {geometry}",
            f"library seed strategies: {', '.join(library_ids)}",
        ]
        inv = failed_invocation
        if inv is not None:
            lines.append(
                f"FAILED ATTEMPT: policy={getattr(inv, 'policy_name', '?')}"
                f"@{getattr(inv, 'policy_version', '?')} outcome={getattr(inv, 'outcome', '?')}"
            )
            notes = getattr(inv, "notes", "") or ""
            if notes:
                lines.append(f"summary: {notes[:600]}")

            def _emit_steps(steps, depth=0):
                for s in steps or []:
                    if isinstance(s, dict):
                        op, ok, reason = s.get("op"), s.get("ok"), s.get("reason", "")
                        children = s.get("children")
                        data = s.get("data")
                    else:
                        op, ok, reason = getattr(s, "op", "?"), getattr(s, "ok", False), getattr(s, "reason", "")
                        children = getattr(s, "children", None)
                        data = getattr(s, "data", None)
                    pad = "  " * depth
                    lines.append(f"{pad}- {op}: {'ok' if ok else 'FAIL'} {str(reason)[:300]}")
                    digest = _data_digest(data)
                    if digest:
                        lines.append(f"{pad}    diagnostics: {digest}")
                    if children:
                        _emit_steps(children, depth + 1)

            _emit_steps(getattr(inv, "steps", None))
        return "\n".join(lines)

    # ---- the reflection call ---------------------------------------------- #
    def propose(
        self,
        *,
        record: Optional[EpisodeRecord],
        object_hint: str,
        geometry: str,
        failed_invocation: Any,
        library_ids: Sequence[str],
    ) -> Optional[StrategyReflectionProposal]:
        """One reflection round. Returns a structurally-valid proposal or None
        (any failure -> None -> caller falls open to the deterministic library)."""
        content: list[dict] = [{
            "type": "text",
            "text": (
                "The previous recovery attempt failed; the judge has re-triggered. "
                "Trace of the failed attempt:\n\n"
                + self._trace_text(object_hint, geometry, failed_invocation, library_ids)
                + "\n\nTime-ordered frames of the recovery window follow."
            ),
        }]
        content.extend(self._frame_parts(record))
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            resp = asyncio.run(self.client.plan_chat(messages, response_schema_hint=False))
        except Exception as e:  # noqa: BLE001 - reflection must fail open
            self._log(f"reflector call failed ({type(e).__name__}: {e}); using library default")
            return None
        text = getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)
        obj = _extract_json(text)
        if obj is None:
            self._log("reflector returned no parseable JSON; using library default")
            return None
        try:
            proposal = StrategyReflectionProposal.from_dict(obj)
        except Exception as e:  # noqa: BLE001
            self._log(f"reflector proposal malformed ({type(e).__name__}: {e}); using library default")
            return None
        if proposal.strategy is not None:
            issues = [i for i in _check_strategy_structure(proposal) if not i.ok]
            if issues:
                self._log(
                    "reflector proposal rejected by structural gate: "
                    + "; ".join(f"{i.id}: {i.detail}" for i in issues[:4])
                )
                return None
        self._log(
            "reflector proposal accepted: "
            f"override={proposal.override_default} "
            f"strategy={getattr(proposal.strategy, 'strategy_id', None)} "
            f"rationale={proposal.rationale[:160]!r}"
        )
        return proposal
