"""Built-in ``check_endpoint`` skill: is this object visible at THAT endpoint?

The tool half of routing. Where ``web_query`` answers "what does the internet
say", this answers "what does that workspace's camera show" for ONE named
endpoint at a time: given an endpoint id and a description of a target, it
fetches that endpoint's own camera frames and asks the VLM whether the target
is there. The planner uses it to turn a *candidate* endpoint (proposed from the
endpoint description's household role, or from a user-memory fact) into a
*confirmed* one before dispatching a manipulation to it.

Two invariants this skill does NOT share with the VLM Judge:

1. **Camera scoping fails CLOSED.** ``judge.supervisor.cameras_for_endpoint``
   deliberately falls back to the full configured camera set when an endpoint
   declares no (or a malformed) ``cameras`` list -- for the judge that is
   "rather than judging blind". For a presence check the same fallback is a
   correctness hazard: checking endpoint A while looking through endpoint B's
   camera reports B's object as present at A, and the answer *looks* perfectly
   reasonable. So this skill refuses instead of widening. See
   :func:`scoped_cameras_for_check`.
2. **No ground truth reaches this skill.** The verdict is computed from frames
   only. Whether a check was *right* is decided offline by the analyzer, which
   is the only place that may hold the expected endpoint. Never pass an
   expected/answer field in ``inputs``.

Returned ``ToolResult.data``::

    {"present": bool, "confidence": float, "evidence": str,
     "endpoint": str, "cameras_used": [str], "frames": int,
     "capture_ms": int, "vlm_ms": int}
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from ..types import ToolResult

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You inspect robot workspace camera frames and answer ONE question: is the "
    "described target object currently visible in this workspace?\n"
    "Rules:\n"
    "- Judge ONLY from the frames given. Never guess from the object's name, "
    "from where such an object usually lives, or from what a room is for.\n"
    "- The frames all show ONE workspace. Answer about THIS workspace only.\n"
    "- If you cannot clearly see the target, answer present=false. A missed "
    "object costs one more look; a false sighting sends the robot to the wrong "
    "place and it never recovers.\n"
    "- Match on the described attributes (colour, material, shape, print). A "
    "different object of the same category is NOT the target.\n"
    'Reply with JSON only: {"present": true|false, "confidence": 0.0-1.0, '
    '"evidence": "<one short sentence naming what you saw>"}'
)


class CameraScopeError(ValueError):
    """Raised when an endpoint's camera scope cannot be resolved fail-closed."""


def scoped_cameras_for_check(
    cameras: list[dict[str, Any]] | None,
    endpoint: Any,
) -> list[str]:
    """Resolve the camera names covering ``endpoint``, or raise.

    Unlike the judge's :func:`cameras_for_endpoint`, every failure mode here is
    an error rather than a fallback to the full set:

    - the endpoint declares no ``cameras`` list, or it is empty/malformed;
    - a declared name matches no configured camera (a typo must not silently
      narrow the scope either -- a check run through the wrong subset is as
      wrong as one run through the full set).

    Returning fewer-but-still-foreign cameras is the failure this guards: the
    caller would get a confident, plausible, wrong answer.
    """
    name = getattr(endpoint, "name", "?")
    declared = (getattr(endpoint, "extra", None) or {}).get("cameras")
    if not isinstance(declared, list) or not declared:
        raise CameraScopeError(
            f"endpoint {name!r} declares no 'cameras' list; check_endpoint "
            f"refuses to fall back to the full camera set (it would inspect "
            f"another workspace and report its objects as present here)"
        )
    configured = {
        str(c.get("name") or "")
        for c in (cameras or [])
        if isinstance(c, dict) and c.get("snapshot_url")
    }
    wanted = [str(n) for n in declared]
    missing = [n for n in wanted if n not in configured]
    if missing:
        raise CameraScopeError(
            f"endpoint {name!r}: camera(s) {missing!r} are not configured "
            f"(configured: {sorted(configured)!r}); refusing to check through "
            f"a partial scope"
        )
    return wanted


class CheckEndpointSkill:
    name = "check_endpoint"

    def __init__(
        self,
        *,
        registry: Any,
        cameras: list[dict[str, Any]] | None,
        scene_capture: Any,
        chat_client: Any,
        model: str,
        max_edge_px: int | None = 768,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self._registry = registry
        self._cameras = list(cameras or [])
        self._capture = scene_capture
        self._chat = chat_client
        self._model = model
        self._max_edge_px = max_edge_px
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        started = time.time()
        raw = inputs or {}

        endpoint_name = str(raw.get("endpoint") or "").strip()
        if not endpoint_name:
            return ToolResult(
                ok=False, error="check_endpoint requires inputs.endpoint"
            )

        target = str(
            raw.get("target_object") or raw.get("target") or instruction or ""
        ).strip()
        if not target:
            return ToolResult(
                ok=False,
                error="check_endpoint requires inputs.target_object (or a "
                "non-empty instruction)",
            )

        try:
            endpoint = self._registry.get(endpoint_name)
        except Exception as e:  # unknown endpoint -> fail closed
            return ToolResult(
                ok=False, error=f"check_endpoint: unknown endpoint "
                f"{endpoint_name!r} ({e})"
            )

        try:
            camera_names = scoped_cameras_for_check(self._cameras, endpoint)
        except CameraScopeError as e:
            return ToolResult(ok=False, error=str(e))

        t0 = time.time()
        snapshots = await self._capture.capture(camera_names=camera_names)
        capture_ms = int((time.time() - t0) * 1000)
        if not snapshots:
            return ToolResult(
                ok=False,
                error=f"check_endpoint: no frame reachable for endpoint "
                f"{endpoint_name!r} via cameras {camera_names!r}; refusing to "
                f"answer without evidence",
                latency_ms=int((time.time() - started) * 1000),
            )

        parts: list[dict] = [{"type": "text", "text": _describe(target, raw)}]
        for snap in snapshots:
            parts.append(snap.to_image_url_part(max_edge_px=self._max_edge_px))

        t1 = time.time()
        try:
            result = await self._chat.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": parts},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return ToolResult(
                ok=False,
                error=f"check_endpoint: VLM call failed: {e}",
                latency_ms=int((time.time() - started) * 1000),
            )
        vlm_ms = int((time.time() - t1) * 1000)

        verdict = _parse_verdict(getattr(result, "text", "") or "")
        if verdict is None:
            return ToolResult(
                ok=False,
                error="check_endpoint: could not parse a JSON verdict from the "
                "VLM reply",
                raw=getattr(result, "text", ""),
                latency_ms=int((time.time() - started) * 1000),
            )

        present, confidence, evidence = verdict
        logger.info(
            "check_endpoint endpoint=%s target=%r present=%s conf=%.2f "
            "frames=%d capture_ms=%d vlm_ms=%d",
            endpoint_name, target[:48], present, confidence, len(snapshots),
            capture_ms, vlm_ms,
        )
        # A confirmed sighting is an observation that scene memory does not
        # hold yet. Flag it the same way the judge flags a reveal, so the loop
        # owes a record and the parser collects it. The description is the
        # target as THE PLANNER named it and the station as it named it --
        # nothing here knows what the object "really" is.
        revealed = (f"{target} is present at {endpoint_name}: {evidence}"
                    if present else
                    f"{target} is not present at {endpoint_name}: {evidence}")
        return ToolResult(
            ok=True,
            data={
                "present": present,
                "confidence": confidence,
                "evidence": evidence,
                "endpoint": endpoint_name,
                "target_object": target,
                "cameras_used": [s.camera for s in snapshots],
                "frames": len(snapshots),
                "capture_ms": capture_ms,
                "vlm_ms": vlm_ms,
                "scene_update_required": bool(present),
                "revealed_contents": revealed if present else None,
                "revealed_place": endpoint_name,
            },
            latency_ms=int((time.time() - started) * 1000),
            raw=getattr(result, "raw", None),
        )


def _describe(target: str, raw: dict) -> str:
    lines = [
        "Workspace under inspection: the one shown in the frame(s) below.",
        f"Target object: {target}",
    ]
    attrs = raw.get("target_attributes")
    if isinstance(attrs, dict) and attrs:
        rendered = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(f"Target attributes: {rendered}")
    elif isinstance(attrs, str) and attrs.strip():
        lines.append(f"Target attributes: {attrs.strip()}")
    lines.append("Is this target visible in this workspace right now?")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_verdict(text: str) -> tuple[bool, float, str] | None:
    """Extract ``(present, confidence, evidence)`` from a VLM reply."""
    body = text.strip()
    if not body:
        return None
    fenced = _FENCE_RE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    obj: Any = None
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(body[start:end + 1])
            except (ValueError, TypeError):
                return None
    if not isinstance(obj, dict) or "present" not in obj:
        return None

    present = obj.get("present")
    if isinstance(present, str):
        present = present.strip().lower() in {"true", "yes", "1"}
    present = bool(present)

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = str(obj.get("evidence") or "").strip()
    return present, confidence, evidence
