"""SearchLocationsSkill: visit a list of locations, look, judge, stop on hit.

The generalization of :class:`SearchContainerSkill` from "openable containers"
to arbitrary *locations*. Each location carries a ``reach_endpoint`` -- the
RobotEndpoint that physically brings the robot to / opens the location:

- a shelf reaches via the quadruped navigator (``go2_navigator``),
- a drawer reaches via its ``open_drawer_N`` endpoint.

Per location the skill runs three steps:

1. Reach the location via :class:`EndpointInvoker` (navigate or open).
2. Capture multi-camera snapshots via :class:`SceneCapture`.
3. Ask the :class:`SceneDescriber` judge whether the target is present, passing
   the target text and (optionally) the user's reference image.

If the target is found the search short-circuits. Either way every location
visited is returned in ``observations`` so the Planner can persist them via the
``update_scene`` skill (writes are planner-driven; this skill does not write
SceneMemory itself).
"""
from __future__ import annotations
import base64
import os
from dataclasses import asdict, dataclass, field
from typing import Optional
from ..providers import SceneCapture, SceneSnapshot
from ..types import ToolResult
from .invoker import EndpointInvoker
from .scene_describer import SceneDescriber, SceneDescription

@dataclass
class SearchObservation:
    location: str
    reach_endpoint: str
    objects: list[str] = field(default_factory=list)
    description: str = ''
    snapshots: list[dict] = field(default_factory=list)
    reach_result: dict = field(default_factory=dict)
    target_present: bool = False
    matched_object: str = ''
    confidence: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

class SearchLocationsSkill:
    name = 'search_locations'

    def __init__(self, *, invoker: EndpointInvoker, scene_capture: SceneCapture, describer: SceneDescriber, default_reach_timeout_s: float=120.0) -> None:
        self._invoker = invoker
        self._scene_capture = scene_capture
        self._describer = describer
        self._default_reach_timeout_s = default_reach_timeout_s

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        inputs = inputs or {}
        target_object = inputs.get('target_object')
        target_norm = str(target_object).strip().lower() if target_object else None
        target_attributes = inputs.get('target_attributes')
        target_text = _join_target(target_object, target_attributes)
        reference_image = _load_reference_image(inputs.get('reference_image_path'))
        locations = inputs.get('locations') or []
        if not isinstance(locations, list) or not locations:
            return ToolResult(ok=False, error='search_locations requires inputs.locations: list of {name, reach_endpoint, reach_instruction?, reach_inputs?} entries')
        observations: list[SearchObservation] = []
        found_location: Optional[str] = None
        for entry in locations:
            if not isinstance(entry, dict):
                observations.append(SearchObservation(location=str(entry), reach_endpoint='', error='location entry must be a dict'))
                continue
            name = str(entry.get('name') or entry.get('location') or '').strip()
            reach_endpoint = str(entry.get('reach_endpoint') or '').strip()
            reach_instruction = str(entry.get('reach_instruction') or entry.get('instruction') or f"go to {name or 'the location'}")
            if not name or not reach_endpoint:
                observations.append(SearchObservation(location=name, reach_endpoint=reach_endpoint, error='missing name or reach_endpoint'))
                continue
            timeout_s = float(entry.get('timeout_s') or self._default_reach_timeout_s)
            obs = await self._search_one(location=name, reach_endpoint=reach_endpoint, reach_instruction=reach_instruction, reach_inputs=entry.get('reach_inputs') or entry.get('inputs') or {}, timeout_s=timeout_s, target_norm=target_norm, target_text=target_text, reference_image=reference_image)
            observations.append(obs)
            if obs.target_present:
                found_location = name
                break
        data = {'target_object': target_object, 'found': found_location is not None, 'location': found_location, 'observations': [o.to_dict() for o in observations]}
        return ToolResult(ok=True, data=data)

    async def _search_one(self, *, location: str, reach_endpoint: str, reach_instruction: str, reach_inputs: dict, timeout_s: float, target_norm: Optional[str], target_text: Optional[str], reference_image: Optional[SceneSnapshot]) -> SearchObservation:
        payload = await self._invoker.call(reach_endpoint, reach_instruction, reach_inputs, timeout_s=timeout_s)
        status = str(payload.get('status') or '')
        if status != 'success':
            return SearchObservation(location=location, reach_endpoint=reach_endpoint, reach_result=payload, error=payload.get('error') or f'reach failed: status={status!r}')
        snaps = await self._scene_capture.capture()
        snapshots = [s.to_dict() for s in snaps]
        description: SceneDescription = await self._describer.describe(snaps, target_text=target_text, reference_image=reference_image, location_name=location, focus_object=target_norm)
        target_present = bool(description.found) or bool(target_norm and any((target_norm in obj for obj in description.objects)))
        return SearchObservation(location=location, reach_endpoint=reach_endpoint, objects=list(description.objects), description=description.description, snapshots=snapshots, reach_result=payload, target_present=target_present, matched_object=description.matched_object, confidence=description.confidence)

def _join_target(target_object, target_attributes) -> Optional[str]:
    obj = str(target_object).strip() if target_object else ''
    attrs = str(target_attributes).strip() if target_attributes else ''
    joined = ' '.join((p for p in (attrs, obj) if p)).strip()
    return joined or None
_EXT_TO_MIME = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif'}

def _load_reference_image(path) -> Optional[SceneSnapshot]:
    if not path or not isinstance(path, str) or (not os.path.isfile(path)):
        return None
    try:
        with open(path, 'rb') as f:
            body = f.read()
    except OSError:
        return None
    if not body:
        return None
    mime = _EXT_TO_MIME.get(os.path.splitext(path)[1].lower(), 'image/jpeg')
    return SceneSnapshot(camera='reference', image_b64=base64.b64encode(body).decode('ascii'), mime=mime)
__all__ = ['SearchLocationsSkill', 'SearchObservation']
