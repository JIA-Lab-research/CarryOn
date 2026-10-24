"""SearchContainerSkill: the openable-container special case of search_locations.

A thin shell over :class:`SearchLocationsSkill` kept for backward compatibility
(and the narrow "open a drawer and look inside" case). It translates the
container-shaped inputs/outputs to/from the generic location form:

- ``inputs.containers[*].open_endpoint`` -> ``locations[*].reach_endpoint``
- result ``location`` -> ``container``; per-observation ``reach_*`` -> ``open_*``

The actual reach -> capture -> judge loop lives in :class:`SearchLocationsSkill`.
"""
from __future__ import annotations
from typing import Optional
from ..providers import SceneCapture
from ..types import ToolResult
from .invoker import EndpointInvoker
from .scene_describer import SceneDescriber
from .search_locations import SearchLocationsSkill

class SearchContainerSkill:
    name = 'search_container'

    def __init__(self, *, invoker: EndpointInvoker, scene_capture: SceneCapture, describer: SceneDescriber, default_open_timeout_s: float=60.0) -> None:
        self._inner = SearchLocationsSkill(invoker=invoker, scene_capture=scene_capture, describer=describer, default_reach_timeout_s=default_open_timeout_s)

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        inputs = inputs or {}
        containers = inputs.get('containers')
        if not isinstance(containers, list) or not containers:
            return ToolResult(ok=False, error='search_container requires inputs.containers: list of {name, open_endpoint, instruction?} entries')
        locations = []
        for entry in containers:
            if not isinstance(entry, dict):
                locations.append(entry)
                continue
            loc = dict(entry)
            loc['name'] = entry.get('name') or entry.get('container')
            loc['reach_endpoint'] = entry.get('open_endpoint')
            if entry.get('instruction'):
                loc['reach_instruction'] = entry.get('instruction')
            elif loc.get('name'):
                loc['reach_instruction'] = f"open {loc['name']}"
            if entry.get('open_inputs') is not None:
                loc['reach_inputs'] = entry.get('open_inputs')
            locations.append(loc)
        inner_inputs = {'target_object': inputs.get('target_object'), 'target_attributes': inputs.get('target_attributes'), 'reference_image_path': inputs.get('reference_image_path'), 'locations': locations}
        result = await self._inner.run(instruction, inner_inputs)
        if not result.ok:
            return result
        return ToolResult(ok=True, data=_to_container_shape(result.data or {}))

def _to_container_shape(data: dict) -> dict:
    observations = []
    for obs in data.get('observations') or []:
        if not isinstance(obs, dict):
            continue
        mapped = dict(obs)
        mapped['container'] = mapped.pop('location', '')
        mapped['open_endpoint'] = mapped.pop('reach_endpoint', '')
        mapped['open_result'] = mapped.pop('reach_result', {})
        observations.append(mapped)
    return {'target_object': data.get('target_object'), 'found': data.get('found', False), 'container': data.get('location'), 'observations': observations}
__all__ = ['SearchContainerSkill']
