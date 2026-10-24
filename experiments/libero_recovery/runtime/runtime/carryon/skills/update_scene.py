"""Write planner-established location facts into SceneMemory.

Accept {location, content} for one location, or a locations list containing
those fields for multiple locations. content is the markdown body to persist."""
from __future__ import annotations
from typing import TYPE_CHECKING
from ..types import ToolResult
if TYPE_CHECKING:
    from ..memory import SceneMemory

class UpdateSceneSkill:
    name = 'update_scene'

    def __init__(self, *, scene_memory: 'SceneMemory') -> None:
        self._scene = scene_memory

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        inputs = inputs or {}
        entries = self._collect_entries(inputs)
        if not entries:
            return ToolResult(ok=False, error='update_scene requires inputs.location + inputs.content (or inputs.locations: list of {location, content})')
        updated: list[str] = []
        for location, content in entries:
            self._scene.update_location(location, content)
            updated.append(location)
        return ToolResult(ok=True, data={'updated': updated, 'summary': f"scene memory updated: {', '.join(updated)}"})

    @staticmethod
    def _collect_entries(inputs: dict) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        many = inputs.get('locations')
        if isinstance(many, list):
            for item in many:
                if not isinstance(item, dict):
                    continue
                loc = str(item.get('location') or item.get('name') or '').strip()
                if loc:
                    out.append((loc, str(item.get('content') or '')))
        loc = str(inputs.get('location') or inputs.get('name') or '').strip()
        if loc:
            out.append((loc, str(inputs.get('content') or '')))
        return out
__all__ = ['UpdateSceneSkill']
