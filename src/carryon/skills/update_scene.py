"""Write planner-established facts into SceneMemory.

Facts may come from the current user instruction or from scene observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import ToolResult

if TYPE_CHECKING:
    from ..memory import SceneMemory


class UpdateSceneSkill:
    name = "update_scene"

    def __init__(self, *, scene_memory: "SceneMemory") -> None:
        self._scene = scene_memory

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        inputs = inputs or {}
        entries = self._collect_entries(inputs)
        if not entries:
            return ToolResult(
                ok=False,
                error=(
                    "update_scene requires inputs.place + inputs.content "
                    "(or inputs.places: list of {place, content})"
                ),
            )
        updated: list[str] = []
        for location, content in entries:
            self._scene.update_location(location, content)
            updated.append(location)
        return ToolResult(
            ok=True,
            data={"updated": updated, "summary": f"scene memory updated: {', '.join(updated)}"},
        )

    @staticmethod
    def _place_name(item: dict) -> str:
        """``place`` is the documented key; the older names still resolve."""
        for key in ("place", "location", "name"):
            value = str(item.get(key) or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _collect_entries(inputs: dict) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        many = inputs.get("places")
        if not isinstance(many, list):
            many = inputs.get("locations")
        if isinstance(many, list):
            for item in many:
                if not isinstance(item, dict):
                    continue
                loc = UpdateSceneSkill._place_name(item)
                if loc:
                    out.append((loc, UpdateSceneSkill._entry_content(item)))
        loc = UpdateSceneSkill._place_name(inputs)
        if loc:
            out.append((loc, UpdateSceneSkill._entry_content(inputs)))
        return out

    @staticmethod
    def _entry_content(item: dict) -> str:
        """Markdown body for one place entry.

        Explicit ``content`` wins. Without it, synthesize from observation
        fields (``objects`` / ``description``), so a batch of per-place
        observations passed straight through as a list still persists
        something meaningful instead of an empty body.
        """
        content = str(item.get("content") or "")
        if content:
            return content
        lines: list[str] = []
        objects = item.get("objects")
        if isinstance(objects, list) and objects:
            lines.append("- objects: " + ", ".join(str(o) for o in objects))
        description = str(item.get("description") or "").strip()
        if description:
            lines.append(f"- description: {description}")
        return "\n".join(lines)


__all__ = ["UpdateSceneSkill"]
