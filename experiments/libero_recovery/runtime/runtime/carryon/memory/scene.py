"""Store observations by location in markdown scene memory.

Each location is a section written through update_scene. The complete snapshot
is supplied to the planner as context."""
from __future__ import annotations
from .markdown_store import MarkdownMemory

class SceneMemory:
    """Markdown-backed per-location observation store."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._md = MarkdownMemory(store_path)

    def update_location(self, location: str, content: str) -> None:
        """Create or replace the ``## <location>`` section with ``content``.

        ``content`` is whatever markdown the Planner composed (coords +
        description for a shelf, just a description for a drawer, ...).
        """
        location = (location or '').strip()
        if not location:
            raise ValueError('SceneMemory.update_location requires a location name')
        self._md.upsert_section(location, content or '')

    def remove_location(self, location: str) -> bool:
        return self._md.remove_section(location)

    def locations(self) -> 'dict[str, str]':
        """``{location: body_markdown}`` for every recorded location."""
        return self._md.sections()

    def get_location(self, location: str) -> 'str | None':
        return self._md.get_section(location)

    def is_empty(self) -> bool:
        return self._md.is_empty()

    def snapshot_for_planner(self, top_k: int | None=None) -> str:
        """Return the complete scene-memory markdown for the planner.

        The top_k argument is accepted but does not truncate the markdown."""
        return self._md.render()

    def render(self) -> str:
        return self._md.render()

    def clear(self) -> None:
        self._md.clear()
__all__ = ['SceneMemory']
