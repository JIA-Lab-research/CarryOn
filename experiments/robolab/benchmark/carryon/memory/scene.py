"""Store observations by location in markdown scene memory.

Each location is a section written through update_scene. The complete snapshot
is supplied to the planner as context."""

from __future__ import annotations

import re

from .markdown_store import MarkdownMemory


_IMAGE_B64_FIELD_RE = re.compile(
    r"(?P<prefix>(?P<key_quote>[\"'])image_b64(?P=key_quote)\s*:\s*"
    r"(?P<value_quote>[\"']))"
    r"(?P<payload>.*?)"
    r"(?P=value_quote)",
    re.DOTALL,
)
_DATA_URL_RE = re.compile(
    r"data:(?P<mime>[-\w.+/]+);base64,"
    r"(?P<payload>[A-Za-z0-9+/=\r\n]+)",
)


def _sanitize_content(content: str) -> str:
    """Remove inline image bytes before scene memory reaches disk/context."""
    text = str(content or "")

    def redact_field(match: re.Match[str]) -> str:
        payload = match.group("payload")
        quote = match.group("value_quote")
        return (
            f"{match.group('prefix')}<omitted {len(payload)} base64 chars>{quote}"
        )

    def redact_data_url(match: re.Match[str]) -> str:
        payload = match.group("payload")
        return (
            f"data:{match.group('mime')};base64,"
            f"<omitted {len(payload)} base64 chars>"
        )

    return _DATA_URL_RE.sub(redact_data_url, _IMAGE_B64_FIELD_RE.sub(redact_field, text))


class SceneMemory:
    """Markdown-backed per-location observation store."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._md = MarkdownMemory(store_path)

    # ----- public API (planner-driven writes) -----

    def update_location(self, location: str, content: str) -> None:
        """Create or replace the ``## <location>`` section with ``content``.

        ``content`` is whatever markdown the Planner composed (coords +
        description for a shelf, just a description for a drawer, ...). Inline
        base64 image payloads are redacted at this final persistence boundary so
        search observations cannot inflate the planner context.
        """
        location = (location or "").strip()
        if not location:
            raise ValueError("SceneMemory.update_location requires a location name")
        self._md.upsert_section(location, _sanitize_content(content))

    def remove_location(self, location: str) -> bool:
        return self._md.remove_section(location)

    def locations(self) -> "dict[str, str]":
        """``{location: body_markdown}`` for every recorded location."""
        return self._md.sections()

    def get_location(self, location: str) -> "str | None":
        return self._md.get_section(location)

    def is_empty(self) -> bool:
        return self._md.is_empty()

    def snapshot_for_planner(self, top_k: int | None = None) -> str:
        """Return the complete scene-memory markdown for the planner.

        The top_k argument is accepted but does not truncate the markdown."""
        return self._md.render()

    def render(self) -> str:
        return self._md.render()

    def clear(self) -> None:
        self._md.clear()


__all__ = ["SceneMemory"]
