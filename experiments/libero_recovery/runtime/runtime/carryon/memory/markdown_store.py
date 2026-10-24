"""MarkdownMemory: a tiny section-structured markdown file store.

The shared on-disk substrate for the planner-driven memory tiers
(:class:`carryon.memory.scene.SceneMemory` and
:class:`carryon.memory.user.UserMemory`). A single ``.md`` file holds an
ordered list of ``## <section>`` blocks; each block's body is free-form
markdown the Planner (or a memory writer) composes. Keeping the store as plain
markdown means:

- it is readable / hand-editable,
- different sections may carry different fields (a shelf section can list
  ``coords`` while a drawer section only carries a ``description``), and
- the whole file can be injected verbatim into the Planner's system prompt.

This class only knows about *sections* (title + body text). The meaning of a
section (a location, a user-fact category, ...) is owned by the caller.
Persistence is append-or-replace with an atomic ``tmp -> os.replace`` write.
"""
from __future__ import annotations
import os
import threading

class MarkdownMemory:
    """An ordered ``{section_title: body}`` map persisted as one markdown file."""

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        self._sections: 'dict[str, str]' = {}
        self._lock = threading.RLock()
        self._load()

    def get_section(self, title: str) -> 'str | None':
        title = (title or '').strip()
        with self._lock:
            return self._sections.get(title)

    def sections(self) -> 'dict[str, str]':
        with self._lock:
            return dict(self._sections)

    def upsert_section(self, title: str, body: str) -> None:
        """Create or replace a section, preserving insertion order."""
        title = (title or '').strip()
        if not title:
            raise ValueError('MarkdownMemory.upsert_section requires a title')
        with self._lock:
            self._sections[title] = (body or '').strip()
            self._save_locked()

    def replace_all(self, sections: 'dict[str, str]') -> None:
        """Replace the entire file with ``sections`` (ordered), then persist."""
        with self._lock:
            self._sections = {str(k).strip(): (str(v) if v is not None else '').strip() for k, v in (sections or {}).items() if str(k).strip()}
            self._save_locked()

    def remove_section(self, title: str) -> bool:
        title = (title or '').strip()
        with self._lock:
            existed = self._sections.pop(title, None) is not None
            if existed:
                self._save_locked()
            return existed

    def render(self) -> str:
        with self._lock:
            return _render(self._sections)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._sections

    def clear(self) -> None:
        with self._lock:
            self._sections.clear()
            self._save_locked()

    def _load(self) -> None:
        if not self.store_path or not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except OSError:
            return
        with self._lock:
            self._sections = _parse(text)

    def _save_locked(self) -> None:
        if not self.store_path:
            return
        os.makedirs(os.path.dirname(self.store_path) or '.', exist_ok=True)
        tmp = f'{self.store_path}.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(_render(self._sections))
            os.replace(tmp, self.store_path)
        except Exception:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            raise

def _parse(text: str) -> 'dict[str, str]':
    """Split markdown into ``{section_title: body}`` on ``## `` headings.

    Any preamble before the first ``## `` heading is ignored. Bodies are
    stripped of surrounding blank lines.
    """
    sections: 'dict[str, str]' = {}
    current: 'str | None' = None
    buf: list[str] = []
    for line in (text or '').splitlines():
        if line.startswith('## '):
            if current is not None:
                sections[current] = '\n'.join(buf).strip()
            current = line[3:].strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = '\n'.join(buf).strip()
    return sections

def _render(sections: 'dict[str, str]') -> str:
    if not sections:
        return ''
    blocks: list[str] = []
    for title, body in sections.items():
        block = f'## {title}'
        if body:
            block += f'\n{body}'
        blocks.append(block)
    return '\n\n'.join(blocks) + '\n'
__all__ = ['MarkdownMemory']
