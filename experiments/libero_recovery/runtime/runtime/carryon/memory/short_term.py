from __future__ import annotations
from typing import Protocol
from ..types import Message

class Memory(Protocol):
    """The conversation memory: an append-only, ts-ordered message log."""

    def append(self, msg: Message) -> None:
        ...

    def messages(self) -> list[Message]:
        ...

    def clear(self) -> None:
        ...

    def snapshot(self) -> dict:
        ...

class SubtaskSceneStore(Protocol):
    """A side-channel that stashes the camera snapshots captured right before a
    subtask started, keyed by ``(plan_id, subtask_id)``.

    Kept separate from :class:`Memory` (interface segregation): only the
    VLM-Judge / Reactor scene-diff path needs it, so the core message log is
    not burdened with camera storage. The VLM Judge supervisor writes via
    :meth:`record_subtask_start_scene`; the Reactor reads "before" frames via
    :meth:`get_subtask_start_scene` (treating it as an optional capability).
    """

    def record_subtask_start_scene(self, plan_id: str, subtask_id: str, snapshots: list[dict]) -> None:
        ...

    def get_subtask_start_scene(self, plan_id: str, subtask_id: str) -> list[dict]:
        ...

class ShortTermMemory:
    """Simple in-memory store. Reads are ts-sorted so out-of-order appends are OK.

    Implements both :class:`Memory` (the message log) and
    :class:`SubtaskSceneStore` (the camera-snapshot side-channel): it keeps the
    snapshots captured right before each subtask started, keyed by
    ``(plan_id, subtask_id)``, held separately from ``messages()`` to avoid
    bloating planner context.
    """

    def __init__(self, max_tokens: int=32000) -> None:
        self.max_tokens = max_tokens
        self._msgs: list[Message] = []
        self._scene_snapshots: dict[tuple[str, str], list[dict]] = {}

    def append(self, msg: Message) -> None:
        self._msgs.append(msg)

    def messages(self) -> list[Message]:
        return sorted(self._msgs, key=lambda m: (m.ts, m.plan_id or '', m.subtask_id or ''))

    def clear(self) -> None:
        self._msgs.clear()
        self._scene_snapshots.clear()

    def snapshot(self) -> dict:
        return {'messages': [{'role': m.role, 'content': m.content, 'endpoint': m.endpoint, 'subtask_id': m.subtask_id, 'plan_id': m.plan_id, 'ts': m.ts, 'metadata': m.metadata} for m in self.messages()], 'scene_snapshots': {f'{plan_id}:{subtask_id}': [{'camera': s.get('camera'), 'mime': s.get('mime'), 'ts': s.get('ts'), 'image_b64_len': len(s.get('image_b64') or '')} for s in snaps] for (plan_id, subtask_id), snaps in self._scene_snapshots.items()}}

    def tail(self, n: int) -> list[Message]:
        return self.messages()[-n:]

    def record_subtask_start_scene(self, plan_id: str, subtask_id: str, snapshots: list[dict]) -> None:
        """Store the camera snapshots captured right before ``subtask_id`` started.

        Each snapshot is expected to be a dict with at least ``camera``,
        ``image_b64``, ``mime`` (the shape produced by
        :meth:`carryon.providers.SceneSnapshot.to_dict`).
        """
        if not plan_id or not subtask_id:
            return
        self._scene_snapshots[plan_id, subtask_id] = [dict(s) for s in snapshots or []]

    def get_subtask_start_scene(self, plan_id: str, subtask_id: str) -> list[dict]:
        return [dict(s) for s in self._scene_snapshots.get((plan_id, subtask_id), [])]
