"""Construction of the three-tier memory subsystem.

Host-side wiring lifted out of ``main._amain``: build the in-process
:class:`ShortTermMemory` plus the persistent :class:`SceneMemory` /
:class:`UserMemory` tiers (each gated by its ``*_enabled`` config flag), rooted
under ``config.memory_data_dir`` (created on demand).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import Config
from .scene import SceneMemory
from .short_term import ShortTermMemory
from .user import UserMemory


@dataclass
class MemoryBundle:
    """The three memory tiers the host wires downstream.

    ``scene`` / ``user`` are ``None`` when disabled in config.
    """

    short_term: ShortTermMemory
    scene: SceneMemory | None
    user: UserMemory | None


def build_memory(config: Config) -> MemoryBundle:
    short_term = ShortTermMemory(max_tokens=config.memory_max_tokens)

    data_dir = config.memory_data_dir
    os.makedirs(data_dir, exist_ok=True)

    scene: SceneMemory | None = None
    if config.scene_memory_enabled:
        scene = SceneMemory(
            store_path=os.path.join(data_dir, "scene.md"),
        )

    user: UserMemory | None = None
    if config.user_memory_enabled:
        user = UserMemory(
            store_dir=os.path.join(data_dir, "user"),
            default_user_id=config.user_id,
        )

    return MemoryBundle(short_term=short_term, scene=scene, user=user)
