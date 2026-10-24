"""Build local skills and endpoint-backed skills.

Local skills are assembled first. Endpoint-backed skills are attached after
transport and result routing are available."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Config
from .base import SkillRegistry
from .invoker import EndpointInvoker
from .update_scene import UpdateSceneSkill
from .update_user_memory import UpdateUserMemorySkill
from .wait import WaitSkill
from .web_query import WebQuerySkill

if TYPE_CHECKING:
    from ..endpoints import EndpointRegistry
    from ..jobs.channels.base import ResultChannel
    from ..memory import SceneMemory, UserMemory
    from ..transports.base import Transport


def build_skill_registry(
    config: Config,
    *,
    scene_memory: "SceneMemory | None" = None,
    user_memory: "UserMemory | None" = None,
) -> SkillRegistry:
    """Register the always-on, dependency-free in-process skills.

    Called *before* ``build_transports`` because ``LocalTransport`` resolves
    skills out of this registry by name. Endpoint-orchestrating skills are
    added later by :func:`build_endpoint_skill_stack`.

    Planner-driven memory writes (``update_scene`` / ``update_user_memory``)
    are registered here when their tiers are enabled -- they only need the
    memory store, not transports/result channel.
    """
    skill_registry = SkillRegistry()
    skill_registry.register(WaitSkill())
    skill_registry.register(WebQuerySkill())
    if scene_memory is not None:
        skill_registry.register(UpdateSceneSkill(scene_memory=scene_memory))
    if user_memory is not None:
        skill_registry.register(
            UpdateUserMemorySkill(
                user_memory=user_memory,
                default_user_id=config.user_id,
            )
        )
    return skill_registry


def build_endpoint_skill_stack(
    config: Config,
    *,
    registry: "EndpointRegistry",
    transports: dict[str, "Transport"],
    result_channel: "ResultChannel",
) -> EndpointInvoker:
    """Build the :class:`EndpointInvoker`.

    The invoker lets in-process skills call real robot endpoints without going
    through the Executor's DAG; it needs ``transports`` and the result channel,
    so this runs *after* :func:`build_skill_registry` / ``build_transports``.

    No endpoint-orchestrating skill is registered here any more. Revealing what
    a place holds is ordinary subtasks the planner composes on the endpoints it
    already has; the only scene-memory endpoint is the ``update_scene`` WRITE
    skill, registered in :func:`build_skill_registry`. See
    ``carryon.agent.context.SCENE_MEMORY_POLICY``.
    """
    return EndpointInvoker(
        registry=registry,
        transports=transports,
        callback_url=result_channel.external_url("__skill__"),
        default_deadline_s=config.job_default_deadline_s,
    )
