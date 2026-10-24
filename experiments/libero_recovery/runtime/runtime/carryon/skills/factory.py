"""Build local skills and endpoint-backed skills.

Local skills are assembled first. Endpoint-backed skills are attached after
transport and result routing are available."""
from __future__ import annotations
from typing import TYPE_CHECKING
from ..config import Config
from .base import SkillRegistry
from .invoker import EndpointInvoker
from .search_container import SearchContainerSkill
from .search_locations import SearchLocationsSkill
from .update_scene import UpdateSceneSkill
from .wait import WaitSkill
from .web_query import WebQuerySkill
if TYPE_CHECKING:
    from ..endpoints import EndpointRegistry
    from ..jobs.channels.base import ResultChannel
    from ..memory import SceneMemory
    from ..providers import SceneCapture
    from ..transports.base import Transport
    from .scene_describer import SceneDescriber

def build_skill_registry(config: Config, *, scene_memory: 'SceneMemory | None'=None) -> SkillRegistry:
    """Register the always-on, dependency-free in-process skills.

    Called *before* ``build_transports`` because ``LocalTransport`` resolves
    skills out of this registry by name. Endpoint-orchestrating skills are
    added later by :func:`build_endpoint_skill_stack`.

    The planner-driven scene-write skill (``update_scene``) is registered here
    too -- it only needs the memory tier, not the transports/result channel --
    gated on that tier being enabled. UserMemory has no planner-facing write
    skill: its single writer is the post-plan UserSummarizer.
    """
    skill_registry = SkillRegistry()
    skill_registry.register(WaitSkill())
    skill_registry.register(WebQuerySkill())
    if scene_memory is not None:
        skill_registry.register(UpdateSceneSkill(scene_memory=scene_memory))
    return skill_registry

def build_endpoint_skill_stack(config: Config, *, skill_registry: SkillRegistry, registry: 'EndpointRegistry', transports: dict[str, 'Transport'], result_channel: 'ResultChannel', scene_memory: 'SceneMemory | None', scene_capture: 'SceneCapture | None') -> EndpointInvoker:
    """Build the :class:`EndpointInvoker` and register endpoint-orchestrating
    skills onto ``skill_registry``.

    The invoker lets in-process skills call real robot endpoints without going
    through the Executor's DAG; it needs ``transports`` and the result channel,
    so this runs *after* :func:`build_skill_registry` / ``build_transports``.

    ``search_container`` is registered only when scene memory is enabled AND a
    camera source (``scene_capture``) exists -- ``scene_capture`` is built iff
    ``planner_vlm_cameras`` is non-empty, so the two conditions together gate
    the skill on "we can actually look inside a container".
    """
    invoker = EndpointInvoker(registry=registry, transports=transports, callback_url=result_channel.external_url('__skill__'), default_deadline_s=config.job_default_deadline_s)
    if scene_memory is not None and scene_capture is not None:
        describer = _build_scene_describer(config)
        skill_registry.register(SearchContainerSkill(invoker=invoker, scene_capture=scene_capture, describer=describer))
        skill_registry.register(SearchLocationsSkill(invoker=invoker, scene_capture=scene_capture, describer=describer, default_reach_timeout_s=config.job_default_deadline_s))
    return invoker

def _build_scene_describer(config: Config) -> 'SceneDescriber':
    """Pick the SceneDescriber backend.

    Prefer the planner VLM's OpenRouter creds when available; otherwise fall
    back to a deterministic mock so tests / no-API setups still work.
    """
    from .scene_describer import MockSceneDescriber, OpenRouterSceneDescriber
    if config.planner_vlm_provider == 'openrouter' and config.planner_vlm_api_key:
        return OpenRouterSceneDescriber(api_key=config.planner_vlm_api_key, model=config.planner_vlm_model, base_url=config.planner_vlm_base_url, timeout_s=config.planner_vlm_timeout_s)
    fixtures = {}
    raw = getattr(config, 'scene_describer_mock_locations', None)
    if isinstance(raw, dict):
        fixtures = {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return MockSceneDescriber(locations=fixtures)
