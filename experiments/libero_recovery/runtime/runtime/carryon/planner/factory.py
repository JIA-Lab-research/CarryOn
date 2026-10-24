"""Construction of the planning stack.

Host-side wiring lifted out of ``main._amain``: build the planner VLM client (+
its I/O tracing logger), the :class:`ContextBuilder`, the :class:`Planner`
orchestrator, and the post-plan user-memory hook (:class:`UserSummarizer`, the
single UserMemory distillation writer). Returns a :class:`PlannerStack` bundle
so the host can wire the pieces into the executor / reactor / agent loop /
command handler.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Awaitable, Callable
from ..agent.context import ContextBuilder
from ..config import Config
from ..endpoints.registry import EndpointRegistry
from ..logging_util import EventLogger
from ..memory import SceneMemory, ShortTermMemory, UserMemory
from ..memory.user_summarizer import UserSummarizer
from .client import MockPlannerVLMClient, PlannerVLMClient, build_planner_vlm_client
from .planner import Planner

@dataclass
class PlannerStack:
    """The assembled planning components the host wires downstream.

    ``planner`` feeds the executor/reactor/agent loop; ``context_builder`` is
    shared with the agent loop; ``planner_vlm`` must be closed on shutdown;
    ``on_plan_finished`` is the executor's post-plan callback that triggers the
    (optional) user-profile summarizer — UserMemory's single distillation
    writer.
    """
    planner: Planner
    context_builder: ContextBuilder
    planner_vlm: PlannerVLMClient
    on_plan_finished: Callable[[object], Awaitable[None]]

def build_planner_stack(config: Config, *, registry: EndpointRegistry, memory: ShortTermMemory, scene_memory: SceneMemory | None, user_memory: UserMemory | None) -> PlannerStack:
    planner_vlm = build_planner_vlm_client(config)
    if isinstance(planner_vlm, MockPlannerVLMClient):
        planner_vlm._endpoint_names = registry.names
    planner_logger = EventLogger(path=config.planner_log_path, stdout=config.planner_log_stdout, channel='planner', default_fields={'provider': config.planner_vlm_provider, 'model': config.planner_vlm_model})
    from ..locations import load_locations
    context_builder = ContextBuilder(memory=memory, registry=registry, scene_memory=scene_memory, scene_top_k=config.scene_memory_inject_top_k, user_memory=user_memory, locations=load_locations(config.locations_config))
    planner = Planner(planner_vlm=planner_vlm, context_builder=context_builder, registry=registry, logger=planner_logger)
    user_summarizer: UserSummarizer | None = None
    if user_memory is not None:
        user_summarizer = UserSummarizer(planner_vlm=planner_vlm, user_memory=user_memory, user_id=config.user_id, min_messages=config.user_memory_min_messages_for_summary, enabled=config.user_memory_summarize_on_finish)

    async def _on_plan_finished(plan) -> None:
        if user_summarizer is None:
            return
        try:
            await user_summarizer.maybe_summarize(plan, memory.messages())
        except Exception:
            pass
    return PlannerStack(planner=planner, context_builder=context_builder, planner_vlm=planner_vlm, on_plan_finished=_on_plan_finished)
