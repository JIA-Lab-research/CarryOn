"""Planning domain: the Planner orchestrator and its Planner VLM model client.

``Planner`` (the orchestrator) is exposed lazily via :pep:`562` ``__getattr__``
rather than imported eagerly: it pulls in ``agent.context -> memory``, whose
``user_summarizer`` imports ``PlannerVLMResponse`` back from this package.
Loading only the leaf ``.client`` / ``.schema`` at package init keeps that
resolution cycle-free; ``from carryon.planner import Planner`` still works
(it triggers the lazy import once memory is fully loaded).
"""
from .client import MockPlannerVLMClient, OpenRouterPlannerVLMClient, PlannerVLMClient, PlannerVLMResponse, build_planner_vlm_client, build_reflection_vlm_client, resolve_reflection_vlm
from .schema import PLAN_SCHEMA, parse_and_validate_plan
__all__ = ['PLAN_SCHEMA', 'MockPlannerVLMClient', 'OpenRouterPlannerVLMClient', 'Planner', 'PlannerVLMClient', 'PlannerVLMResponse', 'build_planner_vlm_client', 'build_reflection_vlm_client', 'parse_and_validate_plan', 'resolve_reflection_vlm']

def __getattr__(name: str):
    if name == 'Planner':
        from .planner import Planner
        return Planner
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
