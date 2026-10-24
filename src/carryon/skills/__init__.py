"""In-process skills callable via the ``local`` transport.

Skills are Python objects implementing :class:`BaseSkill`. They are registered
into a :class:`SkillRegistry` and dispatched by
:class:`carryon.transports.local.LocalTransport` when the Planner picks an
endpoint whose ``transport: local`` and ``address`` matches a registered
skill name.

Distinct from :class:`carryon.endpoints.RobotEndpoint` only in *where* the
work runs: skills run inside the CarryOn process and may orchestrate other
endpoints via :class:`EndpointInvoker`; endpoints are external VLA services.
"""

from .base import BaseSkill, SkillRegistry
from .factory import build_endpoint_skill_stack, build_skill_registry
from .invoker import EndpointInvoker
from .update_scene import UpdateSceneSkill
from .update_user_memory import UpdateUserMemorySkill
from .wait import WaitSkill
from .web_query import WebQuerySkill

__all__ = [
    "BaseSkill",
    "EndpointInvoker",
    "SkillRegistry",
    "UpdateSceneSkill",
    "UpdateUserMemorySkill",
    "WaitSkill",
    "WebQuerySkill",
    "build_endpoint_skill_stack",
    "build_skill_registry",
]
