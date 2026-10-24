from __future__ import annotations
from typing import Protocol
from ..types import ToolResult

class BaseSkill(Protocol):
    """Interface every in-process skill must satisfy.

    ``name`` is the registry key used by ``RobotEndpoint.address`` when the
    endpoint declares ``transport: local``.

    ``run`` performs the skill and returns a :class:`ToolResult`. The
    LocalTransport translates this into a ``CallbackPayload`` (see
    ``carryon.types``) on the result channel. Skills MUST cooperate with
    ``asyncio.CancelledError`` so ``LocalTransport.cancel`` works.
    """
    name: str

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        ...

class SkillRegistry:
    """Simple name -> BaseSkill map.

    Mirrors :class:`carryon.endpoints.EndpointRegistry` so wiring code in
    ``main.py`` reads similarly; no observer / event semantics are needed
    since the local transport always looks up skills by name on each send.
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        name = getattr(skill, 'name', None)
        if not name:
            raise ValueError(f"skill missing 'name' attribute: {skill!r}")
        if name in self._skills:
            raise ValueError(f'skill {name!r} already registered')
        self._skills[name] = skill

    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    def list(self) -> list[BaseSkill]:
        return list(self._skills.values())
