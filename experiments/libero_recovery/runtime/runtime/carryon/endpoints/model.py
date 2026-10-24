from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RobotEndpoint:
    name: str
    transport: str
    address: str
    description: str = ''
    tags: list[str] = field(default_factory=list)
    skill_library: list[str] = field(default_factory=list)
    cancel_path: str = '/cancel'
    return_home_on_cancel: bool = True
    extra: dict = field(default_factory=dict)

    def for_planner_vlm(self) -> dict:
        """The subset the planner VLM is allowed to see (no auth/url details)."""
        return {'name': self.name, 'transport': self.transport, 'description': self.description, 'tags': list(self.tags), 'skill_library': list(self.skill_library)}
