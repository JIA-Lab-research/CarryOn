from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RobotEndpoint:
    name: str
    transport: str  # "http" | "ros" | "mock"
    address: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    skill_library: list[str] = field(default_factory=list)
    cancel_path: str = "/cancel"
    # When this endpoint is cancelled (whether user-driven, deadline, or
    # because the VLM Judge declared the subtask done), should the executor
    # be asked to drive the arm back to its home/init pose before reporting
    # the cancellation? CarryOn forwards this flag in the cancel request
    # body as ``{"return_home": <bool>}``; honoring it is the executor's
    # responsibility (e.g. gello's r3_bridge / continuous-inference loop
    # treats it as "hard reset to init" after braking).
    return_home_on_cancel: bool = True
    extra: dict = field(default_factory=dict)

    def for_planner_vlm(self) -> dict:
        """The subset the planner VLM is allowed to see (no auth/url details)."""
        return {
            "name": self.name,
            "transport": self.transport,
            "description": self.description,
            "tags": list(self.tags),
            "skill_library": list(self.skill_library),
        }
