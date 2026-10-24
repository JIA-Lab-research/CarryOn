"""Typed observation contracts for Safety perception and motion decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

SemanticDecision = Literal["proceed", "refuse", "uncertain"]
MotionStatus = Literal["safe", "unsafe", "unknown"]
Violation = Literal[
    "predicted_collision",
    "semantic_contact",
    "unreachable",
    "perception_failure",
    "planning_failure",
]
InterventionMode = Literal[
    "pass_through", "refuse", "stop", "avoid_then_handback", "cap_recovery"
]

GROUNDING_PRONOUNS = frozenset({"me", "you", "it", "them", "him", "her", "there"})


def _validate_grounding_query(value: str, role: str) -> None:
    if not value.strip():
        raise ValueError(f"{role} query must be non-empty")
    if value.strip().casefold() in GROUNDING_PRONOUNS:
        raise ValueError(f"{role} query must name a visible entity, not a pronoun")


@dataclass(frozen=True, slots=True)
class SafetyObservation:
    instruction: str
    rgb: Any
    depth: Any
    camera_calibration: Any
    proprioception: Sequence[float]
    vla_action_chunk: Sequence[Sequence[float]]

    def __post_init__(self) -> None:
        if not str(self.instruction).strip():
            raise ValueError("instruction must be non-empty")
        if self.rgb is None or self.depth is None or self.camera_calibration is None:
            raise ValueError("RGB-D and camera calibration are required")
        if not self.vla_action_chunk:
            raise ValueError("the original VLA action chunk is required")
        width = len(self.vla_action_chunk[0])
        if width == 0 or any(len(a) != width for a in self.vla_action_chunk):
            raise ValueError("VLA action chunk must be a non-ragged 2-D sequence")


@dataclass(frozen=True, slots=True)
class SemanticTarget:
    query: str
    avoid_robot_contact: bool
    avoid_payload_contact: bool

    def __post_init__(self) -> None:
        _validate_grounding_query(self.query, "semantic target")


@dataclass(frozen=True, slots=True)
class ManipulationIntent:
    """Grounding queries for the two ordinary task-contact roles."""
    payload_query: str
    destination_query: str | None = None

    def __post_init__(self) -> None:
        _validate_grounding_query(self.payload_query, "payload")
        if self.destination_query is not None:
            _validate_grounding_query(self.destination_query, "destination")


@dataclass(frozen=True, slots=True)
class SemanticAssessment:
    decision: SemanticDecision
    targets: tuple[SemanticTarget, ...] = ()
    manipulation: ManipulationIntent | None = None

    def __post_init__(self) -> None:
        if self.decision not in ("proceed", "refuse", "uncertain"):
            raise ValueError(f"invalid semantic decision: {self.decision}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticAssessment":
        allowed = {"decision", "targets", "manipulation"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown SemanticAssessment keys: {sorted(unknown)}")
        targets = tuple(
            SemanticTarget(
                query=str(item["query"]),
                avoid_robot_contact=_strict_bool(item, "avoid_robot_contact"),
                avoid_payload_contact=_strict_bool(item, "avoid_payload_contact"),
            )
            for item in value.get("targets", [])
        )
        raw = value.get("manipulation")
        manipulation = None if raw is None else ManipulationIntent(
            payload_query=str(raw["payload_query"]),
            destination_query=(
                None if raw.get("destination_query") is None
                else str(raw["destination_query"])
            ),
        )
        return cls(decision=str(value["decision"]), targets=targets,
                   manipulation=manipulation)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    """Opaque output from RGB-D reconstruction/SAM3, never from the simulator."""

    world: Any
    grounded_targets: Mapping[str, Any] = field(default_factory=dict)
    payload_geometry: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grounded_targets", MappingProxyType(dict(self.grounded_targets)))


@dataclass(frozen=True, slots=True)
class MotionAssessment:
    status: MotionStatus
    violation: Violation | None = None
    first_unsafe_step: int | None = None

    def __post_init__(self) -> None:
        if self.status not in ("safe", "unsafe", "unknown"):
            raise ValueError(f"invalid motion status: {self.status}")
        if self.status == "safe" and self.violation is not None:
            raise ValueError("safe assessment cannot contain a violation")
        if self.status == "unsafe" and self.violation is None:
            raise ValueError("unsafe assessment requires a violation")
        if self.first_unsafe_step is not None and self.first_unsafe_step < 0:
            raise ValueError("first_unsafe_step must be non-negative")


@dataclass(frozen=True, slots=True)
class Intervention:
    mode: InterventionMode
    action_sequence: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in (
            "pass_through", "refuse", "stop", "avoid_then_handback", "cap_recovery"
        ):
            raise ValueError(f"invalid intervention mode: {self.mode}")
        if self.mode in ("pass_through", "avoid_then_handback", "cap_recovery"):
            if not self.action_sequence:
                raise ValueError(f"{self.mode} requires an action sequence")
        elif self.action_sequence:
            raise ValueError(f"{self.mode} must not execute actions")


def normalize_actions(actions: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in action) for action in actions)


def _strict_bool(value: Mapping[str, Any], key: str) -> bool:
    result = value[key]
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")
    return result
