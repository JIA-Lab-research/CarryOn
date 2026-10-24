"""Data structures for episode recording and recovery reflection.

Records reference media by path and include camera calibration, state samples,
judge events and recovery traces. Rewind targets come from previously visited
joint states in the trajectory buffer.

Positions are metres and joint angles are radians. Quaternions use wxyz order
and unit norm. Poses are row-major 4x4 world_from_X transforms; extrinsics are
world_from_camera. Intrinsics are row-major 3x3 pinhole matrices. Timestamps are
float seconds on the shared monotonic recorder clock."""
from __future__ import annotations
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional
SCHEMA_VERSION = 'recovery-episode/v0'
CameraRole = Literal['eye_in_hand', 'eye_to_hand', 'external_fixed']
JudgeVerdict = Literal['ok', 'fail', 'unsure']
RecoveryOutcome = Literal['success', 'failed', 'aborted', 'not_triggered']
HandbackMode = Literal['full_hold', 'pre_grasp', 'pre_place', 'complete']

@dataclass
class CameraSpec:
    """Camera calibration recorded with an episode.

    intrinsics is a row-major 3x3 matrix; extrinsics is a row-major 4x4
    world_from_camera transform. depth_scale converts stored depth to metres.
    role distinguishes a fixed camera from an arm-mounted camera."""
    name: str
    role: CameraRole
    width: int
    height: int
    intrinsics: list[list[float]]
    extrinsics: list[list[float]]
    depth_scale: float = 0.001

@dataclass
class FrameRef:
    """A single RGB capture for one camera at one timestamp, optionally with depth.

    Media lives on disk; we only carry paths + the timestamp so the record is
    cheap to load, diff, and validate. ``depth_path`` may be ``None`` for cameras
    that did not stream depth at that instant.
    """
    ts: float
    camera: str
    rgb_path: str
    depth_path: Optional[str] = None

@dataclass
class TrajSample:
    """One robot state sample (subscribed from gello's ZMQ PUB stream).

    The rewind target is chosen from this stream, so these samples define the set
    of *physically-visited* configurations the recovery is allowed to return to.
    """
    ts: float
    q: list[float]
    ee_pos: list[float]
    ee_quat: list[float]
    gripper_width: float
    w_cmd: Optional[float] = None
    w_meas: Optional[float] = None
    grasp_contact: Optional[bool] = None
    release_commanded: Optional[bool] = None

@dataclass
class JudgeTick:
    """One vlm_judge observation at a timestamp."""
    ts: float
    verdict: JudgeVerdict
    score: Optional[float] = None
    note: Optional[str] = None

@dataclass
class GraspProposal:
    """A grasp pose the recovery perception layer produced, in world frame."""
    position: list[float]
    quaternion: list[float]
    score: Optional[float] = None
    source: Optional[str] = None
    predicted_width_m: Optional[float] = None

@dataclass
class GroundingResult:
    """Typed target-grounding evidence consumed by recovery perception.

    Grounding answers "which target?" separately from grasp estimation. It can
    carry a point, region, mask or 3D prior, plus provenance for replay/reflection.
    """
    mode: str = 'generic'
    provenance: str = ''
    confidence: Optional[float] = None
    normalized_point: Optional[list[float]] = None
    region: Optional[list[float]] = None
    world_point: Optional[list[float]] = None
    mask_path: Optional[str] = None
    object_hint: Optional[str] = None
    plan: Optional[list[str]] = None
    attempts: list[dict] = field(default_factory=list)

@dataclass
class RecoveryInvocation:
    """Record the recovery trace, including partial or aborted invocations."""
    triggered_ts: float
    failure_type: str
    policy_name: str
    policy_version: str
    request_id: Optional[str] = None
    handback_mode: HandbackMode = 'full_hold'
    fallback_used: Optional[bool] = None
    effective_op_sequence: Optional[list[str]] = None
    rewind_target_ts: Optional[float] = None
    rewind_target_q: Optional[list[float]] = None
    planned_traj_q: Optional[list[list[float]]] = None
    perceived_grasp: Optional[GraspProposal] = None
    outcome: RecoveryOutcome = 'not_triggered'
    notes: Optional[str] = None
    steps: Optional[list[dict]] = None
    grounding: Optional[GroundingResult] = None

@dataclass
class EpisodeRecord:
    """Episode record consumed by recovery reflection and analysis."""
    episode_id: str
    goal: str
    ts_start: float
    ts_end: float
    cameras: list[CameraSpec] = field(default_factory=list)
    frames: list[FrameRef] = field(default_factory=list)
    trajectory: list[TrajSample] = field(default_factory=list)
    judge_timeline: list[JudgeTick] = field(default_factory=list)
    first_fail_ts: Optional[float] = None
    episode_success: Optional[bool] = None
    recovery: Optional[RecoveryInvocation] = None
    schema_version: str = SCHEMA_VERSION
    robot: str = 'fr3'
    gello_version: Optional[str] = None
    recovery_lib_version: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    recording_quality: Optional[dict] = None

    @staticmethod
    def new_id() -> str:
        return f'ep-{uuid.uuid4().hex[:10]}'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> 'EpisodeRecord':
        cams = [CameraSpec(**c) for c in obj.get('cameras', []) or []]
        frames = [FrameRef(**f) for f in obj.get('frames', []) or []]
        traj = [TrajSample(**t) for t in obj.get('trajectory', []) or []]
        judge = [JudgeTick(**j) for j in obj.get('judge_timeline', []) or []]
        rec_obj = obj.get('recovery')
        recovery: Optional[RecoveryInvocation] = None
        if rec_obj:
            grasp_obj = rec_obj.get('perceived_grasp')
            grasp = GraspProposal(**grasp_obj) if grasp_obj else None
            grounding_obj = rec_obj.get('grounding')
            grounding = GroundingResult(**grounding_obj) if grounding_obj else None
            recovery = RecoveryInvocation(**{k: v for k, v in rec_obj.items() if k not in ('perceived_grasp', 'grounding')}, perceived_grasp=grasp, grounding=grounding)
        known = {'cameras', 'frames', 'trajectory', 'judge_timeline', 'recovery'}
        scalars = {k: v for k, v in obj.items() if k not in known}
        return cls(cameras=cams, frames=frames, trajectory=traj, judge_timeline=judge, recovery=recovery, **scalars)

def _matrix(rows: int, cols: int) -> dict[str, Any]:
    return {'type': 'array', 'minItems': rows, 'maxItems': rows, 'items': {'type': 'array', 'minItems': cols, 'maxItems': cols, 'items': {'type': 'number'}}}

def _vec(n: int) -> dict[str, Any]:
    return {'type': 'array', 'minItems': n, 'maxItems': n, 'items': {'type': 'number'}}
EPISODE_SCHEMA: dict[str, Any] = {'type': 'object', 'required': ['episode_id', 'goal', 'ts_start', 'ts_end'], 'properties': {'schema_version': {'type': 'string'}, 'episode_id': {'type': 'string', 'minLength': 1}, 'goal': {'type': 'string'}, 'ts_start': {'type': 'number'}, 'ts_end': {'type': 'number'}, 'robot': {'type': 'string'}, 'first_fail_ts': {'type': ['number', 'null']}, 'episode_success': {'type': ['boolean', 'null']}, 'recording_quality': {'type': ['object', 'null']}, 'cameras': {'type': 'array', 'items': {'type': 'object', 'required': ['name', 'role', 'width', 'height', 'intrinsics', 'extrinsics'], 'properties': {'name': {'type': 'string', 'minLength': 1}, 'role': {'enum': ['eye_in_hand', 'eye_to_hand', 'external_fixed']}, 'width': {'type': 'integer', 'minimum': 1}, 'height': {'type': 'integer', 'minimum': 1}, 'intrinsics': _matrix(3, 3), 'extrinsics': _matrix(4, 4), 'depth_scale': {'type': 'number', 'exclusiveMinimum': 0}}}}, 'frames': {'type': 'array', 'items': {'type': 'object', 'required': ['ts', 'camera', 'rgb_path'], 'properties': {'ts': {'type': 'number'}, 'camera': {'type': 'string', 'minLength': 1}, 'rgb_path': {'type': 'string', 'minLength': 1}, 'depth_path': {'type': ['string', 'null']}}}}, 'trajectory': {'type': 'array', 'items': {'type': 'object', 'required': ['ts', 'q', 'ee_pos', 'ee_quat', 'gripper_width'], 'properties': {'ts': {'type': 'number'}, 'q': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 1}, 'ee_pos': _vec(3), 'ee_quat': _vec(4), 'gripper_width': {'type': 'number'}, 'w_cmd': {'type': ['number', 'null']}, 'w_meas': {'type': ['number', 'null']}, 'grasp_contact': {'type': ['boolean', 'null']}, 'release_commanded': {'type': ['boolean', 'null']}}}}, 'judge_timeline': {'type': 'array', 'items': {'type': 'object', 'required': ['ts', 'verdict'], 'properties': {'ts': {'type': 'number'}, 'verdict': {'enum': ['ok', 'fail', 'unsure']}, 'score': {'type': ['number', 'null']}, 'note': {'type': ['string', 'null']}}}}, 'recovery': {'type': ['object', 'null'], 'required': ['triggered_ts', 'failure_type', 'policy_name', 'policy_version'], 'properties': {'triggered_ts': {'type': 'number'}, 'failure_type': {'type': 'string', 'minLength': 1}, 'policy_name': {'type': 'string', 'minLength': 1}, 'policy_version': {'type': 'string', 'minLength': 1}, 'request_id': {'type': ['string', 'null']}, 'handback_mode': {'enum': ['full_hold', 'pre_grasp', 'pre_place', 'complete']}, 'fallback_used': {'type': ['boolean', 'null']}, 'effective_op_sequence': {'type': ['array', 'null'], 'items': {'type': 'string', 'minLength': 1}}, 'rewind_target_ts': {'type': ['number', 'null']}, 'rewind_target_q': {'type': ['array', 'null'], 'items': {'type': 'number'}}, 'planned_traj_q': {'type': ['array', 'null']}, 'perceived_grasp': {'type': ['object', 'null'], 'required': ['position', 'quaternion'], 'properties': {'position': _vec(3), 'quaternion': _vec(4), 'score': {'type': ['number', 'null']}, 'source': {'type': ['string', 'null']}}}, 'grounding': {'type': ['object', 'null'], 'properties': {'mode': {'type': 'string'}, 'provenance': {'type': 'string'}, 'confidence': {'type': ['number', 'null']}, 'normalized_point': {'type': ['array', 'null'], 'minItems': 2, 'maxItems': 2, 'items': {'type': 'number'}}, 'region': {'type': ['array', 'null'], 'minItems': 4, 'maxItems': 4, 'items': {'type': 'number'}}, 'world_point': {'type': ['array', 'null'], 'minItems': 2, 'maxItems': 3, 'items': {'type': 'number'}}, 'mask_path': {'type': ['string', 'null']}, 'object_hint': {'type': ['string', 'null']}, 'plan': {'type': ['array', 'null'], 'items': {'type': 'string'}}, 'attempts': {'type': 'array', 'items': {'type': 'object'}}}}, 'outcome': {'enum': ['success', 'failed', 'aborted', 'not_triggered']}, 'notes': {'type': ['string', 'null']}, 'steps': {'type': ['array', 'null']}}}}}

class EpisodeSchemaError(ValueError):
    """Raised when an episode record violates :data:`EPISODE_SCHEMA`."""

def validate_episode_dict(obj: dict[str, Any]) -> None:
    """Validate a raw dict against :data:`EPISODE_SCHEMA`.

    Raises :class:`EpisodeSchemaError` on the first violation. Kept separate from
    the higher-level *semantic* checks in :mod:`carryon.recovery.assertions`
    (which assume the record is already schema-valid).
    """
    try:
        import jsonschema
    except ImportError as e:
        raise EpisodeSchemaError('jsonschema package is required') from e
    try:
        jsonschema.validate(obj, EPISODE_SCHEMA)
    except jsonschema.ValidationError as e:
        raise EpisodeSchemaError(f'schema: {e.message} at {list(e.path)}') from e

def new_episode_record(goal: str, *, ts_start: Optional[float]=None) -> EpisodeRecord:
    """Construct an episode record from recorder data."""
    t0 = ts_start if ts_start is not None else time.time()
    return EpisodeRecord(episode_id=EpisodeRecord.new_id(), goal=goal, ts_start=t0, ts_end=t0)
