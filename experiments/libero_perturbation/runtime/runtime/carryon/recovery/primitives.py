"""Recovery backend interface and dependency-free primitive helpers.

Policies compose deterministic primitives through RecoveryBackend. Helpers
select rewind targets, transform grasp frames, and check holding, displacement,
pose and joint convergence. Positions use metres, joints use radians and
quaternions use wxyz order; the arm joint vector has seven elements."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from .schema import EpisodeRecord, GraspProposal, GroundingResult, TrajSample
DEFAULT_REWIND_LEAD_S: float = 1.0
DEFAULT_GRIPPER_EMPTY_WIDTH: float = 0.005
DEFAULT_GRIPPER_CLOSED_WIDTH: float = 0.075
DEFAULT_POS_TOL_M: float = 0.02
DEFAULT_ANG_TOL_RAD: float = math.radians(10)
DEFAULT_JOINT_TOL_RAD: float = math.radians(3)
DEFAULT_OBJECT_MOVED_TOL_M: float = 0.04

@dataclass
class MotionResult:
    """Outcome of a motion primitive (rewind / goto / regrasp)."""
    ok: bool
    reason: str = ''
    final_q: Optional[list[float]] = None

class RecoveryBackend(Protocol):
    """Hardware/perception contract the recovery primitives are built on.

    A backend bridges to gello (motion, gripper, joint state over the shared ZMQ
    bridge) and to the perception stack (external fixed camera + depth -> grasp).
    Policies depend only on this Protocol, never on a concrete robot module.
    """

    def ground_target(self, object_hint: str, *, plan: list[str], target_xy: Optional[tuple]=None, target_px: Optional[tuple]=None) -> Optional[GroundingResult]:
        """Produce explicit target-grounding evidence before grasp perception."""
        ...

    def perceive_grasp(self, object_hint: str, source: str='topdown', *, target_xy: Optional[tuple]=None, target_px: Optional[tuple]=None, grounding: Optional[GroundingResult]=None) -> Optional[GraspProposal]:
        """Detect ``object_hint`` from the (eye-to-hand) camera + depth and return
        a world-frame grasp pose, or ``None`` if it cannot be grounded.

        ``source`` selects the grasp source. ``"topdown"`` (default) is the
        deterministic mask/depth top-down PCA grasp -- the reliable path for
        upright tabletop objects (cup, bottle, carrot, fruit, blocks) on our
        rig. ``"cgn"`` forces Contact-GraspNet's learned 6-DoF grasp (use only
        when an LLM explicitly needs a non-top-down approach, e.g. a handle).
        ``"auto"`` runs the hybrid shell: CGN first (subject to a geometric
        table gate that distrusts sub-table grasps), mask top-down on decline.
        The returned :class:`GraspProposal`'s ``source`` records which path
        actually fired."""
        ...

    def perceive_grasps(self, object_hint: str, *, top_k: int=15, source: str='topdown', target_xy: Optional[tuple]=None, target_px: Optional[tuple]=None, grounding: Optional[GroundingResult]=None) -> list[GraspProposal]:
        """Top-K candidates for cap-x-style cuRobo goalset planning.

        Returns up to ``top_k`` grasps sorted by score (descending). The motion
        layer iterates IK + collision-free planning over the list and picks the
        first reachable candidate (cap-x's ``parse_grasp_poses_for_curobo`` +
        ``plan_to_grasp_poses``). For deterministic mask top-down (which has no
        natural top-K surface), a length-1 list is returned. Returns ``[]`` on
        perception failure."""
        ...

    def rewind_to_pose(self, q_target: Sequence[float]) -> MotionResult:
        """Plan a collision-free path (curobo) to a *previously-visited* joint
        state and execute it. ``q_target`` comes from the trajectory ring buffer,
        so it is reachable by construction; only the path needs validation."""
        ...

    def goto_pose(self, position: Sequence[float], quaternion_wxyz: Sequence[float], z_approach: float=0.0, verify: bool=True) -> MotionResult:
        """Move the end-effector to a Cartesian pose. ``z_approach`` > 0 first
        moves to a standoff that far back along the **grasp's approach axis**
        (gripper local -z, not world -z). For pure top-down grasps this is the
        familiar "lift up" standoff; for tilted 6-DoF grasps it shifts in xy too,
        so the final descent traces the gripper's approach axis. See
        :func:`grasp_local_standoff`."""
        ...

    def goto_grasp_goalset(self, grasps: Sequence[GraspProposal], *, z_approach: float=0.1, verify: bool=True, plan_grasp_to_lift: bool=False) -> MotionResult:
        """Move to a reachable grasp selected from a top-K goalset."""
        ...

    def regrasp(self, object_hint: str, pre_grasp_dis: float=0.1, source: str='auto') -> MotionResult:
        """Composite convenience primitive: perceive -> open -> approach (standoff
        ``pre_grasp_dis``) -> descend -> close. Returns failure if perception or
        any motion step fails. Policies may call this or compose the steps
        themselves for finer control."""
        ...

    def lift(self, dz: float=0.05) -> MotionResult:
        """Lift the hand straight up (world +Z) by ``dz`` metres, holding the
        current gripper width. Used to **confirm a grasp under gravity**: a
        too-high/loose grasp drops the object during the lift, so a following
        ``assert_holding`` sees the jaws collapse to the close command and the miss
        is caught -- unlike the static post-close width check, which cannot tell a
        liftable grasp from one that merely stalls the jaws on the object."""
        ...

    def open_gripper(self) -> MotionResult:
        ...

    def close_gripper(self) -> MotionResult:
        ...

    def current_q(self) -> list[float]:
        """Current joint positions (len 7 for FR3)."""
        ...

    def gripper_width(self) -> float:
        """Current gripper opening width in meters."""
        ...

    def geometric_check(self, predicate: 'Predicate') -> bool:
        """Evaluate a deterministic geometric predicate against the *current*
        observation/state. The backend supplies the live measurements; the
        predicate (see helpers below) is pure logic over those measurements."""
        ...

class Predicate(Protocol):
    name: str

    def __call__(self, ctx: 'PredicateContext') -> bool:
        ...

@dataclass
class PredicateContext:
    """Live measurements the backend passes to a predicate at check time.

    Kept as a flat, optional bag so predicates only read what they need and the
    same context works for in-distribution checks and AgentChord-style monitors.
    """
    q: Optional[list[float]] = None
    ee_pos: Optional[list[float]] = None
    ee_quat: Optional[list[float]] = None
    gripper_width: Optional[float] = None
    object_pos: Optional[list[float]] = None
    object_pos_ref: Optional[list[float]] = None
REWIND_HOLD_MIN_M = 0.008
REWIND_HOLD_MAX_M = 0.075

def select_rewind_target(rec: EpisodeRecord, *, lead_s: float=DEFAULT_REWIND_LEAD_S, n_steps: Optional[int]=None, dt: Optional[float]=None, hold_min_m: float=REWIND_HOLD_MIN_M, hold_max_m: float=REWIND_HOLD_MAX_M) -> Optional[TrajSample]:
    """Select a previously visited joint state before the first sustained hold ends.

    Use first_fail_ts if no sustained hold is recorded. The lead is lead_s, or
    n_steps * dt when n_steps is supplied. Select the latest sample at or before
    that target time. If the failure timestamp is missing or the lead exceeds the
    available history, use the earliest recorded sample. An empty trajectory
    returns None."""
    if not rec.trajectory:
        return None
    if n_steps is not None and dt is not None:
        lead_s = n_steps * dt
    if rec.first_fail_ts is None:
        print('[rewind] fail-safe: first_fail_ts missing at recovery build; using earliest recorded pose as rewind target', flush=True)
        return min(rec.trajectory, key=lambda s: s.ts)
    segs = []
    cur = None
    for smp in sorted(rec.trajectory, key=lambda x: x.ts):
        if smp.ts > rec.first_fail_ts:
            break
        try:
            w = float(smp.gripper_width)
        except (TypeError, ValueError):
            continue
        if hold_min_m <= w <= hold_max_m:
            if cur is None:
                cur = [smp.ts, smp.ts]
            else:
                cur[1] = smp.ts
        elif cur is not None:
            segs.append(tuple(cur))
            cur = None
    if cur is not None:
        segs.append(tuple(cur))
    MIN_SUSTAIN_S = 0.5
    anchor = rec.first_fail_ts
    for a, b in segs:
        if b - a >= MIN_SUSTAIN_S:
            anchor = b
            break
    target_ts = anchor - lead_s
    eligible = [s for s in rec.trajectory if s.ts <= target_ts]
    if eligible:
        return max(eligible, key=lambda s: s.ts)
    print('[rewind] fail-safe: no trajectory sample older than target_ts (early drop / short pre-fail window); using earliest recorded pose', flush=True)
    return min(rec.trajectory, key=lambda s: s.ts)

def select_rewind_targets(rec: EpisodeRecord, *, lead_s: float=DEFAULT_REWIND_LEAD_S, n: int=4, spacing_s: float=1.0) -> list[TrajSample]:
    """Return the primary rewind target and progressively earlier fallbacks.

    select_rewind_target supplies the first candidate. Subsequent candidates are
    spacing_s earlier in the recorded trajectory. Deduplicate consecutive equal samples."""
    out: list[TrajSample] = []
    for i in range(max(1, int(n))):
        t = select_rewind_target(rec, lead_s=lead_s + i * max(1e-06, spacing_s))
        if t is None:
            break
        if out and out[-1].ts == t.ts:
            continue
        out.append(t)
    return out

def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum(((x - y) ** 2 for x, y in zip(a, b))))

def quat_angle_rad(q1: Sequence[float], q2: Sequence[float]) -> float:
    """Geodesic angle between two unit quaternions (wxyz), in radians."""
    dot = abs(sum((a * b for a, b in zip(q1, q2))))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)

def grasp_local_standoff(position: Sequence[float], quaternion_wxyz: Sequence[float], distance_m: float) -> list[float]:
    """Return a standoff opposite the grasp approach direction.

    For a world-frame grasp with wxyz quaternion, the approach is R(quat)[:, 2].
    The standoff is position - distance_m * R(quat)[:, 2]. For a top-down grasp
    this moves upward in world z; a tilted grasp also changes world xy."""
    if distance_m == 0.0:
        return [float(position[0]), float(position[1]), float(position[2])]
    ax, ay, az = grasp_approach_axis(quaternion_wxyz)
    return [float(position[0]) - distance_m * ax, float(position[1]) - distance_m * ay, float(position[2]) - distance_m * az]

def grasp_approach_axis(quaternion_wxyz: Sequence[float]) -> list[float]:
    """Return gripper local +z / approach axis in world coordinates.

    This is the third rotation-matrix column for a unit ``wxyz`` quaternion. It is
    the single frame convention used by all CarryOn grasp motion primitives:
    detector outputs and ``GraspProposal.position`` are fingertip/contact-center
    poses, while motion planners drive the ``panda_hand`` tool frame along this
    approach axis. Keeping this helper central prevents ``goto_grasp`` and
    ``goto_grasp_goalset`` from silently diverging.
    """
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = (w / n, x / n, y / n, z / n)
    return [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)]

def fingertip_to_hand_pose(position: Sequence[float], quaternion_wxyz: Sequence[float], tcp_offset_m: float) -> tuple[list[float], list[float]]:
    """Convert a fingertip/contact grasp pose into a ``panda_hand`` tool pose.

    Contract, matching cap-x's cuRobo helpers:
    ``GraspProposal.position`` is the desired fingertip/contact-center point in
    world coordinates; cuRobo plans for the hand/tool frame. Therefore the hand
    target is ``fingertip - tcp_offset_m * approach_axis``. Quaternions are
    normalized ``wxyz`` and remain in the world frame.
    """
    q = [float(v) for v in quaternion_wxyz]
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    qn = [w / n, x / n, y / n, z / n]
    approach = grasp_approach_axis(qn)
    p = [float(v) for v in position]
    hand = [p[i] - float(tcp_offset_m) * approach[i] for i in range(3)]
    return (hand, qn)

def grasp_proposal_hand_pose(grasp: GraspProposal, tcp_offset_m: float) -> tuple[list[float], list[float]]:
    """``GraspProposal`` wrapper for :func:`fingertip_to_hand_pose`."""
    return fingertip_to_hand_pose(grasp.position, grasp.quaternion, tcp_offset_m)

def gripper_holding(width: float, *, empty_width: float=DEFAULT_GRIPPER_EMPTY_WIDTH, open_width: float=DEFAULT_GRIPPER_CLOSED_WIDTH, closed_cmd_width: Optional[float]=None, hold_margin: float=0.0) -> bool:
    """True iff the gripper is closed on *something*.

    Two ways the jaws can be "not closed on air":

    - the fixed floor ``empty_width`` (legacy): ``width > empty_width``; and
    - **command-relative** (robust): when the close *command* would shut the jaws
      to ``closed_cmd_width``, a real object blocks them OPEN of that target, so
      ``width >= closed_cmd_width + hold_margin``. This is what distinguishes a
      grasp from a miss when the close target is non-zero (e.g. 12 mm): a miss
      reaches ~12 mm and a fixed 4 mm floor would *false-positive*; requiring the
      jaws to stall wider than the commanded close catches it.

    The effective lower bound is the stricter (larger) of the two; the upper bound
    ``open_width`` rejects jaws that never closed from open."""
    lower = empty_width
    if closed_cmd_width is not None:
        lower = max(lower, float(closed_cmd_width) + float(hold_margin))
    return lower < width < open_width

def object_displacement(pos: Sequence[float], pos_ref: Sequence[float]) -> float:
    return _l2(pos, pos_ref)

def object_moved(pos: Sequence[float], pos_ref: Sequence[float], *, threshold: float=DEFAULT_OBJECT_MOVED_TOL_M) -> bool:
    """AgentChord-style ``object_moved`` monitor: True == the object drifted more
    than ``threshold`` from its reference position (failure signal during grasp)."""
    return object_displacement(pos, pos_ref) > threshold

def pose_reached(cur_pos: Sequence[float], cur_quat: Sequence[float], tgt_pos: Sequence[float], tgt_quat: Sequence[float], *, pos_tol: float=DEFAULT_POS_TOL_M, ang_tol: float=DEFAULT_ANG_TOL_RAD) -> bool:
    """True iff the end-effector is within ``pos_tol`` (m) and ``ang_tol`` (rad) of
    a target pose."""
    return _l2(cur_pos, tgt_pos) <= pos_tol and quat_angle_rad(cur_quat, tgt_quat) <= ang_tol

def joints_close(q_cur: Sequence[float], q_tgt: Sequence[float], *, tol: float=DEFAULT_JOINT_TOL_RAD) -> bool:
    """True iff every joint is within ``tol`` (rad) of the target -- used to confirm
    a rewind actually returned to the intended visited state."""
    if len(q_cur) != len(q_tgt):
        return False
    return all((abs(a - b) <= tol for a, b in zip(q_cur, q_tgt)))

def densify_joint_path(waypoints: Sequence[Sequence[float]], *, max_step_rad: float) -> list[list[float]]:
    """Linearly resample a joint path to bound each per-joint step by max_step_rad.

    This limits waypoint jumps in the streamed trajectory. A non-positive
    max_step_rad returns the path unchanged."""
    pts = [list(map(float, wp)) for wp in waypoints]
    if max_step_rad <= 0.0 or len(pts) < 2:
        return pts
    dense: list[list[float]] = [pts[0]]
    for q in pts[1:]:
        prev = dense[-1]
        span = max((abs(a - b) for a, b in zip(prev, q)), default=0.0)
        n = max(1, int(math.ceil(span / max_step_rad)))
        for i in range(1, n + 1):
            t = i / n
            dense.append([a + (b - a) * t for a, b in zip(prev, q)])
    return dense
