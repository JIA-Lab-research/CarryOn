"""Recovery backend combining joint control, camera observations and perception.

ZmqBridgeClient supplies state and joint/gripper commands; GelloControlClient
supplies images and control handoff. Motion planning and grasp detection are
injected so this module does not import their dependencies."""
from __future__ import annotations
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from .bridge import TrajectoryRingBuffer, ZmqBridgeClient
from .gello_client import GelloControlClient
from .jam_z_reflection import GraspNoHoldSignal, JamResidualSignal, JamZReflector
from .primitives import DEFAULT_JOINT_TOL_RAD, MotionResult, Predicate, PredicateContext, densify_joint_path, fingertip_to_hand_pose, grasp_approach_axis, grasp_local_standoff, joints_close
from .schema import GraspProposal, GroundingResult
from .semantic_memory import scene_bundle_identity

class MotionPlanner(Protocol):
    """Collision-free planner / IK. A real impl wraps curobo (+ pyroki for IK).

    All methods return a list of joint waypoints (each len 7) to stream to the
    bridge, or ``None`` if planning failed (unreachable / in collision)."""

    def plan_to_q(self, q_start: Sequence[float], q_goal: Sequence[float]) -> Optional[list[list[float]]]:
        ...

    def plan_to_pose(self, q_start: Sequence[float], position: Sequence[float], quaternion_wxyz: Sequence[float]) -> Optional[list[list[float]]]:
        ...

class GraspDetector(Protocol):
    """Perception backend: RGB-D + calibration -> world-frame grasp pose."""

    def detect(self, *, rgb: object, depth_m: object, intrinsics: dict, object_hint: str) -> Optional[GraspProposal]:
        ...
EXPLICIT_TOPDOWN_SOURCE_ALIASES = frozenset({'topdown', 'fallback', 'mask', 'mask_depth'})
LIVE_REGISTERED_GRASP_SOURCES = frozenset({'auto', *EXPLICIT_TOPDOWN_SOURCE_ALIASES})
_FRANKA_JOINT7_MIN_RAD = -2.8973
_FRANKA_JOINT7_MAX_RAD = 2.8973

def _rewind_symmetric_joint7_target(q_start: Sequence[float], q_target: Sequence[float]) -> list[float]:
    """Use a nearer 180-degree-equivalent wrist roll for rewind only.

    The parallel gripper is symmetric under a pi rotation about joint 7. Keep
    joints 1-6 byte-for-byte unchanged, and replace only joint 7 when a +/-pi
    equivalent is inside the Franka limit and is strictly nearer to the
    current joint-7 position. Otherwise preserve the recorded target.
    """
    target = list(q_target)
    if len(q_start) <= 6 or len(target) <= 6:
        return target
    current = float(q_start[6])
    recorded = float(target[6])
    best = recorded
    best_travel = abs(recorded - current)
    for candidate in (recorded - math.pi, recorded + math.pi):
        if not _FRANKA_JOINT7_MIN_RAD <= candidate <= _FRANKA_JOINT7_MAX_RAD:
            continue
        travel = abs(candidate - current)
        if travel < best_travel:
            best = candidate
            best_travel = travel
    target[6] = best
    return target

@dataclass
class BackendConfig:
    gripper_open_width: float = 0.08
    gripper_close_width: float = 0.0
    stream_hz: float = 40.0
    max_exec_step_rad: float = 0.01
    settle_s: float = 0.4
    gripper_dwell_s: float = 0.8
    verify_joint_tol_rad: float = DEFAULT_JOINT_TOL_RAD
    rewind_joint_tol_rad: float = math.radians(12)
    rewind_wrist_roll_tol_rad: float = math.radians(45)
    rewind_max_exec_step_rad: float = 0.0
    rewind_escape_nudge_rad: float = 0.0
    lift_max_exec_step_rad: float = 0.0
    goalset_plan_lift: bool = True
    rewind_ee_accept_mm: float = 50.0
    state_timeout_s: float = 2.0
    settle_poll_s: float = 10.0
    settle_stable_reads: int = 3
    settle_eps_rad: float = 0.0015
    settle_no_verify_poll_s: float = 4.0
    allow_unplanned_rewind: bool = False
    unplanned_max_step_rad: float = 0.05
    grasp_camera: str = 'external'
    tcp_offset_m: float = 0.0
    goalset_workspace_reach_m: float = 1.1
    scene_exclude_radius_m: float = 0.06
    scene_voxel_pitch_m: float = 0.02
    goalset_scene_collision: bool = False
    goalset_max_attempts: int = 4
    goalset_prefer_low_tilt: bool = True
    goalset_max_tilt_deg: float = 40.0
    goalset_failopen_tilt_band_deg: float = 15.0
    goalset_settle_poll_s: Optional[float] = None
    goalset_settle_no_verify_poll_s: Optional[float] = None
    goalset_approach_verify: bool = False
    goalset_descent_verify: bool = False
    descent_progress_tol_mm: float = 30.0

class GelloRecoveryBackend:
    """Concrete :class:`carryon.recovery.primitives.RecoveryBackend`."""

    def __init__(self, bridge: ZmqBridgeClient, gello: GelloControlClient, *, planner: Optional[MotionPlanner]=None, detector: Optional[GraspDetector]=None, detector_sources: Optional[dict[str, GraspDetector]]=None, buffer: Optional[TrajectoryRingBuffer]=None, config: Optional[BackendConfig]=None, jam_z_reflector: Optional[JamZReflector]=None, semantic_reflector=None) -> None:
        self.bridge = bridge
        self.gello = gello
        self.planner = planner
        self.detector = detector
        self.detector_sources = {str(name).strip().lower(): source_detector for name, source_detector in (detector_sources or {}).items() if str(name).strip() and source_detector is not None}
        self.buffer = buffer
        self.cfg = config or BackendConfig()
        self.jam_z_reflector = jam_z_reflector
        self.semantic_reflector = semantic_reflector
        self._last_grasp_object_key: Optional[str] = None
        self._last_executed_grasp_xyz_m: Optional[list[float]] = None
        self._last_descent_jam_signal: Optional[JamResidualSignal] = None
        self.last_jam_z_reflection_debug: dict = {}
        self.last_semantic_reflection_debug: dict = {}
        self.last_perceive_reason: str = ''
        self.last_perceive_debug: dict = {}
        self.identity_rejected: list = []
        self._place_dest_hint: str = ''
        self.last_object_center_world: Optional[list[float]] = None
        self._last_gripper_cmd_width: Optional[float] = None
        self._last_scene_perception: Optional[tuple] = None
        self.semantic_scene_bundle_required: bool = False
        self._last_scene_bundle_identity: Optional[dict] = None
        self.last_goalset_execution_debug: dict = {}
        self._lift_width_trace_active = False
        self._lift_width_trace_samples: list[dict] = []
        self._last_lift_width_trace: dict = {}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw not in ('0', 'false', 'False', 'no', 'off')

    def _goalset_max_attempts(self) -> int:
        return max(1, self._env_int('CARRYON_GOALSET_MAX_ATTEMPTS', int(self.cfg.goalset_max_attempts)))

    def _goalset_prefer_low_tilt(self) -> bool:
        return self._env_flag('CARRYON_GOALSET_TILT_SORT', bool(self.cfg.goalset_prefer_low_tilt))

    def _goalset_max_tilt_deg(self) -> float:
        return self._env_float('CARRYON_GOALSET_MAX_TILT_DEG', float(self.cfg.goalset_max_tilt_deg))

    def _goalset_approach_verify(self, caller_verify: bool) -> bool:
        return caller_verify and self._env_flag('CARRYON_GOALSET_APPROACH_VERIFY', bool(self.cfg.goalset_approach_verify))

    def _goalset_descent_verify(self, caller_verify: bool) -> bool:
        return caller_verify and self._env_flag('CARRYON_GOALSET_DESCENT_VERIFY', bool(self.cfg.goalset_descent_verify))

    def _goalset_settle_poll_s(self) -> float:
        env = os.environ.get('CARRYON_GOALSET_SETTLE_POLL_S')
        if env is not None:
            try:
                return float(env)
            except (TypeError, ValueError):
                pass
        if self.cfg.goalset_settle_poll_s is not None:
            return float(self.cfg.goalset_settle_poll_s)
        return float(self.cfg.settle_poll_s)

    def _goalset_settle_no_verify_poll_s(self) -> float:
        env = os.environ.get('CARRYON_GOALSET_SETTLE_NO_VERIFY_POLL_S')
        if env is not None:
            try:
                return float(env)
            except (TypeError, ValueError):
                pass
        if self.cfg.goalset_settle_no_verify_poll_s is not None:
            return float(self.cfg.goalset_settle_no_verify_poll_s)
        return float(self.cfg.settle_no_verify_poll_s)

    @staticmethod
    def _sort_goalset_candidates(grasps: Sequence[GraspProposal]) -> list[GraspProposal]:
        """Order GraspGen candidates by descending score."""
        return sorted(list(grasps), key=lambda g: -float(getattr(g, 'score', 0.0) or 0.0))

    def _record_lift_width_sample(self, state: dict) -> None:
        if not self._lift_width_trace_active:
            return
        value = state.get('w_meas')
        if value is None:
            value = state.get('w_cmd', state.get('gripper'))
        if value is None:
            return
        try:
            width = float(value)
        except (TypeError, ValueError):
            return
        ee_pos = state.get('ee_pos')
        ee_z = None
        try:
            ee_z = float(ee_pos[2])
        except (IndexError, TypeError, ValueError):
            pass
        self._lift_width_trace_samples.append({'ts': time.time(), 'width_m': width, 'grasp_contact': state.get('grasp_contact'), 'ee_z_m': ee_z})

    def _begin_lift_width_trace(self) -> None:
        self._lift_width_trace_samples = []
        self._lift_width_trace_active = True
        self._read_state(timeout_s=min(0.25, float(self.cfg.state_timeout_s)))

    def _finish_lift_width_trace(self) -> None:
        try:
            self._read_state(timeout_s=min(0.25, float(self.cfg.state_timeout_s)))
        finally:
            self._lift_width_trace_active = False
        samples = [dict(sample) for sample in self._lift_width_trace_samples]
        widths = [float(sample['width_m']) for sample in samples]
        self._last_lift_width_trace = {'schema': 'carryon-lift-width-trace/v1', 'sample_count': len(samples), 'width_min_m': min(widths) if widths else None, 'width_max_m': max(widths) if widths else None, 'delta_width_m': max(widths) - min(widths) if widths else None, 'all_contact_true': bool(samples) and all((sample.get('grasp_contact') is True for sample in samples)), 'samples': samples}
        self._lift_width_trace_samples = []

    def last_lift_width_trace(self) -> dict:
        result = dict(self._last_lift_width_trace)
        result['samples'] = [dict(sample) for sample in self._last_lift_width_trace.get('samples', [])]
        return result

    def _read_state(self, *, timeout_s: Optional[float]=None) -> Optional[dict]:
        """Block (briefly) for the freshest bridge state. CONFLATE means each
        successful recv is the latest; we retry until one arrives or timeout."""
        deadline = time.time() + (timeout_s if timeout_s is not None else self.cfg.state_timeout_s)
        while time.time() < deadline:
            st = self.bridge.get_state()
            if st:
                self._record_lift_width_sample(st)
                return st
            time.sleep(0.01)
        return None

    @staticmethod
    def _joints_converged(q_now: Sequence[float], target: Sequence[float], *, tol: float, tol_last: Optional[float]=None) -> bool:
        """Per-joint convergence test. When ``tol_last`` is set it applies to the
        FINAL joint only (wrist roll) so a task-irrelevant wrist residual does not
        fail an otherwise-converged rewind; all other joints use ``tol``."""
        if len(q_now) != len(target):
            return False
        last = len(target) - 1
        for i, (a, b) in enumerate(zip(q_now, target)):
            t = tol_last if tol_last is not None and i == last else tol
            if abs(float(a) - float(b)) > t:
                return False
        return True

    def _hold_until_target(self, hold_q: Sequence[float], gripper_width: float, *, tol_rad: Optional[float]=None, poll_s: Optional[float]=None, tol_last_rad: Optional[float]=None) -> Optional[list[float]]:
        """Hold ``hold_q`` (re-sending it at ``stream_hz``) until the measured
        joints are close to the target and no longer moving, then return them.

        Crucial safety detail: the bridge enforces a continuous-command-stream
        watchdog (warns "no stream for 0.50s"). During verification we therefore
        keep streaming the final waypoint, never leaving a >0.5 s gap.

        Do not treat stillness alone as success: the FR3 bridge slews targets
        slowly (EMA + velocity limit), so the arm may be briefly still at the
        start of a hold before it begins moving. Success requires both proximity
        to ``hold_q`` and ``settle_stable_reads`` consecutive still frames."""
        tol = tol_rad if tol_rad is not None else self.cfg.verify_joint_tol_rad
        period = 1.0 / max(0.001, self.cfg.stream_hz)
        deadline = time.time() + (self.cfg.settle_poll_s if poll_s is None else float(poll_s))
        prev: Optional[list[float]] = None
        latest: Optional[list[float]] = None
        stable = 0
        while time.time() < deadline:
            self.bridge.send_command(list(hold_q), gripper_width)
            st = self._read_state()
            if st and 'q' in st:
                latest = [float(v) for v in st['q']]
                still = prev is not None and all((abs(a - b) < self.cfg.settle_eps_rad for a, b in zip(latest, prev)))
                near_target = self._joints_converged(latest, list(hold_q), tol=tol, tol_last=tol_last_rad)
                if near_target and still:
                    stable += 1
                    if stable >= self.cfg.settle_stable_reads:
                        return latest
                else:
                    stable = 0
                prev = latest
            time.sleep(period)
        return latest

    def _settle_until_still(self, hold_q: Sequence[float], gripper_width: float, *, poll_s: float) -> Optional[list[float]]:
        """Stream ``hold_q`` until the measured joints stop moving (steady state),
        WITHOUT asserting joint-tolerance convergence.

        This is the verify=False counterpart of :meth:`_hold_until_target`. The sim
        (and the real bridge) slews to a commanded target slowly (EMA + velocity
        limit, and in sim a rate-limited per-step cap), so returning the instant the
        last waypoint is sent leaves the arm mid-flight. A following ``close_gripper``
        would then snapshot that mid-flight (tilted, in-air) pose and close on
        nothing. Holding the final target until the arm is still lets it actually
        arrive (as close as the slow controller gets) before the gripper acts.

        Unlike :meth:`_hold_until_target` this never requires proximity to ``hold_q``
        and never fails -- residual error on a task-irrelevant joint (e.g. a
        cylindrical can's wrist yaw) is acceptable; we only need the arm to settle.
        Continuously streaming also keeps the bridge command fresh so the sim's
        TTL-scoped recovery consumer does not expire the target to an idle freeze
        and stop short of the goal."""
        period = 1.0 / max(0.001, self.cfg.stream_hz)
        deadline = time.time() + poll_s
        prev: Optional[list[float]] = None
        latest: Optional[list[float]] = None
        stable = 0
        while time.time() < deadline:
            self.bridge.send_command(list(hold_q), gripper_width)
            st = self._read_state()
            if st and 'q' in st:
                latest = [float(v) for v in st['q']]
                still = prev is not None and all((abs(a - b) < self.cfg.settle_eps_rad for a, b in zip(latest, prev)))
                if still:
                    stable += 1
                    if stable >= self.cfg.settle_stable_reads:
                        return latest
                else:
                    stable = 0
                prev = latest
            time.sleep(period)
        return latest

    def current_q(self) -> list[float]:
        st = self._read_state()
        if not st or 'q' not in st:
            raise RuntimeError('no fresh bridge state for current_q()')
        return [float(v) for v in st['q']]

    def gripper_width(self) -> float:
        """Gripper width from the stream. Prefers the hardware-measured width
        (``w_meas``, published by the bridge from the gripper gRPC ``GetState``)
        so holding checks reflect what the jaws actually did; falls back to the
        commanded width (``w_cmd``) on bridges/firmware that don't publish it."""
        st = self._read_state()
        if not st:
            raise RuntimeError('no fresh bridge state for gripper_width()')
        w = st.get('w_meas')
        if w is None:
            w = st.get('w_cmd')
        if w is None:
            w = st.get('gripper')
        if w is None:
            raise RuntimeError('bridge state has no gripper signal (w_meas/w_cmd)')
        return float(w)

    def _stream_joint_path(self, waypoints: Sequence[Sequence[float]], gripper_width: float, *, verify: bool=True, tol_rad: Optional[float]=None, tol_last_rad: Optional[float]=None, ee_accept_mm: Optional[float]=None, max_step_rad: Optional[float]=None, settle_poll_s: Optional[float]=None, settle_no_verify_poll_s: Optional[float]=None, motion_segment: Optional[str]=None) -> MotionResult:
        if not waypoints:
            return MotionResult(ok=False, reason='empty joint path')
        tol = tol_rad if tol_rad is not None else self.cfg.verify_joint_tol_rad
        step = max_step_rad if max_step_rad is not None and max_step_rad > 0.0 else self.cfg.max_exec_step_rad
        stream_wps = densify_joint_path(waypoints, max_step_rad=step)
        if not verify and len(stream_wps) > 80:
            stride = max(1, int(math.ceil(len(stream_wps) / 80)))
            stream_wps = list(stream_wps[::stride])
            if list(stream_wps[-1]) != list(waypoints[-1]):
                stream_wps.append(list(waypoints[-1]))
        period = 1.0 / max(0.001, self.cfg.stream_hz)
        for wp in stream_wps:
            self.bridge.send_command(list(wp), gripper_width)
            time.sleep(period)
        final = list(stream_wps[-1])
        if not verify:
            poll = self.cfg.settle_no_verify_poll_s if settle_no_verify_poll_s is None else settle_no_verify_poll_s
            q_now = self._settle_until_still(final, gripper_width, poll_s=float(poll))
            return MotionResult(ok=True, final_q=q_now or final)
        q_now = self._hold_until_target(final, gripper_width, tol_rad=tol, poll_s=settle_poll_s, tol_last_rad=tol_last_rad)
        if q_now is None:
            return MotionResult(ok=False, reason='no state to verify motion', final_q=list(final))
        if self._joints_converged(q_now, list(final), tol=tol, tol_last=tol_last_rad):
            return MotionResult(ok=True, final_q=q_now)
        dpos_mm, dang_deg = self._cartesian_gap_vals(list(final), q_now)
        if ee_accept_mm is not None and ee_accept_mm > 0.0:
            if dpos_mm is not None and dpos_mm <= ee_accept_mm:
                return MotionResult(ok=True, final_q=q_now, reason=f'joint gate failed but EE within tol (dpos={dpos_mm:.1f}mm<={ee_accept_mm:.1f}mm, dang={dang_deg:.1f}deg)')
        err = [float(a) - float(b) for a, b in zip(q_now, final)]
        max_err = max((abs(v) for v in err), default=0.0)
        tail = '' if tol_last_rad is None else f', tol_wrist={tol_last_rad:.3f}'
        cart = 'cart=n/a' if dpos_mm is None else f'cart: dpos={dpos_mm:.1f}mm dang={dang_deg:.1f}deg'
        return MotionResult(ok=False, reason=f'did not converge to target (tol={tol:.3f} rad{tail}, max_err={max_err:.3f}, err={[round(v, 3) for v in err]}, {cart})', final_q=q_now)

    def _interp_joint_path(self, q_start: Sequence[float], q_goal: Sequence[float]) -> list[list[float]]:
        """Bounded straight-line joint interpolation (fallback only)."""
        max_delta = max((abs(a - b) for a, b in zip(q_start, q_goal))) if q_start else 0.0
        n = max(1, int(math.ceil(max_delta / max(1e-06, self.cfg.unplanned_max_step_rad))))
        path = []
        for i in range(1, n + 1):
            t = i / n
            path.append([a + (b - a) * t for a, b in zip(q_start, q_goal)])
        return path

    def _diag_plan_to_q(self, q_start: Sequence[float], q_target: Sequence[float]) -> None:
        """DIAG (env CARRYON_DIAG_COLLISION): isolate whether the START (freeze
        pose) or the GOAL is what cuRobo flags as in-collision. plan_to_q(q, q)
        runs cuRobo's endpoint feasibility on a single state, so a trivial
        self-plan succeeds iff that state is collision-free. Also reports each
        state's lowest collision-sphere z and tool xyz for table calibration."""
        p = self.planner
        try:
            zs, zg = (p.lowest_body_z(q_start), p.lowest_body_z(q_target))
            es, eg = (p.ee_position(q_start), p.ee_position(q_target))
            start_ok = bool(p.plan_to_q(q_start, q_start))
            goal_ok = bool(p.plan_to_q(q_target, q_target))
            print(f'[DIAG-collision] start_feasible={start_ok} goal_feasible={goal_ok} | lowest_body_z start={zs} goal={zg} | ee_pos start={es} goal={eg} | q_start={[round(float(v), 3) for v in q_start]} q_target={[round(float(v), 3) for v in q_target]}', flush=True)
        except Exception as e:
            print(f'[DIAG-collision] error: {type(e).__name__}: {e}', flush=True)

    def rewind_to_pose(self, q_target: Sequence[float], *, motion_segment: str='retreat') -> MotionResult:
        try:
            q_start = self.current_q()
        except RuntimeError as e:
            return MotionResult(ok=False, reason=str(e))
        q_rewind_target = _rewind_symmetric_joint7_target(q_start, q_target)
        grip = self._hold_gripper_width()
        if not self._is_gripper_holding():
            self.open_gripper()
            grip = self._safe_gripper_width()
        if self.planner is not None:
            if os.environ.get('CARRYON_DIAG_COLLISION'):
                self._diag_plan_to_q(q_start, q_rewind_target)
            path = self.planner.plan_to_q(q_start, q_rewind_target)
            nudge = float(getattr(self.cfg, 'rewind_escape_nudge_rad', 0.0) or 0.0)
            if not path and nudge > 0.0:
                deltas = [float(b) - float(a) for a, b in zip(q_start, q_rewind_target)]
                scale = min(1.0, nudge / max(1e-06, max((abs(d) for d in deltas))))
                q_mid = [float(a) + d * scale for a, d in zip(q_start, deltas)]
                esc = self._stream_joint_path(self._interp_joint_path(q_start, q_mid), grip, verify=False, motion_segment=motion_segment)
                q_re = esc.final_q or q_mid
                path = self.planner.plan_to_q(list(q_re), q_rewind_target)
                if path:
                    q_start = list(q_re)
            if not path:
                return MotionResult(ok=False, reason='planner found no collision-free path to rewind target')
            return self._stream_joint_path(path, grip, tol_rad=self.cfg.rewind_joint_tol_rad, tol_last_rad=self.cfg.rewind_wrist_roll_tol_rad, ee_accept_mm=self.cfg.rewind_ee_accept_mm, max_step_rad=self.cfg.rewind_max_exec_step_rad, motion_segment=motion_segment)
        if self.cfg.allow_unplanned_rewind:
            path = self._interp_joint_path(q_start, q_rewind_target)
            return self._stream_joint_path(path, grip, tol_rad=self.cfg.rewind_joint_tol_rad, tol_last_rad=self.cfg.rewind_wrist_roll_tol_rad, ee_accept_mm=self.cfg.rewind_ee_accept_mm, max_step_rad=self.cfg.rewind_max_exec_step_rad, motion_segment=motion_segment)
        return MotionResult(ok=False, reason='no motion planner wired (rewind fails closed)')

    def at_rewind_target(self, q_target: Sequence[float], *, pos_tol_mm: Optional[float]=None) -> tuple[bool, str]:
        """Task-space rewind assertion: the arm is 'at' the rewind/placement target
        when its EE POSITION matches (within ``pos_tol_mm``), NOT when every joint
        matches. The wrist roll is a free, task-irrelevant DOF for a symmetric gripper
        and can sit far off; judging by EE position passes that spin while still
        failing a genuine positional miss. Falls back to a (wrist-lenient) joint check
        when no FK is available. Mirrors the ``_stream_joint_path`` EE fail-open so the
        backend rewind gate and the policy ``assert_at_rewind`` agree."""
        try:
            q_now = self.current_q()
        except RuntimeError as e:
            return (False, str(e))
        tol_mm = pos_tol_mm if pos_tol_mm is not None else self.cfg.rewind_ee_accept_mm
        dpos_mm, dang_deg = self._cartesian_gap_vals(list(q_target), q_now)
        if dpos_mm is None or tol_mm <= 0.0:
            ok = self._joints_converged(q_now, list(q_target), tol=self.cfg.rewind_joint_tol_rad, tol_last=self.cfg.rewind_wrist_roll_tol_rad)
            return (ok, '' if ok else 'arm not at rewind target (no FK; joint gate)')
        ok = dpos_mm <= tol_mm
        return (ok, '' if ok else f'arm not at rewind target (EE dpos={dpos_mm:.1f}mm>{tol_mm:.1f}mm, dang={dang_deg:.1f}deg)')

    def goto_pose(self, position: Sequence[float], quaternion_wxyz: Sequence[float], z_approach: float=0.0, verify: bool=True, gripper_width: Optional[float]=None) -> MotionResult:
        if self.planner is None:
            return MotionResult(ok=False, reason='no motion planner wired (goto_pose fails closed)')
        grip = self._safe_gripper_width() if gripper_width is None else float(gripper_width)
        try:
            q_start = self.current_q()
        except RuntimeError as e:
            return MotionResult(ok=False, reason=str(e))
        tcp = float(self.cfg.tcp_offset_m)
        if z_approach and z_approach > 0.0:
            standoff_hand = grasp_local_standoff(position, quaternion_wxyz, z_approach + tcp)
            path = self.planner.plan_to_pose(q_start, standoff_hand, quaternion_wxyz)
            if not path:
                err = getattr(self.planner, 'last_error', '') or 'unknown planner failure'
                return MotionResult(ok=False, reason=f'planner failed to reach standoff pose (hand={list(map(float, standoff_hand))}, quat={list(map(float, quaternion_wxyz))}, z_approach={float(z_approach):.3f}, tcp={tcp:.3f}): {err}')
            res = self._stream_joint_path(path, grip, verify=verify)
            if not res.ok:
                return res
            q_start = res.final_q or q_start
        hand = grasp_local_standoff(position, quaternion_wxyz, tcp)
        path = self.planner.plan_to_pose(q_start, list(hand), list(quaternion_wxyz))
        if not path:
            err = getattr(self.planner, 'last_error', '') or 'unknown planner failure'
            return MotionResult(ok=False, reason=f'planner failed to reach target pose (hand={list(map(float, hand))}, quat={list(map(float, quaternion_wxyz))}, tcp={tcp:.3f}): {err}')
        return self._stream_joint_path(path, grip, verify=verify)

    def goto_grasp_goalset(self, grasps: Sequence[GraspProposal], *, z_approach: float=0.1, verify: bool=True, plan_grasp_to_lift: bool=False, prefer_low_tilt: Optional[bool]=None, max_tilt_deg: Optional[float]=None, failopen_tilt_band_deg: Optional[float]=None, allow_tilt_failopen: Optional[bool]=None, retry_after_physical_failure: bool=True, object_key: Optional[str]=None) -> MotionResult:
        """Plan and execute a top-K grasp goalset, selecting a reachable candidate.

        This is the cap-x-style generalization seam for clutter/reachability
        failures: perception may return many grasp candidates, and cuRobo chooses
        a collision-free/reachable goal instead of betting the whole recovery on
        argmax-1. The primitive only moves to the selected grasp pose; subsequent
        ``close_gripper`` / ``lift`` / ``assert_holding`` policy steps keep the
        declarative policy orthogonal and verifiable.
        """
        if self.planner is None:
            return MotionResult(ok=False, reason='no motion planner wired (goto_grasp_goalset fails closed)')
        if not grasps:
            return MotionResult(ok=False, reason='empty grasp goalset')
        plan_fn = getattr(self.planner, 'plan_grasp_goalset', None)
        if not callable(plan_fn):
            return MotionResult(ok=False, reason='planner has no grasp-goalset primitive')
        self._goalset_lift_path = None
        self._last_descent_jam_signal = None
        self._goalset_retry_standoff_q = None
        self._goalset_retry_standoff_witness = {}
        plan_grasp_to_lift = bool(plan_grasp_to_lift) or bool(getattr(self.cfg, 'goalset_plan_lift', False))
        scene_note = self._push_scene_collision(grasps)
        try:
            return self._goalset_with_retry(grasps, z_approach=z_approach, verify=verify, plan_grasp_to_lift=plan_grasp_to_lift, prefer_low_tilt=prefer_low_tilt, max_tilt_deg=max_tilt_deg, failopen_tilt_band_deg=failopen_tilt_band_deg, allow_tilt_failopen=allow_tilt_failopen, scene_note=scene_note, retry_after_physical_failure=bool(retry_after_physical_failure), object_key=object_key or self._last_grasp_object_key)
        finally:
            clear = getattr(self.planner, 'clear_scene', None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass

    @staticmethod
    def _topdown_quaternion_wxyz(yaw_deg: float) -> list[float]:
        half = math.radians(float(yaw_deg)) / 2.0
        return [0.0, math.cos(half), math.sin(half), 0.0]

    def goto_grasp_topdown_family(self, frozen_target, fallback_grasps, *, yaw_degs=(), z_approach=None, verify=None, plan_grasp_to_lift=None, fallback_max_tilt_deg=40.0):
        """Execute one native candidate without yaw rebuilding or secondary fallback.

        Requires an upstream immutable common-geometry family with P already
        resolved once. The common-geometry binding installs this executor.
        """
        from carryon.recovery.schema import GraspProposal
        from selection_origin_binding import register
        stage = GraspProposal(position=list(frozen_target.position), quaternion=list(frozen_target.quaternion), score=frozen_target.score, source=frozen_target.source, predicted_width_m=frozen_target.predicted_width_m)
        register(self, stage, frozen_target)
        self._topdown_jam_retreat_path = None
        debug = {'schema': 'common-native-exact-family/draft-v1', 'frozen_target_xyz': list(stage.position), 'frozen_target_quaternion': list(stage.quaternion), 'topdown_attempts': [], 'fallback_used': False}
        self.last_topdown_family_debug = debug
        result = self.goto_grasp_goalset([stage], z_approach=z_approach, verify=verify, plan_grasp_to_lift=plan_grasp_to_lift, prefer_low_tilt=False, allow_tilt_failopen=False, retry_after_physical_failure=False)
        debug['topdown_attempts'].append({'stage': 'exact_native_candidate', 'ok': bool(result.ok), 'reason': str(result.reason)})
        self.last_topdown_family_debug = debug
        return result

    def retreat_after_topdown_physical_failure(self) -> MotionResult:
        """Retreat from a partial approach/descent before fresh perception."""
        path = getattr(self, '_topdown_jam_retreat_path', None)
        self._topdown_jam_retreat_path = None
        if not path:
            return MotionResult(ok=False, reason='no recorded topdown retreat path')
        try:
            grip = self._safe_gripper_width()
        except RuntimeError as exc:
            return MotionResult(ok=False, reason=str(exc))
        return self._stream_joint_path(path, grip, verify=True, motion_segment='retreat')

    def retreat_to_last_goalset_standoff(self) -> MotionResult:
        """Confirm that the current pose has reached the preplanned grasp standoff."""
        target = getattr(self, '_goalset_retry_standoff_q', None)
        witness = dict(getattr(self, '_goalset_retry_standoff_witness', {}) or {})
        debug = {'schema': 'carryon-pose-retry-retreat/v1', 'preplanned_witness': witness, 'target_q': list(target) if target is not None else None}
        self.last_pose_retry_retreat_debug = debug
        if not target:
            debug['ok'] = False
            debug['reason'] = 'no frozen goalset lift/retreat endpoint'
            return MotionResult(ok=False, reason=debug['reason'])
        try:
            q_now = self.current_q()
        except RuntimeError as exc:
            debug['ok'] = False
            debug['reason'] = str(exc)
            return MotionResult(ok=False, reason=str(exc))
        tol = float(self.cfg.verify_joint_tol_rad)
        max_error = max((abs(float(a) - float(b)) for a, b in zip(q_now, target)))
        debug.update(current_q=[float(v) for v in q_now], max_joint_error_rad=max_error, tolerance_rad=tol)
        if not self._joints_converged(q_now, target, tol=tol):
            debug['ok'] = False
            debug['reason'] = 'arm is not at frozen goalset retreat endpoint; new planning is forbidden'
            return MotionResult(ok=False, final_q=list(q_now), reason=debug['reason'])
        debug['ok'] = True
        debug['reason'] = 'already at frozen goalset preplanned retreat endpoint'
        return MotionResult(ok=True, final_q=list(q_now), reason=debug['reason'])

    def _push_scene_collision(self, grasps: Sequence[GraspProposal]) -> str:
        """Build a cuRobo collision scene from the last perception frame.

        Carves out a sphere around the grasp target (mean of candidate positions)
        so the gripper can still contact it. Returns a short note for step.reason.
        Fail-open (returns a note; leaves planner table-only on any problem)."""
        set_scene = getattr(self.planner, 'set_scene_pointcloud', None)
        if not callable(set_scene) or not self._last_scene_perception:
            return 'scene: off'
        env = os.environ.get('CARRYON_GOALSET_SCENE')
        enabled = bool(self.cfg.goalset_scene_collision) if env is None else env == '1'
        if not enabled:
            return 'scene: disabled'
        try:
            from .geometry import scene_pointcloud_world
        except Exception:
            return 'scene: import-failed'
        try:
            depth_m, intrinsics = self._last_scene_perception
            pts = scene_pointcloud_world(depth_m, intrinsics)
            if pts is None or len(pts) == 0:
                return 'scene: no-points'
            n = len(grasps)
            cx = sum((float(g.position[0]) for g in grasps)) / n
            cy = sum((float(g.position[1]) for g in grasps)) / n
            cz = sum((float(g.position[2]) for g in grasps)) / n
            installed = set_scene(pts, exclude_center_world=[cx, cy, cz], exclude_radius_m=float(self.cfg.scene_exclude_radius_m), pitch_m=float(self.cfg.scene_voxel_pitch_m))
            return f'scene: clutter n_pts={len(pts)}' if installed else 'scene: table-only'
        except Exception as e:
            return f'scene: error {type(e).__name__}'

    def _goalset_with_retry(self, grasps: Sequence[GraspProposal], *, z_approach: float, verify: bool, plan_grasp_to_lift: bool, prefer_low_tilt: Optional[bool], max_tilt_deg: Optional[float], failopen_tilt_band_deg: Optional[float], allow_tilt_failopen: Optional[bool], scene_note: str, retry_after_physical_failure: bool, object_key: Optional[str]) -> MotionResult:
        """(2) Try goalset candidates until one actually DESCENDS to the grasp.

        cuRobo plans against an approximate world; the sim physics is the ground
        truth. If the selected candidate's descent does not make real Cartesian
        progress toward the grasp (``dpos`` stays near the full standoff -- the
        observed jam), drop that candidate and re-plan over the rest. This turns
        ``topk_goalset`` into a closed loop: perception proposes many, the planner
        proposes reachable, and execution confirms the one that physically works --
        all object-agnostic (no per-object offsets)."""
        self.last_goalset_execution_debug = {'physical_motion_started': False, 'physical_descent_started': False, 'goalset_index': None, 'selected_source': None}
        grip = self._safe_gripper_width()
        grasps, ws_note = self._workspace_filter_candidates(grasps)
        if not grasps:
            return MotionResult(ok=False, reason=f'goalset rejected: every candidate outside the reachable workspace ({ws_note}) -- upstream grounding likely masked the wrong region, not a motion failure')
        if ws_note:
            scene_note = f'{scene_note} | {ws_note}'
        low_tilt = self._goalset_prefer_low_tilt() if prefer_low_tilt is None else bool(prefer_low_tilt)
        remaining = self._sort_goalset_candidates(grasps) if low_tilt else list(grasps)
        cap = self._goalset_max_tilt_deg() if max_tilt_deg is None else float(max_tilt_deg)
        spill: list = []
        cap_note = 'tilt_cap=off'
        if cap and cap > 0.0:
            low = [g for g in remaining if self._approach_tilt_deg(g) <= cap]
            high = [g for g in remaining if self._approach_tilt_deg(g) > cap]
            if low:
                remaining, spill = (low, high)
                cap_note = f'tilt_cap={cap:.0f}deg(n_low={len(low)},spill={len(high)})'
            else:
                if allow_tilt_failopen is False:
                    if high:
                        tilts = [self._approach_tilt_deg(g) for g in high]
                        tmin = min(tilts)
                        tmax = max(tilts)
                        return MotionResult(ok=False, reason=f'goalset rejected by strict tilt cap: tilt_cap={cap:.0f}deg(no_low->closed n={len(high)} min={tmin:.1f} max={tmax:.1f})')
                    return MotionResult(ok=False, reason=f'goalset rejected by strict tilt cap: tilt_cap={cap:.0f}deg(no_candidates)')
                band = float(getattr(self.cfg, 'goalset_failopen_tilt_band_deg', 0.0) or 0.0) if failopen_tilt_band_deg is None else float(failopen_tilt_band_deg)
                if band > 0.0 and high:
                    tilts = [self._approach_tilt_deg(g) for g in high]
                    tmin = min(tilts)
                    kept = [g for g, t in zip(high, tilts) if t <= tmin + band]
                    remaining = kept or high
                    cap_note = f'tilt_cap={cap:.0f}deg(no_low->flattest n={len(remaining)}/{len(high)} min={tmin:.0f}+{band:.0f}deg)'
                else:
                    cap_note = f'tilt_cap={cap:.0f}deg(no_low->all)'
        max_attempts = self._goalset_max_attempts()
        settle_poll = self._goalset_settle_poll_s()
        settle_no_verify = self._goalset_settle_no_verify_poll_s()
        order_note = 'order=tilt_asc' if low_tilt else 'order=as_given'
        scene_note = f'{scene_note} | {order_note} | {cap_note}'
        last_reason = 'goalset: no attempts'
        for attempt in range(max_attempts):
            if not remaining:
                if spill:
                    remaining, spill = (spill, [])
                else:
                    break
            try:
                q_start = self.current_q()
            except RuntimeError as e:
                return MotionResult(ok=False, reason=str(e))
            plan = self.planner.plan_grasp_goalset(q_start, list(remaining), tcp_offset_m=float(self.cfg.tcp_offset_m), pre_grasp_dis=float(z_approach), plan_grasp_to_lift=bool(plan_grasp_to_lift))
            if not plan or not bool(plan.get('ok')):
                status = (plan or {}).get('status', 'planner returned no goalset plan')
                last_reason = f'goalset planner failed: {status} | attempt={attempt + 1}/{max_attempts} | {scene_note}'
                self._drop_candidate(remaining, plan.get('goalset_index') if plan else None)
                continue
            approach = plan.get('approach_path')
            grasp_path = plan.get('grasp_path')
            if not approach or not grasp_path:
                last_reason = f'goalset planner returned incomplete approach/grasp path | {scene_note}'
                break
            idx = plan.get('goalset_index')
            diag = self._goalset_diag(remaining, idx, grasp_path)
            approach_verify = self._goalset_approach_verify(verify)
            descent_verify = self._goalset_descent_verify(verify)
            tag = f'attempt={attempt + 1}/{max_attempts} | {scene_note} | {diag} | approach_verify={approach_verify} descent_verify={descent_verify}'
            stream_kw = {'settle_poll_s': settle_poll, 'settle_no_verify_poll_s': settle_no_verify}
            try:
                if isinstance(idx, bool) or idx is None or int(idx) != idx or (not 0 <= int(idx) < len(remaining)):
                    return MotionResult(ok=False, reason='selection binding refused: invalid planner local index')
                selected = remaining[int(idx)]
                from selection_origin_binding import capture_before_motion
                binding = capture_before_motion(self, selected, int(idx))
            except (ValueError, TypeError, OverflowError) as exc:
                return MotionResult(ok=False, reason='selection binding refused: ' + str(exc))
            self.last_goalset_execution_debug['executed_selection'] = binding
            if selected is not None:
                self._last_executed_grasp_xyz_m = [float(v) for v in selected.position]
            self.last_goalset_execution_debug.update({'physical_motion_started': True, 'goalset_index': idx, 'selected_source': getattr(selected, 'source', None), 'attempt': attempt + 1})
            res = self._stream_joint_path(approach, grip, verify=approach_verify, motion_segment='standoff', **stream_kw)
            if not res.ok:
                last_reason = f'goalset approach failed: {res.reason} | {tag}'
                if not retry_after_physical_failure:
                    closest = 0
                    if res.final_q is not None:
                        closest = min(range(len(approach)), key=lambda i: sum(((float(a) - float(b)) ** 2 for a, b in zip(approach[i], res.final_q))))
                    self._topdown_jam_retreat_path = list(reversed(list(approach[:closest + 1])))
                    return MotionResult(ok=False, final_q=res.final_q, reason='physical_approach_failure_requires_fresh_grounding: ' + last_reason)
                self._drop_candidate(remaining, idx)
                continue
            self.last_goalset_execution_debug['physical_descent_started'] = True
            res = self._stream_joint_path(grasp_path, grip, verify=descent_verify, motion_segment='descent', **stream_kw)
            dpos_mm, dang_deg = self._cartesian_gap_vals(grasp_path[-1], res.final_q)
            planned_mm, _ = self._cartesian_gap_vals(grasp_path[-1], grasp_path[0])
            moved_mm, _ = self._cartesian_gap_vals(grasp_path[0], res.final_q)
            cart = f'cart: dpos={dpos_mm:.1f}mm dang={dang_deg:.1f}deg' if dpos_mm is not None else 'cart: n/a'
            descended = dpos_mm is None or dpos_mm <= float(self.cfg.descent_progress_tol_mm)
            if res.ok and descended:
                lift_path = plan.get('lift_path') or None
                self._goalset_lift_path = lift_path
                self._goalset_retry_standoff_q = [float(v) for v in lift_path[-1]] if lift_path else None
                self._goalset_retry_standoff_witness = {'schema': 'carryon-goalset-retry-standoff/v1', 'source': 'goalset_preplanned_lift_endpoint', 'executed_selection': binding, 'goalset_index': idx, 'path_waypoint_count': len(lift_path) if lift_path else 0, 'target_q': self._goalset_retry_standoff_q}
                return MotionResult(ok=True, reason=f'goalset_index={idx} | {tag} | {cart}', final_q=res.final_q)
            why = 'descent jammed' if res.ok or dpos_mm is not None else res.reason
            planned_txt = f'{planned_mm:.1f}mm' if planned_mm is not None else 'n/a'
            moved_txt = f'{moved_mm:.1f}mm' if moved_mm is not None else 'n/a'
            cart = f'{cart} | descent_ok={res.ok} planned={planned_txt} moved={moved_txt}'
            last_reason = f'goalset descent failed: {why} | {tag} | {cart}'
            mechanical_remaining = list(remaining)
            self._drop_candidate(mechanical_remaining, idx)
            mechanical_candidates_after_drop = len(mechanical_remaining) + len(spill)
            self.last_goalset_execution_debug['mechanical_candidates_after_drop'] = mechanical_candidates_after_drop
            if mechanical_candidates_after_drop == 0 and object_key and (selected is not None) and (dpos_mm is not None) and (planned_mm is not None) and (moved_mm is not None):
                stuck_z_m = self._fingertip_z_m(res.final_q)
                if stuck_z_m is not None:
                    self._last_descent_jam_signal = JamResidualSignal(object_key=str(object_key), planned_mm=float(planned_mm), moved_mm=float(moved_mm), dpos_mm=float(dpos_mm), stuck_z_m=float(stuck_z_m), current_z_m=float(selected.position[2]))
                    self.last_jam_z_reflection_debug = {'schema': 'carryon-fixv2-descent-jam-deferred/v1', 'deferred_to_postrow': True, 'current_action_unchanged': True, 'mechanical_candidates_after_drop': 0}
            if not retry_after_physical_failure:
                closest = 0
                if res.final_q is not None:
                    closest = min(range(len(grasp_path)), key=lambda i: sum(((float(a) - float(b)) ** 2 for a, b in zip(grasp_path[i], res.final_q))))
                self._topdown_jam_retreat_path = list(reversed(list(grasp_path[:closest + 1])))
                return MotionResult(ok=False, final_q=res.final_q, reason='physical_descent_jam_requires_fresh_grounding: ' + last_reason)
            self._drop_candidate(remaining, idx)
        return MotionResult(ok=False, reason=last_reason)

    @staticmethod
    def _drop_candidate(remaining: list, idx: Optional[int]) -> None:
        """Remove the just-tried candidate so re-planning explores the rest."""
        if idx is not None and 0 <= int(idx) < len(remaining):
            remaining.pop(int(idx))
        elif remaining:
            remaining.pop(0)

    def _cartesian_gap(self, q_planned: Optional[Sequence[float]], q_achieved: Optional[Sequence[float]]) -> str:
        """Cartesian EE gap between the planner's grasp config and the settled arm.

        The descent gate is joint-space (tol on every joint), but grasp success is
        really a CARTESIAN property. A large joint residual on a terminal wrist
        (e.g. joint7 roll of a symmetric gripper) can be a sub-mm TCP miss, while a
        proximal-joint residual is a real positional miss. Reporting the FK gap of
        planned vs achieved q tells us whether the failing joint tol is actually a
        meaningful grasp miss -- the deciding evidence for whether the joint gate is
        the wrong success criterion for GraspGen wrist configs."""
        dpos_mm, dang_deg = self._cartesian_gap_vals(q_planned, q_achieved)
        if dpos_mm is None:
            return 'cart: n/a'
        return f'cart: dpos={dpos_mm:.1f}mm dang={dang_deg:.1f}deg'

    def _cartesian_gap_vals(self, q_planned: Optional[Sequence[float]], q_achieved: Optional[Sequence[float]]) -> tuple[Optional[float], Optional[float]]:
        """Numeric (dpos_mm, dang_deg) EE gap between two configs, or (None, None)."""
        fk = getattr(self.planner, 'ee_pose', None)
        if fk is None or q_planned is None or q_achieved is None:
            return (None, None)
        try:
            pp = fk(list(q_planned))
            pa = fk(list(q_achieved))
            if not pp or not pa:
                return (None, None)
            (px, py, pz), pq = pp
            (ax, ay, az), aq = pa
            dpos_mm = 1000.0 * math.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)
            dot = abs(sum((float(a) * float(b) for a, b in zip(pq, aq))))
            dot = max(-1.0, min(1.0, dot))
            dang_deg = math.degrees(2.0 * math.acos(dot))
            return (dpos_mm, dang_deg)
        except Exception:
            return (None, None)

    def _fingertip_z_m(self, q: Optional[Sequence[float]]) -> Optional[float]:
        """Return settled FK height in the grasp target's fingertip convention.

        FK reports panda_hand in world coordinates; grasp targets and persistent
        corrected_z_m use fingertip/contact-center coordinates. Invert the same
        orientation-aware TCP transform used by goto_pose, using the ACTUAL FK
        orientation (not the planned grasp orientation or a fixed world-z offset).
        """
        fk = getattr(self.planner, 'ee_pose', None)
        if fk is None or q is None:
            return None
        try:
            pose = fk(list(q))
            if not pose:
                return None
            hand_position, hand_quaternion = pose
            fingertip = grasp_local_standoff(hand_position, hand_quaternion, -float(self.cfg.tcp_offset_m))
            return float(fingertip[2])
        except (IndexError, TypeError, ValueError):
            return None

    def _ee_z_m(self, q: Optional[Sequence[float]]) -> Optional[float]:
        """Return deployable FK end-effector z for a settled joint state."""
        fk = getattr(self.planner, 'ee_pose', None)
        if fk is None or q is None:
            return None
        try:
            pose = fk(list(q))
            return float(pose[0][2]) if pose else None
        except (IndexError, TypeError, ValueError):
            return None

    def _goalset_diag(self, grasps: Sequence[GraspProposal], idx: Optional[int], grasp_path: Sequence[Sequence[float]]) -> str:
        """Compact one-line geometry summary of the SELECTED goalset candidate.

        ``tilt_deg`` is the angle of the grasp approach axis (gripper local +z,
        == quaternion's 3rd rotation column) away from straight-down (world -Z):
        0 deg == pure top-down, 90 deg == horizontal side grasp. A cluster of
        high-tilt selections on the failing trials implicates 6-DoF approaches
        sweeping into clutter the cuRobo table-only world does not model."""
        try:
            g = None
            if idx is not None and 0 <= int(idx) < len(grasps):
                g = grasps[int(idx)]
            elif grasps:
                g = grasps[0]
            if g is None:
                return 'diag: no selected grasp'
            approach = grasp_approach_axis(g.quaternion)
            cos_down = max(-1.0, min(1.0, -approach[2]))
            tilt_deg = math.degrees(math.acos(cos_down))
            pos = [round(float(v), 4) for v in g.position]
            hand, _ = fingertip_to_hand_pose(g.position, g.quaternion, self.cfg.tcp_offset_m)
            hand_pos = [round(float(v), 4) for v in hand]
            tilts = sorted((self._approach_tilt_deg(c) for c in grasps))
            tmin = tilts[0] if tilts else 0.0
            tmed = tilts[len(tilts) // 2] if tilts else 0.0
            tmax = tilts[-1] if tilts else 0.0
            return f"diag: sel_idx={idx} n_cand={len(grasps)} tilt_deg={tilt_deg:.1f} approach_z={approach[2]:.2f} fingertip={pos} hand={hand_pos} tcp={float(self.cfg.tcp_offset_m):.3f} score={round(float(getattr(g, 'score', 0.0)), 3)} src={getattr(g, 'source', None)} cand_tilt[min/med/max]={tmin:.0f}/{tmed:.0f}/{tmax:.0f}"
        except Exception as e:
            return f'diag: error {type(e).__name__}: {e}'

    def _workspace_filter_candidates(self, grasps: Sequence[GraspProposal]) -> tuple[list[GraspProposal], str]:
        """Drop candidates beyond straight-line reach of the robot base.

        Object-agnostic sanity gate, not a planner: it only rejects candidates no
        7-dof tabletop arm could ever reach (default 1.1 m vs Franka 0.855 m), so
        a grounding failure surfaces as its own diagnosis instead of burning
        cuRobo attempts. Fails open (keeps everything) when the planner does not
        expose its base transform or the knob is 0."""
        reach = float(getattr(self.cfg, 'goalset_workspace_reach_m', 0.0) or 0.0)
        if reach <= 0.0 or not grasps:
            return (list(grasps), '')
        base = None
        planner_cfg = getattr(self.planner, 'config', None)
        if planner_cfg is not None:
            base = getattr(planner_cfg, 'robot_base_xyz_m', None)
        if base is None or len(tuple(base)) < 3:
            return (list(grasps), '')
        bx, by, bz = (float(base[0]), float(base[1]), float(base[2]))
        kept: list[GraspProposal] = []
        dropped = 0
        for g in grasps:
            try:
                px, py, pz = (float(g.position[0]), float(g.position[1]), float(g.position[2]))
            except Exception:
                dropped += 1
                continue
            if ((px - bx) ** 2 + (py - by) ** 2 + (pz - bz) ** 2) ** 0.5 <= reach:
                kept.append(g)
            else:
                dropped += 1
        note = f'workspace={reach:.1f}m(drop {dropped}/{len(list(grasps))})' if dropped else ''
        return (kept, note)

    @staticmethod
    def _approach_tilt_deg(g: GraspProposal) -> float:
        """Tilt of a grasp's approach axis away from straight-down (world -Z)."""
        try:
            _ax, _ay, az = grasp_approach_axis(g.quaternion)
            return math.degrees(math.acos(max(-1.0, min(1.0, -az))))
        except Exception:
            return 0.0

    def lift(self, dz: float=0.05) -> MotionResult:
        self._begin_lift_width_trace()
        try:
            return self._lift_impl(dz)
        finally:
            self._finish_lift_width_trace()

    def _lift_impl(self, dz: float=0.05) -> MotionResult:
        """Lift the hand straight up (world +Z) by ``dz`` m, holding the current
        gripper width -- a grasp confirmation under gravity (see Protocol docstring).

        The current orientation is taken from the planner's **own forward kinematics**
        (``ee_pose``, wxyz), NOT the bridge ``ee_quat``: feeding a bridge quaternion of
        a different convention straight into :meth:`plan_to_pose` aimed the IK at a
        wrong orientation and made the wrist swing wildly. Re-using the planner's FK
        keeps the target orientation byte-identical to the current one, so the move is
        a pure +Z translation. This is a hand-frame move (no ``tcp_offset``)."""
        if self.planner is None:
            return MotionResult(ok=False, reason='no motion planner wired (lift fails closed)')
        cached = getattr(self, '_goalset_lift_path', None)
        self._goalset_lift_path = None
        gentle = float(getattr(self.cfg, 'lift_max_exec_step_rad', 0.0) or 0.0)
        if cached:
            try:
                q_now = self.current_q()
            except RuntimeError as e:
                return MotionResult(ok=False, reason=str(e))
            drift = max((abs(float(a) - float(b)) for a, b in zip(q_now, cached[0])))
            if drift <= 0.1:
                if len(cached) > 40:
                    stride = max(1, int(math.ceil(len(cached) / 40)))
                    knots = list(cached[::stride])
                    if list(knots[-1]) != list(cached[-1]):
                        knots.append(list(cached[-1]))
                    cached = knots
                res = self._stream_joint_path(cached, self._hold_gripper_width(), max_step_rad=gentle if gentle > 0.0 else None)
                if res.ok:
                    res.reason = f"lift via goalset pre-planned retreat | {res.reason or ''}"
                    return res
        fk = getattr(self.planner, 'ee_pose', None)
        if fk is None:
            return MotionResult(ok=False, reason='planner has no ee_pose FK (cannot lift safely)')
        try:
            q_start = self.current_q()
        except RuntimeError as e:
            return MotionResult(ok=False, reason=str(e))
        pose = fk(q_start)
        if not pose:
            return MotionResult(ok=False, reason='forward kinematics unavailable for lift()')
        ee_pos, quat_wxyz = pose
        grip = self._hold_gripper_width()
        target = [ee_pos[0], ee_pos[1], ee_pos[2] + float(dz)]
        path = self.planner.plan_to_pose(q_start, list(target), list(quat_wxyz))
        if not path:
            return MotionResult(ok=False, reason='planner failed to reach lift pose')
        gentle = float(getattr(self.cfg, 'lift_max_exec_step_rad', 0.0) or 0.0)
        return self._stream_joint_path(path, grip, max_step_rad=gentle if gentle > 0.0 else None)

    def vertical_lift_to_clearance(self, dz: float) -> MotionResult:
        """Move a held object straight up before lateral pre-place transport.

        This is deliberately a single scalar primitive. It reuses the proven
        pure-world-+Z live lift implementation, but never consumes or replays a
        cached goalset retreat: the ordinary post-grasp ``lift`` has already
        completed before this clearance move.
        """
        if not math.isfinite(float(dz)) or not 0.0 < float(dz) <= 0.2:
            return MotionResult(ok=False, reason='vertical_lift_to_clearance dz must be in (0, 0.20] m')
        self._goalset_lift_path = None
        return self.lift(float(dz))

    def _safe_gripper_width(self) -> float:
        try:
            return self.gripper_width()
        except RuntimeError:
            return self.cfg.gripper_open_width

    def _hold_gripper_width(self) -> float:
        """Width to stream while HOLDING during a move (lift/rewind).

        Prefers the last explicit gripper COMMAND (open/close target) over the
        measured width. On a binarising shim, feeding back a held object's
        measured width (wider than the close target) flips the gripper to OPEN
        and drops the object on the first lift frame."""
        if self._last_gripper_cmd_width is not None:
            return self._last_gripper_cmd_width
        return self._safe_gripper_width()

    def _is_gripper_holding(self) -> bool:
        """True iff the jaws are measurably open around an object (something
        is between them). Decides whether a retreat/rewind/escape-nudge move
        may open the gripper first (Obs 2): a FAILED grasp leaves the jaws
        closed on AIR, and moving through the scene closed-empty is
        ineffective and knocks objects over. Fail-safe: on any uncertainty
        assume HOLDING, so a real grasp is never opened."""
        try:
            st = self._read_state() or {}
        except Exception:
            return True
        w = st.get('w_meas')
        if w is None:
            w = st.get('w_cmd')
        if w is None:
            return True
        return float(w) > 0.006

    def _command_gripper(self, width: float) -> MotionResult:
        try:
            q_hold = self.current_q()
        except RuntimeError as e:
            return MotionResult(ok=False, reason=str(e))
        period = 1.0 / max(0.001, self.cfg.stream_hz)
        n = max(1, int(self.cfg.gripper_dwell_s / period))
        self._last_gripper_cmd_width = float(width)
        for _ in range(n):
            self.bridge.send_command(q_hold, width)
            time.sleep(period)
        return MotionResult(ok=True, final_q=q_hold)

    def open_gripper(self) -> MotionResult:
        return self._command_gripper(self.cfg.gripper_open_width)

    def close_gripper(self, width: Optional[float]=None) -> MotionResult:
        """Command the jaws shut. ``width`` (meters) overrides the configured close
        target per-call, so a declarative policy can tune the grip for a specific
        object (a thinner object needs a tighter close to actually pinch it).
        ``None`` uses the rig default ``cfg.gripper_close_width``."""
        return self._command_gripper(self.cfg.gripper_close_width if width is None else float(width))

    def _select_detector(self, source: str):
        """Resolve the requested grasp source through the configured detector registry."""
        det = self.detector
        mode = (source or '').lower()
        if mode in self.detector_sources:
            return (self.detector_sources[mode], None)
        is_hybrid = hasattr(det, 'primary') and hasattr(det, 'fallback')
        if mode == '':
            return (det, None)
        if mode == 'auto':
            return (det, 'auto' if is_hybrid else None)
        if mode in ('cgn', 'primary', 'contact_graspnet'):
            return (None, None)
        if mode in ('topdown', 'fallback', 'mask', 'mask_depth'):
            return (getattr(det, 'fallback') if is_hybrid else None, None)
        if mode in ('multiview_obb', 'multiview', 'obb'):
            mv = getattr(det, 'multiview_obb', None)
            if mv is not None:
                return (mv, None)
            return (det, 'fallback' if is_hybrid else None)
        return (det, None)

    def _fetch_perception_inputs(self):
        """Return ``(rgb, depth_m, intrinsics)`` from the camera, or ``None``
        on any fetch failure. Centralised so ``perceive_grasp`` and
        ``perceive_grasps`` share one HTTP burst. On failure, records which fetch
        stage threw in ``last_perceive_reason`` (camera contention / a dead camera
        looks identical to "could not ground object" without this)."""
        cam = self.cfg.grasp_camera
        stage = 'intrinsics'
        try:
            intrinsics = self.gello.fetch_intrinsics(cam)
            stage = 'depth'
            depth_m = self.gello.fetch_depth_meters(cam)
            stage = 'rgb'
            fetch_raw = getattr(self.gello, 'fetch_raw_rgb_jpeg', None)
            rgb_jpeg = fetch_raw(cam) if callable(fetch_raw) else self.gello.fetch_rgb_jpeg(cam)
        except Exception as e:
            self.last_perceive_reason = f'camera fetch failed at {stage} (camera={cam!r}): {type(e).__name__}: {e}'
            return None
        self._last_scene_bundle_identity = None
        if self.semantic_scene_bundle_required:
            try:
                self._last_scene_bundle_identity = scene_bundle_identity(rgb_jpeg, depth_m, intrinsics, captured_unix_ns=time.time_ns())
            except Exception as exc:
                self._last_scene_bundle_identity = {'schema': 'carryon-scene-bundle-identity/v1', 'error': f'{type(exc).__name__}: {exc}'}
        return (rgb_jpeg, depth_m, intrinsics)

    def _grounding_failure_reason(self, detector, object_hint: str) -> str:
        """Why ``detector.detect`` returned None: surface SAM3's ``last_error`` if
        the segmenter recorded one (grounding failed), else report no candidate."""
        det_err = getattr(detector, 'last_error', '') or ''
        if det_err:
            return f'detector failed for {object_hint!r}: {det_err}'
        seg = getattr(detector, 'segmenter', None) or getattr(getattr(detector, 'fallback', None), 'segmenter', None)
        err = getattr(seg, 'last_error', '') if seg is not None else ''
        if err:
            return f'SAM3 grounding failed for {object_hint!r}: {err}'
        return f'detector found no grasp candidate for {object_hint!r} (mask/depth yielded none)'

    def verify_object_identity(self, object_hint: str) -> tuple[bool, str]:
        """Pre-grasp identity check (RATs-style): crop the grounded mask region,
        mark it on the full frame, and ask the reflection VLM whether it is the
        expected object. Fails OPEN on infra problems (no VLM wired, no mask
        recorded, API error) -- an identity check must never be the reason a
        working recovery dies -- and fails CLOSED only on an explicit mismatch
        verdict, which routes into the existing retry machinery."""
        vlm = getattr(self, 'identity_vlm', None)
        if vlm is None:
            return (True, 'identity check skipped (no VLM wired)')
        seg = getattr(self.detector, 'segmenter', None) or getattr(getattr(self.detector, 'fallback', None), 'segmenter', None)
        mask = getattr(seg, 'last_grounding_mask', None) if seg is not None else None
        rgb = getattr(seg, 'last_grounding_rgb', None) if seg is not None else None
        if mask is None or rgb is None:
            return (True, 'identity check skipped (no grounding mask recorded)')
        try:
            import asyncio
            import base64
            import json as _json
            import cv2
            import numpy as np
            if isinstance(rgb, (bytes, bytearray)):
                img = cv2.imdecode(np.frombuffer(rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.asarray(rgb)
            mm = np.asarray(mask, dtype=bool)
            if img is None or mm.shape != img.shape[:2] or (not mm.any()):
                return (True, 'identity check skipped (mask/frame mismatch)')
            ys, xs = np.nonzero(mm)
            y0, y1 = (int(ys.min()), int(ys.max()))
            x0, x1 = (int(xs.min()), int(xs.max()))
            H, W = mm.shape
            pad = 24
            side = max(80, y1 - y0 + 2 * pad, x1 - x0 + 2 * pad)
            cy, cx = ((y0 + y1) // 2, (x0 + x1) // 2)
            ya = max(0, min(H - side, cy - side // 2))
            xa = max(0, min(W - side, cx - side // 2))
            crop = img[ya:ya + side, xa:xa + side]
            crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_CUBIC)
            if float(np.asarray(crop, dtype=np.float32).std()) < 5.0:
                return (True, 'identity check skipped (blank/uniform crop -- render race?)')
            full = img.copy()
            cv2.drawMarker(full, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)

            def _b64(arr):
                ok, buf = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                return base64.b64encode(buf.tobytes()).decode('ascii')
            system = 'You verify object identity for a robot about to grasp. The first image is a zoomed crop of the grasp target; the second is the full workspace with the target marked by a magenta cross. Return JSON only: {"matches": true/false, "actual": "<what the marked object is>"} -- matches is whether the marked object is the object the TASK should manipulate (packaging may differ from the name; judge by plausibility, not exact wording). The expected-object name comes from a visual judge and is occasionally wrong; when a task instruction is given, IT is authoritative over the name.'
            instr = str(getattr(self, 'task_instruction', '') or '').strip()
            expected_txt = f'Expected object: {object_hint!r}.'
            if instr:
                expected_txt += f' Task instruction: {instr!r}.'
            expected_txt += ' JSON only.'
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [{'type': 'text', 'text': expected_txt}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + _b64(crop)}}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + _b64(full)}}]}]
            loop = getattr(self, '_identity_loop', None)
            if loop is None or loop.is_closed():
                loop = asyncio.new_event_loop()
                self._identity_loop = loop
            try:
                resp = loop.run_until_complete(vlm.plan_chat(messages, response_schema_hint=False))
            except TypeError:
                resp = loop.run_until_complete(vlm.plan_chat(messages))
            text = getattr(resp, 'text', None) or getattr(resp, 'content', None) or str(resp)
            import re as _re
            m = _re.search('\\{.*\\}', text, _re.DOTALL)
            verdict = _json.loads(m.group(0)) if m else None
            if not isinstance(verdict, dict) or 'matches' not in verdict:
                return (True, f'identity check failed open (unparseable verdict: {text[:80]!r})')
            self.last_perceive_debug['identity_verdict'] = verdict
            try:
                from pathlib import Path as _P
                import time as _t
                _dbg = _P('logs/identity_debug')
                _dbg.mkdir(parents=True, exist_ok=True)
                _tag = f'{int(_t.time())}_{object_hint.replace(chr(32), chr(95))}_' + ('match' if bool(verdict.get('matches')) else 'MISMATCH_' + str(verdict.get('actual', '?')).replace(chr(32), chr(95))[:24])
                cv2.imwrite(str(_dbg / f'{_tag}_crop.jpg'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            except Exception:
                pass
            if bool(verdict.get('matches')):
                return (True, f'identity confirmed: {object_hint!r}')
            actual = str(verdict.get('actual', '?')).strip()
            try:
                self.identity_rejected.append(mm)
                del self.identity_rejected[:-4]
            except Exception:
                pass
            return (False, f'identity mismatch: expected {object_hint!r}, VLM saw {actual!r} at the grasp target')
        except Exception as e:
            return (True, f'identity check failed open ({type(e).__name__}: {e})')

    def exclude_destination(self, dest_hint: str) -> bool:
        """Exclude the placement destination from recovery grasp candidates.

        Return True if a destination mask was added. Skip exclusion if perception or
        camera access fails, allowing recovery to continue."""
        dest = str(dest_hint or '').strip()
        if not dest:
            return False
        seg = getattr(self.detector, 'segmenter', None) or getattr(getattr(self.detector, 'fallback', None), 'segmenter', None)
        if seg is None or not hasattr(seg, 'segment'):
            return False
        try:
            import numpy as _np
            inputs = self._fetch_perception_inputs()
            if inputs is None:
                return False
            mask = seg.segment(inputs[0], prompt=dest)
            if mask is None:
                print('[regrasp] destination %r not grounded; no pre-exclusion' % dest, flush=True)
                return False
            mm = _np.asarray(mask, dtype=bool)
            if not mm.any():
                return False
            self.identity_rejected.append(mm)
            del self.identity_rejected[:-4]
            self._place_dest_hint = dest
            print('[regrasp] destination %r pre-excluded from grasp candidates (%d px)' % (dest, int(mm.sum())), flush=True)
            return True
        except Exception as e:
            print('[regrasp] destination pre-exclusion failed open: %s: %s' % (type(e).__name__, e), flush=True)
            return False

    @staticmethod
    def _is_container_destination(destination: str) -> bool:
        """Classify a destination noun without consulting scene identity/pose."""
        noun = ' '.join(str(destination or '').strip().lower().replace('_', ' ').split())
        if noun.startswith('the '):
            noun = noun[4:]
        container_nouns = ('basket', 'bowl', 'bin', 'tray', 'plate', 'receptacle', 'container')
        return any((noun == item or noun.endswith(' ' + item) for item in container_nouns))

    @staticmethod
    def _opening_from_rim_points(_np, points):
        """Estimate a container centre from the RGB-D observation."""
        pts = _np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 64:
            raise ValueError('insufficient container point support')
        if not _np.isfinite(pts).all():
            raise ValueError('non-finite container point support')
        z_cut = float(_np.percentile(pts[:, 2], 80))
        rim = pts[pts[:, 2] >= z_cut]
        if len(rim) < 32:
            raise ValueError('insufficient upper-rim support')
        origin = _np.median(rim[:, :2], axis=0)
        _u, singular, vh = _np.linalg.svd(rim[:, :2] - origin, full_matrices=False)
        if len(singular) != 2 or float(singular[1]) < 0.0001:
            raise ValueError('degenerate upper-rim support')
        uv = (rim[:, :2] - origin) @ vh.T
        low = _np.percentile(uv, 5, axis=0)
        high = _np.percentile(uv, 95, axis=0)
        span = high - low
        if float(_np.min(span)) < 0.02:
            raise ValueError('upper-rim support span is too small')
        center_xy = origin + 0.5 * (low + high) @ vh
        rim_z = float(_np.percentile(rim[:, 2], 95))
        if not _np.isfinite([center_xy[0], center_xy[1], rim_z]).all():
            raise ValueError('invalid container opening estimate')
        diagnostics = {'rim_points': int(len(rim)), 'rim_z_cut': z_cut, 'rim_span_axis_0': float(span[0]), 'rim_span_axis_1': float(span[1])}
        return (float(center_xy[0]), float(center_xy[1]), rim_z, diagnostics)

    def ground_destination(self) -> MotionResult:
        """Ground the instruction destination from current production RGB-D."""
        import numpy as _np
        from .geometry import _masked_points_world
        self._place_destination_grounding = None
        dest = getattr(self, '_place_dest_hint', '') or ''
        if not dest:
            try:
                from .integration import _destination_from_instruction
                dest = _destination_from_instruction(getattr(self, 'task_instruction', '')) or ''
            except Exception:
                dest = ''
        if not dest:
            return MotionResult(ok=False, reason='no place destination in instruction')
        seg = getattr(self.detector, 'segmenter', None) or getattr(getattr(self.detector, 'fallback', None), 'segmenter', None)
        if seg is None or not hasattr(seg, 'segment'):
            return MotionResult(ok=False, reason='no segmenter for place destination')
        inputs = self._fetch_perception_inputs()
        if inputs is None:
            return MotionResult(ok=False, reason=self.last_perceive_reason or 'camera fetch failed')
        rgb_jpeg, depth_m, intrinsics = inputs
        try:
            mask = seg.segment(rgb_jpeg, prompt=dest)
        except Exception as exc:
            return MotionResult(ok=False, reason=f'place destination segment failed: {type(exc).__name__}: {exc}')
        if mask is None:
            return MotionResult(ok=False, reason=f'place destination {dest!r} not grounded')
        mm = _np.asarray(mask, dtype=bool)
        if not mm.any():
            return MotionResult(ok=False, reason=f'place destination {dest!r} empty mask')
        pts = _masked_points_world(_np, mm, depth_m, intrinsics)
        if pts is None or len(pts) == 0:
            return MotionResult(ok=False, reason='place destination depth back-projection failed')
        destination_semantics = 'solid_surface'
        geometry_diagnostics = {}
        if self._is_container_destination(dest):
            try:
                cx, cy, top_z, geometry_diagnostics = self._opening_from_rim_points(_np, pts)
                destination_semantics = 'container_opening'
            except (ValueError, _np.linalg.LinAlgError) as exc:
                return MotionResult(ok=False, reason=f'container opening geometry failed closed: {exc}')
        else:
            cx = float(_np.median(pts[:, 0]))
            cy = float(_np.median(pts[:, 1]))
            top_z = float(_np.percentile(pts[:, 2], 90))
        if not _np.isfinite([cx, cy, top_z]).all():
            return MotionResult(ok=False, reason='place destination world point non-finite')
        self._place_destination_grounding = {'destination': dest, 'center_x': cx, 'center_y': cy, 'top_z': top_z, 'mask_pixels': int(mm.sum()), 'destination_semantics': destination_semantics, 'geometry_diagnostics': geometry_diagnostics, 'grounded_at_s': time.monotonic()}
        print('[place] grounded current destination %r xy=(%.3f, %.3f) top_z=%.3f mask_pixels=%d semantics=%s' % (dest, cx, cy, top_z, int(mm.sum()), destination_semantics), flush=True)
        return MotionResult(ok=True)

    def goto_place(self, *, clearance_m: float=0.03, release_bias_m: float=0.02, standoff_only: bool=False) -> MotionResult:
        """Carry the held object to release height or a closed-gripper standoff."""
        grounding = getattr(self, '_place_destination_grounding', None)
        if not isinstance(grounding, dict):
            grounded = self.ground_destination()
            if not grounded.ok:
                return grounded
            grounding = getattr(self, '_place_destination_grounding', None)
        self._place_destination_grounding = None
        if not isinstance(grounding, dict):
            return MotionResult(ok=False, reason='place destination grounding unavailable')
        dest = str(grounding['destination'])
        cx = float(grounding['center_x'])
        cy = float(grounding['center_y'])
        top_z = float(grounding['top_z'])
        try:
            bias = max(0.0, min(0.08, float(release_bias_m)))
        except (TypeError, ValueError):
            return MotionResult(ok=False, reason='invalid place release bias')
        release_z = top_z + bias
        if self.planner is None:
            return MotionResult(ok=False, reason='no motion planner wired (goto_place fails closed)')
        fk = getattr(self.planner, 'ee_pose', None)
        if fk is None:
            return MotionResult(ok=False, reason='planner has no ee_pose FK (cannot carry safely)')
        try:
            q_now = self.current_q()
        except RuntimeError as exc:
            return MotionResult(ok=False, reason=str(exc))
        pose = fk(q_now)
        if not pose:
            return MotionResult(ok=False, reason='forward kinematics unavailable for goto_place()')
        _ee_pos, quat = pose
        approach_clearance = max(0.06, 2.0 * float(clearance_m))
        goal_z = release_z + approach_clearance if standoff_only else release_z
        goal = [cx, cy, goal_z]
        print('[place] destination %r -> goal xy=(%.3f, %.3f) top_z=%.3f release_z=%.3f goal_z=%.3f mode=%s' % (dest, cx, cy, top_z, release_z, goal_z, 'standoff_only' if standoff_only else 'release'), flush=True)
        return self.goto_pose(goal, quat, z_approach=0.0 if standoff_only else approach_clearance, gripper_width=self._hold_gripper_width())

    def verify_place(self, object_hint: str, *, settle_s: float=1.0, min_overlap: float=0.15) -> tuple[bool, str]:
        """Deployable RGB verification for complete-mode placement.

        Both the moved object and instruction-derived destination are segmented
        from the post-release camera frame. No simulator pose, reward, task
        success predicate, or benchmark object id is consulted. A missing mask
        fails closed.
        """
        import time as _time
        import numpy as _np
        dest = getattr(self, '_place_dest_hint', '') or ''
        if not dest:
            try:
                from .integration import _destination_from_instruction
                dest = _destination_from_instruction(getattr(self, 'task_instruction', '')) or ''
            except Exception:
                dest = ''
        obj = str(object_hint or '').strip()
        if not obj or not dest:
            return (False, 'object or destination prompt unavailable')
        seg = getattr(self.detector, 'segmenter', None) or getattr(getattr(self.detector, 'fallback', None), 'segmenter', None)
        if seg is None or not hasattr(seg, 'segment'):
            return (False, 'no segmenter for deployable place verification')
        _time.sleep(max(0.0, min(float(settle_s), 3.0)))
        inputs = self._fetch_perception_inputs()
        if inputs is None:
            return (False, self.last_perceive_reason or 'post-release camera fetch failed')
        rgb_jpeg = inputs[0]
        try:
            obj_mask = seg.segment(rgb_jpeg, prompt=obj)
            dest_mask = seg.segment(rgb_jpeg, prompt=dest)
        except Exception as e:
            return (False, f'place verification segmentation failed: {type(e).__name__}: {e}')
        if obj_mask is None or dest_mask is None:
            return (False, 'post-release object or destination mask missing')
        om = _np.asarray(obj_mask, dtype=bool)
        dm = _np.asarray(dest_mask, dtype=bool)
        if om.shape != dm.shape or not om.any() or (not dm.any()):
            return (False, 'post-release masks empty or shape-mismatched')
        overlap = float(_np.logical_and(om, dm).sum()) / float(max(1, om.sum()))
        oys, oxs = _np.nonzero(om)
        oy, ox = (float(oys.mean()), float(oxs.mean()))
        dys, dxs = _np.nonzero(dm)
        centroid_inside = float(dxs.min()) <= ox <= float(dxs.max()) and float(dys.min()) <= oy <= float(dys.max())
        bbox_contained = int(oxs.min()) >= int(dxs.min()) and int(oxs.max()) <= int(dxs.max()) and (int(oys.min()) >= int(dys.min())) and (int(oys.max()) <= int(dys.max()))
        ok = overlap >= max(0.0, min(float(min_overlap), 1.0)) or bbox_contained
        return (ok, f'visual place verification overlap={overlap:.3f} centroid_in_destination={centroid_inside} object_bbox_contained={bbox_contained} object_centroid_px=[{ox:.1f},{oy:.1f}] object_bbox_px=[{int(oxs.min())},{int(oys.min())},{int(oxs.max())},{int(oys.max())}] destination_bbox_px=[{int(dxs.min())},{int(dys.min())},{int(dxs.max())},{int(dys.max())}]')

    def ground_target(self, object_hint: str, *, plan: list[str], target_xy: Optional[tuple]=None, target_px: Optional[tuple]=None) -> GroundingResult:
        """Return explicit target-grounding evidence for the grasp detector.

        This first implementation keeps the hot path deterministic: it packages
        already-available task/Judge/VLA priors into a typed result. Heavier point
        models and multiview grounding can be added behind this same primitive.
        """
        for prim in plan:
            if prim == 'judge_point' and target_px is not None:
                return GroundingResult(mode='judge_point', provenance='RecoveryContext.target_px', normalized_point=[float(target_px[0]), float(target_px[1])], object_hint=object_hint, plan=list(plan))
            if prim == 'recent_contact' and target_xy is not None:
                return GroundingResult(mode='recent_contact', provenance='RecoveryContext.target_xy_world', world_point=[float(target_xy[0]), float(target_xy[1])], object_hint=object_hint, plan=list(plan))
            if prim == 'task_target':
                break
        return GroundingResult(mode='task_target', provenance='object_hint', object_hint=object_hint, plan=list(plan))

    def perceive_grasp(self, object_hint: str, source: str='topdown', *, target_xy: Optional[tuple]=None, target_px: Optional[tuple]=None, grounding: Optional[GroundingResult]=None, sam_text_prompts: Optional[list[str]]=None) -> Optional[GraspProposal]:
        self.last_perceive_reason = ''
        if self.detector is None:
            self.last_perceive_reason = 'no grasp detector wired'
            return None
        detector, prefer_override = self._select_detector(source)
        if detector is None:
            self.last_perceive_reason = f'no detector for source={source!r}'
            return None
        inputs = self._fetch_perception_inputs()
        if inputs is None:
            return None
        rgb_jpeg, depth_m, intrinsics = inputs
        kwargs = dict(rgb=rgb_jpeg, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint)
        extra = {}
        if self._last_scene_bundle_identity is not None:
            extra['scene_bundle_identity'] = dict(self._last_scene_bundle_identity)
        if prefer_override is not None:
            extra['prefer'] = prefer_override
        if grounding is not None:
            extra['grounding'] = grounding
            if grounding.normalized_point is not None:
                target_px = tuple(grounding.normalized_point)
            if grounding.world_point is not None and len(grounding.world_point) >= 2:
                target_xy = tuple(grounding.world_point[:2])
        if target_xy is not None:
            extra['target_xy'] = target_xy
        if target_px is not None:
            extra['target_px'] = target_px
        if sam_text_prompts:
            extra['sam_text_prompts'] = list(sam_text_prompts)
        rejected = getattr(self, 'identity_rejected', None)
        if rejected:
            extra['exclude_masks'] = list(rejected)
        scene_ref = getattr(self, 'scene_reference', None)
        if scene_ref is not None:
            g_plan = getattr(grounding, 'plan', None) if grounding is not None else None
            if g_plan is None or 'scene_diff' in g_plan:
                extra['scene_reference'] = scene_ref
        try:
            grasp = detector.detect(**extra, **kwargs)
        except TypeError:
            for drop in ('scene_bundle_identity', 'exclude_masks', 'scene_reference', 'sam_text_prompts', 'grounding', 'target_px', 'prefer', 'target_xy'):
                if drop in extra:
                    print('[backend] perceive_grasp: detector rejected kwargs; dropping %r and retrying (capability silently reduced)' % (drop,), flush=True)
                extra.pop(drop, None)
                try:
                    grasp = detector.detect(**extra, **kwargs)
                    break
                except TypeError:
                    continue
            else:
                grasp = detector.detect(**kwargs)
        if grasp is None:
            self.last_perceive_reason = self._grounding_failure_reason(detector, object_hint)
        return grasp

    def reflect_parameter_memory_at_recovery_end(self, steps) -> dict:
        """Earn one future-row z prior from grasp-no-hold or descent-jam evidence."""
        flattened = []

        def visit(items) -> None:
            for item in items or []:
                flattened.append(item)
                visit(getattr(item, 'children', None))
        visit(steps)
        close_seen = any((getattr(item, 'op', '') == 'close_gripper' and bool(getattr(item, 'ok', False)) for item in flattened))
        assert_failed = any((getattr(item, 'op', '') == 'assert_holding' and (not bool(getattr(item, 'ok', False))) for item in flattened))
        evidence = {'schema': 'carryon-fixv2-parameter-postrow-evidence/v1', 'close_gripper_succeeded': close_seen, 'assert_holding_failed': assert_failed, 'descent_jam_evidence_present': self._last_descent_jam_signal is not None, 'triggered': False, 'trigger': None, 'write_succeeded': False, 'vlm_call_attempted': False, 'current_action_unchanged': True}
        reflector = self.jam_z_reflector
        target = self._last_executed_grasp_xyz_m
        key = self._last_grasp_object_key or self.task_instruction
        if reflector is None:
            return evidence
        if reflector.write_trigger != 'postrow_failure_evidence':
            reflector.record_instrument_event(event='WRITE_SKIPPED_TRIGGER_DISABLED', object_key=key)
            return evidence
        if not key:
            reflector.record_instrument_event(event='WRITE_SKIPPED_KEY_ABSENT', object_key=None)
            return evidence
        resolution = self.semantic_reflector.resolve_object_key(key) if self.semantic_reflector is not None else None
        if resolution is not None and resolution['resolved_key'] is not None:
            key = resolution['resolved_key']
            if resolution['resolution'] == 'REGISTRY_SUBSTRING_UNIQUE':
                reflector.record_instrument_event(event='WRITE_KEY_RESOLVED_BY_REGISTRY', object_key=resolution['original_key'], candidates=resolution['candidate_keys'], resolved_key=key)
        elif resolution is not None:
            reflector.record_instrument_event(event='WRITE_KEY_UNRESOLVED_FALLBACK_ORIGINAL', object_key=key, candidates=resolution['candidate_keys'])
        if close_seen and assert_failed and (target is not None) and (len(target) >= 3):
            decision = reflector.reflect_grasp_no_hold(GraspNoHoldSignal(object_key=key, current_z_m=float(target[2]), close_gripper_succeeded=True, assert_holding_failed=True))
            evidence['trigger'] = 'grasp_no_hold'
        elif self._last_descent_jam_signal is not None:
            source_signal = self._last_descent_jam_signal
            signal = JamResidualSignal(object_key=key, planned_mm=source_signal.planned_mm, moved_mm=source_signal.moved_mm, dpos_mm=source_signal.dpos_mm, stuck_z_m=source_signal.stuck_z_m, current_z_m=source_signal.current_z_m)
            decision = reflector.reflect(signal, target_before_xyz_m=[0.0, 0.0, float(signal.current_z_m)])
            evidence['trigger'] = 'descent_jam'
        else:
            reflector.record_instrument_event(event='WRITE_SKIPPED_NO_EVIDENCE', object_key=key, resolved_key=key)
            return evidence
        evidence.update({'triggered': True, 'object_key': key, 'write_succeeded': bool(decision.write_succeeded), 'vlm_call_attempted': bool(decision.vlm_call_attempted), 'value_source': decision.source, 'parse_error': decision.parse_error, 'persistent_keys': sorted(reflector.persistent_keys)})
        self.last_jam_z_reflection_debug = dict(evidence)
        return evidence

    def reflect_semantic_memory_at_recovery_end(self, steps) -> dict:
        """Propose one future-row REGROUND entry from failed mechanical evidence."""
        failed = []

        def visit(items) -> None:
            for item in items or []:
                op = str(getattr(item, 'op', ''))
                ok = bool(getattr(item, 'ok', False))
                if op and (not ok):
                    failed.append({'op': op, 'reason': str(getattr(item, 'reason', ''))})
                visit(getattr(item, 'children', None))
        visit(steps)
        evidence = {'schema': 'carryon-fixv2-semantic-postrow-evidence/v1', 'object_key': self._last_grasp_object_key or self.task_instruction or None, 'failed_step_count': len(failed), 'triggered': False, 'write_succeeded': False, 'vlm_call_attempted': False, 'current_action_unchanged': True}
        if self.semantic_reflector is None:
            return evidence
        evidence = self.semantic_reflector.reflect(object_key=self._last_grasp_object_key or self.task_instruction or '', failed_steps=failed)
        self.last_semantic_reflection_debug = dict(evidence)
        return evidence

    def perceive_grasps(self, object_hint: str, *, top_k: int=15, source: str='topdown', target_xy: Optional[tuple]=None, target_px: Optional[tuple]=None, grounding: Optional[GroundingResult]=None, sam_text_prompts: Optional[list[str]]=None) -> list[GraspProposal]:
        """Top-K grasp candidates for cap-x-style cuRobo goalset planning.

        Returns up to ``top_k`` :class:`GraspProposal`\\ s sorted by score
        (descending). The motion layer is expected to iterate IK + collision-free
        planning over the list and pick the first one that's reachable -- the
        argmax-1 grasp may be unreachable for kinematic / collision reasons even
        when it has the highest score, and rerunning perception each time is
        wasteful (cap-x's ``parse_grasp_poses_for_curobo(top_k=15)`` +
        ``plan_to_grasp_poses(for idx in range(n))``).

        Returns ``[]`` when the detector or fetch fails. ``source`` follows the
        same semantics as :meth:`perceive_grasp`."""
        self._last_grasp_object_key = str(object_hint).strip().lower() or None
        self.last_perceive_debug = {}
        if self.detector is None or top_k <= 0:
            self.last_perceive_reason = 'no grasp detector wired' if self.detector is None else 'top_k must be > 0'
            return []
        detector, prefer_override = self._select_detector(source)
        if detector is None:
            self.last_perceive_reason = f'no detector for source={source!r}'
            return []
        inputs = self._fetch_perception_inputs()
        if inputs is None:
            return []
        rgb_jpeg, depth_m, intrinsics = inputs
        self._last_scene_perception = (depth_m, intrinsics)
        kwargs = dict(rgb=rgb_jpeg, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint, top_k=int(top_k))
        extra = {}
        if self._last_scene_bundle_identity is not None:
            extra['scene_bundle_identity'] = dict(self._last_scene_bundle_identity)
        if prefer_override is not None:
            extra['prefer'] = prefer_override
        if grounding is not None:
            extra['grounding'] = grounding
            if grounding.normalized_point is not None:
                target_px = tuple(grounding.normalized_point)
            if grounding.world_point is not None and len(grounding.world_point) >= 2:
                target_xy = tuple(grounding.world_point[:2])
        if target_xy is not None:
            extra['target_xy'] = target_xy
        if target_px is not None:
            extra['target_px'] = target_px
        if sam_text_prompts:
            extra['sam_text_prompts'] = list(sam_text_prompts)
        rejected = getattr(self, 'identity_rejected', None)
        if rejected:
            extra['exclude_masks'] = list(rejected)
        scene_ref = getattr(self, 'scene_reference', None)
        if scene_ref is not None:
            g_plan = getattr(grounding, 'plan', None) if grounding is not None else None
            if g_plan is None or 'scene_diff' in g_plan:
                extra['scene_reference'] = scene_ref
        fn = getattr(detector, 'detect_many', None)
        if callable(fn):
            try:
                out = list(fn(**extra, **kwargs))
                self.last_perceive_debug = dict(getattr(detector, 'last_debug', {}) or {})
                if not out:
                    self.last_perceive_reason = self._grounding_failure_reason(detector, object_hint)
                return out
            except TypeError:
                for drop in ('scene_bundle_identity', 'exclude_masks', 'scene_reference', 'sam_text_prompts', 'grounding', 'target_px', 'prefer', 'target_xy'):
                    if drop in extra:
                        print('[backend] perceive_grasps: detector rejected kwargs; dropping %r and retrying (capability silently reduced)' % (drop,), flush=True)
                    extra.pop(drop, None)
                    try:
                        out = list(fn(**extra, **kwargs))
                        self.last_perceive_debug = dict(getattr(detector, 'last_debug', {}) or {})
                        if not out:
                            self.last_perceive_reason = self._grounding_failure_reason(detector, object_hint)
                        return out
                    except TypeError:
                        continue
                pass
        single = self.perceive_grasp(object_hint, source=source, target_xy=target_xy, target_px=target_px, grounding=grounding, sam_text_prompts=sam_text_prompts)
        return [single] if single is not None else []

    def record_semantic_execution(self, *, physical_descent_started: bool, motion_evidence: dict) -> dict:
        detector = self.detector
        candidates = (detector, getattr(detector, 'fallback', None))
        for candidate in candidates:
            selector = getattr(candidate, 'visual_object_selector', None)
            marker = getattr(selector, 'mark_execution', None)
            if callable(marker):
                return marker(physical_descent_started=physical_descent_started, motion_evidence=motion_evidence)
        return {'schema': 'carryon-semantic-execution-event/v1', 'executed': False, 'error': 'semantic selector marker unavailable'}

    def regrasp(self, object_hint: str, pre_grasp_dis: float=0.1, source: str='topdown') -> MotionResult:
        grasp = self.perceive_grasp(object_hint, source=source)
        if grasp is None:
            return MotionResult(ok=False, reason=f"perception could not ground '{object_hint}'")
        res = self.open_gripper()
        if not res.ok:
            return MotionResult(ok=False, reason=f'open before regrasp failed: {res.reason}')
        res = self.goto_pose(grasp.position, grasp.quaternion, z_approach=pre_grasp_dis)
        if not res.ok:
            return MotionResult(ok=False, reason=f'approach failed: {res.reason}')
        return self.close_gripper()

    def geometric_check(self, predicate: Predicate) -> bool:
        st = self._read_state() or {}
        w = st.get('w_meas')
        if w is None:
            w = st.get('w_cmd')
        ctx = PredicateContext(q=[float(v) for v in st['q']] if 'q' in st else None, ee_pos=[float(v) for v in st['ee_pos']] if 'ee_pos' in st else None, ee_quat=[float(v) for v in st['ee_quat']] if 'ee_quat' in st else None, gripper_width=float(w) if w is not None else None)
        return bool(predicate(ctx))
