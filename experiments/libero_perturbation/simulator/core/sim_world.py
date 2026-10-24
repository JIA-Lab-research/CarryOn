"""Single-threaded LIBERO world + controller arbitration for the sim gello-shim.

MuJoCo is not thread-safe, so exactly ONE thread (the ``SimWorld`` loop) ever
touches the env. The HTTP and ZMQ servers are pure I/O front-ends that enqueue
commands and read shared snapshots (rendered frames, robot state) the loop
publishes under a lock.

Control arbitration (the ``ctrl_switch`` requirement):

- ``VLA`` mode    : the env runs the OSC_POSE controller and consumes the VLA's
                    7-dim ``[dpos(3), drot(3), gripper]`` delta actions.
- ``STANDBY`` mode: after a recovery ``cancel(standby=true)``, the loop stops
                    consuming VLA actions and just holds (no-op), so CarryOn can
                    drive the arm over the ZMQ bridge.
- ``RECOVERY`` mode: the loop swaps the arm controller to JOINT_POSITION and
                    follows the latest ZMQ joint target (``target_q - current_q``
                    deltas, the same streaming model gello uses), with the
                    gripper width binarised to the robosuite gripper action.

``handback`` (an ``/execute`` with ``skip_reset``) swaps back to OSC_POSE and
resumes VLA consumption from the *current* (recovery-restored) pose.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

GRIPPER_IDX = 6
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
# pre_grasp: how long after a pre_grasp handback to ignore a spurious
# release-standby so the resumed VLA can descend + pick from the standoff.
PREGRASP_RESUME_WINDOW_S = 8.0


def normalized_depth_stats(depth_norm: np.ndarray) -> dict:
    """Return lossless boundary diagnostics for normalized-depth producer output."""
    arr = np.asarray(depth_norm)
    finite = np.isfinite(arr)
    below = finite & (arr < 0.0)
    above = finite & (arr > 1.0)
    invalid = (~finite) | below | above
    finite_values = arr[finite]
    samples = []
    for coord in np.argwhere(invalid)[:64]:
        index = tuple(int(v) for v in coord)
        value = arr[index]
        samples.append({
            "index": list(index),
            "value": float(value) if np.isfinite(value) else str(value),
        })
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "total_count": int(arr.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "below_zero_count": int(below.sum()),
        "above_one_count": int(above.sum()),
        "invalid_count": int(invalid.sum()),
        "finite_min": float(finite_values.min()) if finite_values.size else None,
        "finite_max": float(finite_values.max()) if finite_values.size else None,
        "invalid_samples": samples,
    }


def require_normalized_depth(
    depth_norm: np.ndarray,
    *,
    dump_dir: Optional[str] = None,
    context: str = "snapshot",
) -> np.ndarray:
    """Preserve invalid producer output and fail closed; never clamp it."""
    arr = np.ascontiguousarray(np.asarray(depth_norm))
    stats = normalized_depth_stats(arr)
    if stats["invalid_count"] == 0:
        return arr
    stats["context"] = str(context)
    stats["captured_at_ns"] = time.time_ns()
    if dump_dir:
        root = Path(dump_dir)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"depth_violation_{stats['captured_at_ns']}"
        npz_tmp, npz_path = root / f"{stem}.npz.tmp", root / f"{stem}.npz"
        with npz_tmp.open("wb") as handle:
            np.savez_compressed(handle, depth_norm=arr)
        npz_tmp.replace(npz_path)
        stats["frame_path"] = str(npz_path)
        json_tmp, json_path = root / f"{stem}.json.tmp", root / f"{stem}.json"
        json_tmp.write_text(json.dumps(stats, indent=2, sort_keys=True))
        json_tmp.replace(json_path)
    raise ValueError(
        "normalized depth producer boundary violation: "
        + json.dumps(stats, sort_keys=True)
    )


def render_normalized_rgbd(
    sim,
    *,
    camera_name: str,
    width: int,
    height: int,
    dump_dir: Optional[str],
    context: str,
    max_attempts: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a verified RGB-D pair, retrying only rejected producer frames.

    No value is clamped or converted here. Persistent unit/non-finite failures
    remain fatal after every rejected frame has been preserved.
    """
    import logging

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        rendered = sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=True,
        )
        if not isinstance(rendered, tuple) or len(rendered) != 2:
            raise RuntimeError("high-resolution perception render returned no depth")
        rgb_raw, depth_raw = rendered
        rgb = np.ascontiguousarray(np.asarray(rgb_raw)[::-1])
        depth_norm = np.ascontiguousarray(np.asarray(depth_raw)[::-1])
        try:
            depth_norm = require_normalized_depth(
                depth_norm,
                dump_dir=dump_dir,
                context=f"{context}:attempt={attempt}",
            )
        except ValueError as exc:
            logging.error(
                "normalized-depth producer rejected render attempt %d/%d: %s",
                attempt, max_attempts, exc,
            )
            if attempt == max_attempts:
                raise
            continue
        return rgb, depth_norm
    raise AssertionError("unreachable normalized-depth render loop")


class Mode(str, Enum):
    IDLE = "idle"
    VLA = "vla"
    STANDBY = "standby"
    RECOVERY = "recovery"


@dataclass
class JointCommand:
    arm: list           # 7 target joint positions (rad)
    gripper: float      # commanded gripper width (m)
    ts: float = field(default_factory=time.time)


@dataclass
class CameraFrame:
    rgb: np.ndarray            # (H, W, 3) uint8
    depth_m: Optional[np.ndarray]  # (H, W) float32 meters, or None
    intrinsics: dict           # fx, fy, cx, cy, width, height, depth_scale, extrinsics
    ts: float
    producer_frame_seq: int
    simulation_step_id: Optional[int]
    simulation_time_s: Optional[float]
    camera_role: str
    camera_source: str


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Resize an (H, W, 3) uint8 image to (size, size). Area-average on
    downscale keeps small LIBERO objects sharp; returns a contiguous copy."""
    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img)
    import cv2

    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    out = cv2.resize(img, (size, size), interpolation=interp)
    return np.ascontiguousarray(out)


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class SimWorld:
    """Owns the LIBERO env + the control loop. Thread-safe command intake."""

    def __init__(
        self,
        *,
        task_suite_name: str = "libero_object",
        camera_name: str = "agentview",
        horizon_override: int = 0,
        resolution: int = 256,
        percep_resolution: Optional[int] = None,
        depth_scale: float = 0.001,
        seed: int = 7,
        num_steps_wait: int = 10,
        recovery_step_rad: float = 0.10,
        recovery_step_hz: float = 20.0,
        recovery_cmd_timeout_s: float = 0.25,
        close_width_m: float = 0.02,
    ) -> None:
        self.task_suite_name = task_suite_name
        self.camera_name = camera_name
        self.horizon_override = int(horizon_override or 0)
        self.resolution = resolution
        # Render policy observations natively at resolution. Render the judge and
        # recovery camera independently at percep_resolution.
        self.percep_resolution = int(percep_resolution) if percep_resolution else int(resolution)
        self.depth_scale = depth_scale
        self.seed = seed
        self.num_steps_wait = num_steps_wait
        self.recovery_step_rad = recovery_step_rad
        self.recovery_step_hz = recovery_step_hz
        self.recovery_cmd_timeout_s = recovery_cmd_timeout_s
        self.close_width_m = close_width_m

        self._lock = threading.RLock()
        self._segment_condition = threading.Condition(self._lock)
        self._vla_io_condition = threading.Condition(self._lock)
        self._vla_client_lock = threading.Lock()
        self._vla_io_thread = None
        self._vla_request = None
        self._vla_discarded_responses = []
        self._mode = Mode.IDLE
        self._env = None
        self._task_description = ""

        # shared snapshots
        self._frame: Optional[CameraFrame] = None
        self._wrist_rgb: Optional[np.ndarray] = None
        self._state: Optional[dict] = None  # bridge PUB state dict
        from qualification_history import RGBHistory
        self._qualification_rgb_history = RGBHistory(os.environ['QUALIFICATION_HISTORY_SESSION'])
        self._producer_frame_seq = 0
        self._qualification_rgb_history.clear()
        self._idle_refresh_steps = 0
        self._last_idle_refresh_s = 0.0

        # command intake
        self._latest_joint_cmd: Optional[JointCommand] = None
        self._last_applied_joint_cmd_ts: float = 0.0
        self._last_recovery_step_s: float = 0.0
        self._pending_execute: Optional[dict] = None
        self._cancel_requested = False
        # Arm-neutral timing telemetry and judge-clock anchor. Inference return
        # remains visible, but /execute is acknowledged only after the first
        # complete env.step + disturbance + snapshot cycle.
        self._first_action_returned = threading.Event()
        self._first_action_returned_ts: Optional[float] = None
        self._first_action_returned_generation: Optional[int] = None
        self._first_cycle_completed = threading.Event()
        self._first_cycle_completed_ts: Optional[float] = None
        self._first_cycle_completed_generation: Optional[int] = None
        self._next_segment_generation = 0
        self._active_segment_generation = 0
        self._episode_first_action_returned_ts: Optional[float] = None
        self._episode_first_cycle_completed_ts: Optional[float] = None
        # First action actually submitted to MuJoCo after the episode opens.
        # Model/service startup and inference return are separate clocks.
        self._first_robot_motion_ts: Optional[float] = None
        self._vla_warmup_latency_s: Optional[float] = None

        # controllers
        self._osc_controller = None
        self._joint_controller = None
        self._active_controller_type = "OSC_POSE"

        # VLA client + disturbance are injected by the shim
        self.vla_client = None
        self.disturbance = None
        self._disturbance_reset_done = False

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step = 0
        # Actual post-filter command intent: -1=open, +1=close.
        self._last_grip = -1.0
        # Recovery-mode environment steps do not advance the VLA task horizon.
        self._recovery_frozen_steps = 0

        # MuJoCo's EGL/GL render context is thread-affine: it must be created and
        # used on the SAME thread or rendering segfaults. The control loop renders
        # every tick, so the env (and its GL context) must be built ON the loop
        # thread, not in the caller. These coordinate that deferred build.
        self._ready = threading.Event()
        self._build_request: Optional[tuple] = None
        self._build_exc: Optional[BaseException] = None

        # Once the LIBERO episode terminates, robosuite refuses further
        # env.step() ("executing action in terminated episode"). Stop ticking the
        # env entirely after completion regardless of mode.
        self._episode_over = False
        # Request-id-bound, idempotent terminal handshakes. HTTP threads only
        # enqueue; the world thread performs controller/env work and publishes a
        # terminal state for the caller to acknowledge.
        self._complete_terminal_req: Optional[str] = None
        self._complete_terminal_id: Optional[str] = None
        self._complete_terminal_state = "idle"
        # Terminal-family #3: a non-standby final cancel (including recovery-cap
        # exhaustion after handback) must end the simulator episode as failure.
        # HTTP threads enqueue; the world thread publishes the result callback.
        self._failed_terminal_req: Optional[str] = None
        self._failed_terminal_id: Optional[str] = None
        self._failed_terminal_reason = "cancel"
        self._failed_terminal_state = "idle"
        self._pregrasp_resume_id: Optional[str] = None
        self._pregrasp_resume_state = "idle"
        # pre_grasp handback: when set, resume VLA on the loop thread; until the
        # deadline, ignore the spurious post-handback release-standby so the VLA
        # can pick. Only set by /pregrasp_resume (pre_grasp handback), so normal
        self._pregrasp_resume_req: Optional[str] = None
        self._pregrasp_resume_until = 0.0
        # Target displacement queued for the simulator thread.
        self._displace_request: Optional[tuple] = None
        self._displace_result: Optional[dict] = None
        self._displace_done = threading.Event()
        # Evaluator-only simulator GT timeline for post-hoc grounding accuracy.
        # Never served over HTTP and never consumed by recovery/selector/gate.
        self._offline_gt_timeline: list[dict] = []

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def build_env(self, task_id: int = 0, episode_idx: int = 0):
        """Construct the LIBERO env for one task/episode (called on the loop thread)."""
        import pathlib

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[self.task_suite_name]()
        task = suite.get_task(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_path = (
            pathlib.Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        init_states = suite.get_task_init_states(task_id)
        # Keep policy camera observations at their native resolution.
        # Render agentview RGB-D for judging and recovery separately.
        cam_names = [self.camera_name, "robot0_eye_in_hand"]
        cam_h = [self.resolution, self.resolution]
        cam_w = [self.resolution, self.resolution]
        _env_extra = {}
        if self.horizon_override > 0:
            # Reserve initial settling steps in addition to the requested VLA-action horizon.
            _env_extra["horizon"] = self.horizon_override + self.num_steps_wait
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_names=cam_names,
            camera_heights=cam_h,
            camera_widths=cam_w,
            camera_depths=True,
            **_env_extra,
        )
        env.seed(self.seed)
        env.reset()
        obs = env.set_init_state(init_states[episode_idx % len(init_states)])
        self._env = env
        self._task_description = task.language
        self._producer_frame_seq = 0
        self._qualification_rgb_history.clear()
        self._idle_refresh_steps = 0
        self._last_idle_refresh_s = 0.0
        # settle
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        self._build_controllers()
        self._update_snapshots(obs)
        return obs

    def _build_controllers(self) -> None:
        """Capture the OSC controller and build a JOINT_POSITION twin to swap in."""
        from robosuite.controllers import controller_factory, load_controller_config

        robot = self._env.robots[0]
        self._osc_controller = robot.controller
        # Shallow copy only: controller_config holds runtime references (notably
        # `sim`, a mujoco handle with ctypes pointers) that cannot be deep-copied
        # ("ctypes objects containing pointers cannot be pickled"). We only read
        # scalar fields out of it below and must carry `sim` over BY REFERENCE
        # for controller_factory, so a top-level dict copy is both correct and
        # what the JOINT_POSITION twin needs.
        base_cfg = dict(robot.controller_config)

        jp_cfg = load_controller_config(default_controller="JOINT_POSITION")
        # carry over the runtime-injected fields the factory needs
        for k in ("robot_name", "sim", "eef_name", "eef_rot_offset", "joint_indexes",
                  "actuator_range", "policy_freq", "ndim"):
            if k in base_cfg:
                jp_cfg[k] = base_cfg[k]
        jp_cfg["type"] = "JOINT_POSITION"
        jp_cfg["control_delta"] = True               # we feed (target - current) deltas
        jp_cfg["interpolation"] = None
        jp_cfg["output_max"] = self.recovery_step_rad
        jp_cfg["output_min"] = -self.recovery_step_rad
        jp_cfg["input_max"] = 1.0
        jp_cfg["input_min"] = -1.0
        self._joint_controller = controller_factory("JOINT_POSITION", jp_cfg)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sim-world", daemon=True)
        self._thread.start()

    def start_with_env(self, task_id: int, episode_idx: int, *, timeout: float = 180.0) -> None:
        """Start the loop thread, build the env ON that thread, and block until
        it is ready (so the GL context lives on the rendering thread). Raises any
        build error in the caller and on timeout."""
        self._build_request = (int(task_id), int(episode_idx))
        self.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(f"sim env build did not become ready within {timeout}s")
        if self._build_exc is not None:
            raise self._build_exc

    def warm_vla_once(self) -> float:
        """Run one inference without stepping the env, then report latency."""
        obs_state = self._current_obs_for_vla()
        if obs_state is None or self.vla_client is None:
            raise RuntimeError("VLA warmup requires a ready observation and client")
        img, wrist, state = obs_state
        example = {
            "image": [img, wrist],
            "lang": str(self._task_description),
        }
        if os.environ.get("VLA_BACKEND", "starvla") == "pi05":
            example["state"] = state
        t0 = time.monotonic()
        with self._vla_client_lock:
            response = self.vla_client.step(example=example, step=0)
        latency_s = time.monotonic() - t0
        if (
            not isinstance(response, dict)
            or ("raw_action" not in response and "action" not in response)
        ):
            raise RuntimeError("VLA warmup returned neither raw_action nor action")
        self._vla_warmup_latency_s = float(latency_s)
        return float(latency_s)

    def wait_first_action_returned(self, timeout_s: float) -> Optional[float]:
        """Wait for the active VLA segment's first inference response."""
        if not self._first_action_returned.wait(timeout=max(0.0, float(timeout_s))):
            return None
        return self._first_action_returned_ts

    def wait_first_cycle_completed(self, timeout_s: float) -> Optional[float]:
        """Wait until the active VLA segment applies and publishes one action."""
        if not self._first_cycle_completed.wait(
            timeout=max(0.0, float(timeout_s))
        ):
            return None
        return self._first_cycle_completed_ts

    def wait_first_cycle_completed_for_generation(
        self, expected_generation: int, timeout_s: float
    ) -> Optional[float]:
        """Wait for a newly submitted segment, never an earlier set Event."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._segment_condition:
            while self._first_cycle_completed_generation != expected_generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._segment_condition.wait(timeout=remaining)
            return self._first_cycle_completed_ts

    def first_action_for_generation(
        self, expected_generation: int
    ) -> Optional[float]:
        with self._lock:
            if self._first_action_returned_generation != expected_generation:
                return None
            return self._first_action_returned_ts

    @property
    def first_cycle_completed_ts(self) -> Optional[float]:
        return self._first_cycle_completed_ts

    @property
    def first_action_returned_ts(self) -> Optional[float]:
        return self._first_action_returned_ts

    @property
    def episode_first_action_returned_ts(self) -> Optional[float]:
        return self._episode_first_action_returned_ts

    @property
    def episode_first_cycle_completed_ts(self) -> Optional[float]:
        return self._episode_first_cycle_completed_ts

    @property
    def first_robot_motion_ts(self) -> Optional[float]:
        return self._first_robot_motion_ts

    @property
    def vla_warmup_latency_s(self) -> Optional[float]:
        return self._vla_warmup_latency_s

    def _mark_first_robot_motion(self) -> float:
        """Latch the first real control step, excluding build/warm-up steps."""
        now = time.time()
        with self._lock:
            if self._first_robot_motion_ts is None:
                self._first_robot_motion_ts = now
            return self._first_robot_motion_ts

    def stop(self) -> None:
        self._stop.set()
        with self._vla_io_condition:
            self._vla_io_condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    # command intake (thread-safe; called by HTTP/ZMQ handlers)          #
    # ------------------------------------------------------------------ #
    def submit_execute(self, instruction: str, inputs: dict, request_id: str,
                       callback_url: str) -> int:
        with self._lock:
            self._next_segment_generation += 1
            segment_generation = self._next_segment_generation
            self._pending_execute = {
                "segment_generation": segment_generation,
                "instruction": instruction,
                "inputs": dict(inputs or {}),
                "request_id": request_id,
                "callback_url": callback_url,
            }
            return segment_generation

    def _apply_pending_execute(self) -> bool:
        """Apply an HTTP execute on the EGL-affine simulator loop thread."""
        with self._lock:
            pending = self._pending_execute
            if pending is None:
                return False
            self._pending_execute = None
        try:
            self.begin_vla_episode(
                skip_reset=bool((pending.get("inputs") or {}).get("skip_reset")),
                segment_generation=int(pending["segment_generation"]),
            )
        except Exception as e:  # noqa: BLE001
            import logging
            logging.info("execute handoff: begin_vla_episode raised: %s", e)
        return True

    def request_cancel(self, standby: bool) -> None:
        with self._lock:
            # pre_grasp resume window: after a pre_grasp handback, the
            # recovery's positioning open is misread by the checkpoint as a task
            # release -> a spurious standby cancel that would freeze the VLA out.
            # Within the bounded window (set ONLY by /pregrasp_resume), ignore a
            # standby cancel so the VLA keeps control and picks. Non-standby
            # cancels and all cancels outside the window are unaffected -> normal
            if standby and time.time() < self._pregrasp_resume_until:
                self._cancel_requested = True
                return
            self._cancel_requested = True
            if standby:
                if self._vla_request is not None:
                    self._vla_request["invalidated"] = True
                self._mode = Mode.STANDBY
                self._latest_joint_cmd = None
                self._last_applied_joint_cmd_ts = 0.0
                self._last_recovery_step_s = 0.0

    def request_displace(self, name: str, dx: float, dy: float, *, timeout: float = 20.0) -> dict:
        """Queue a target displacement for execution on the simulator thread."""
        with self._lock:
            self._displace_done.clear()
            self._displace_result = None
            self._displace_request = (str(name), float(dx), float(dy))
        if not self._displace_done.wait(timeout=timeout):
            return {"error": f"displace did not complete within {timeout}s"}
        return dict(self._displace_result or {"error": "no displace result"})

    def push_joint_command(self, arm, gripper: float) -> None:
        with self._lock:
            self._latest_joint_cmd = JointCommand(arm=list(arm), gripper=float(gripper))
            # The loop consumes this command and swaps the controller itself.
            # A ZMQ thread must not change env.action_dim under a pending VLA step.

    # ------------------------------------------------------------------ #
    # shared snapshot readers (thread-safe; for HTTP camera + ZMQ PUB)   #
    # ------------------------------------------------------------------ #
    def get_frame(self) -> Optional[CameraFrame]:
        with self._lock:
            return self._frame

    def get_bridge_state(self) -> Optional[dict]:
        with self._lock:
            if not self._state:
                return None
            state = dict(self._state)
            state['physics_width_history'] = [dict(x) for x in getattr(self, '_physics_width_history', [])]
            return state

    def get_mode(self) -> Mode:
        with self._lock:
            return self._mode

    # ------------------------------------------------------------------ #
    # controller switching                                               #
    # ------------------------------------------------------------------ #
    def _swap_controller(self, ctype: str) -> None:
        robot = self._env.robots[0]
        if ctype == "JOINT_POSITION":
            robot.controller = self._joint_controller
        else:
            robot.controller = self._osc_controller
        robot.controller.reset_goal()
        # robosuite caches RobotEnv._action_dim at reset. Runtime controller
        # swaps change robot.action_dim (OSC_POSE: 6 + gripper = 7;
        # JOINT_POSITION: 7 + gripper = 8), so keep the cached env dimension in
        # sync or env.step() and robot.control() disagree about action length.
        base_env = self._robo_env()
        if base_env is not None and hasattr(base_env, "_action_dim"):
            base_env._action_dim = sum(r.action_dim for r in base_env.robots)
        self._active_controller_type = ctype

    def _robo_env(self):
        """Unwrap the LIBERO wrapper to the robosuite MujocoEnv (has .timestep /
        .horizon). Returns None if unavailable."""
        return getattr(self._env, "env", self._env)

    # ------------------------------------------------------------------ #
    # the one true loop                                                  #
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        # Build the env on this (the rendering) thread when requested, then
        # publish readiness so the shim can start serving snapshots.
        if self._build_request is not None and not self._ready.is_set():
            try:
                tid, eid = self._build_request
                self.build_env(task_id=tid, episode_idx=eid)
            except BaseException as e:  # noqa: BLE001 - surface to start_with_env
                self._build_exc = e
                self._ready.set()
                return
            self._ready.set()
        while not self._stop.is_set():
            if self._episode_over:
                # episode terminated; never step a terminated robosuite env again
                time.sleep(0.05)
                continue
            if self._apply_pending_execute():
                continue
            if self._process_complete_terminal_request():
                continue
            if self._process_failed_terminal_request():
                continue
            if self._pregrasp_resume_req is not None:
                # pre_grasp handback (loop thread only -- controller swap is
                # EGL-thread-affine): resume VLA control from the pre-grasp standoff
                # so the in-distribution VLA descends + picks. The resume window
                # (set in request_pregrasp_resume) makes request_cancel ignore the
                # spurious release-standby until it elapses.
                with self._lock:
                    request_id = self._pregrasp_resume_req
                    self._pregrasp_resume_req = None
                    self._pregrasp_resume_state = "resuming"
                try:
                    self.begin_vla_episode(skip_reset=True)
                except Exception as e:  # noqa: BLE001
                    import logging
                    logging.info("pre_grasp resume: begin_vla_episode raised: %s", e)
                    state = "failed"
                else:
                    state = "waiting_first_action"
                with self._lock:
                    if self._pregrasp_resume_id == request_id:
                        self._pregrasp_resume_state = state
                continue
            if self._displace_request is not None:
                # Apply queued object displacement on the simulator thread.
                with self._lock:
                    name, dx, dy = self._displace_request
                    self._displace_request = None
                result: dict = {}
                try:
                    import disturbances as _D
                    env = self._env
                    if env is None or self._episode_over:
                        result = {"error": "env not available or episode over"}
                    else:
                        pose = _D.get_object_pose(env, name)
                        if pose is None:
                            result = {"error": f"no pose for {name!r}"}
                        elif not _D.set_object_pose(
                            env, name,
                            [float(pose[0]) + dx, float(pose[1]) + dy, *[float(v) for v in pose[2:]]],
                        ):
                            result = {"error": "set_object_pose failed"}
                        else:
                            # Re-render on THIS (loop) thread via a zero-action hold
                            # step, exactly like _tick_hold, so the teleport shows in
                            # the served frame + _raw_obs (GT reads _raw_obs).
                            if self._active_controller_type == "OSC_POSE":
                                action = [0.0] * 6 + [self._last_gripper_action()]
                            else:
                                cur_q = self._arm_qpos()
                                action = [0.0] * len(cur_q) + [self._last_gripper_action()]
                            obs, _r, _d, _i = env.step(action)
                            self._update_snapshots(obs)
                            xy = self.object_world_xy(name)
                            px = self.object_target_px(name)
                            in_frame = bool(
                                px is not None and 0.03 < px[0] < 0.97 and 0.03 < px[1] < 0.97
                            )
                            result = {
                                "name": name,
                                "xy": [xy[0], xy[1]] if xy else None,
                                "px": [px[0], px[1]] if px else None,
                                "in_frame": in_frame,
                            }
                except Exception as e:  # noqa: BLE001
                    result = {"error": f"{type(e).__name__}: {e}"}
                self._displace_result = result
                self._displace_done.set()
                continue
            try:
                self._apply_joint_takeover()
                mode = self.get_mode()
                if mode == Mode.VLA:
                    self._tick_vla()
                elif mode == Mode.RECOVERY:
                    self._tick_recovery()
                    # complete-mode scoring gate: a placement
                    # COMPLETED by recovery (handback_mode=complete) is otherwise
                    # NEVER scored -- Option-B freezes the task horizon so the env
                    # never reaches a terminal in recovery mode, and _on_episode_done
                    # is only reached from VLA mode. Poll the env success predicate
                    # here so a recovery-completed task is SCORED success (not the
                    # silent null we keep getting bitten by). Only ever fires True
                    # after a REAL placement: full_hold lifts+rewinds and pre_grasp
                    # opens at standoff -- neither puts the object in the destination.
                    if not self._episode_over and self._env_success():
                        self._on_episode_done(success=True, terminal_reason="normal_success")
                else:  # IDLE / STANDBY -> freeze (see _tick_idle_freeze)
                    self._tick_idle_freeze()
            except Exception as e:  # noqa: BLE001 -- never let the loop die silently
                # robosuite raises this once the episode has terminated (e.g. the
                # horizon was reached) even if the terminating step did not report
                # done=True. Finalize instead of spinning and re-stepping forever.
                if "terminated episode" in str(e):
                    if not self._episode_over:
                        self._finish_from_env()
                    continue
                import logging
                logging.exception("sim-world tick failed: %s", e)
                time.sleep(0.05)

        # The EGL context was created on this loop thread, so close it here.
        # Letting Python destruct the env on the main thread produced
        # EGL_NOT_INITIALIZED/libGLU traceback noise after otherwise clean runs.
        env = self._env
        if env is not None:
            try:
                env.close()
            except Exception as e:  # noqa: BLE001 - teardown remains visible
                import logging
                logging.exception("sim-world EGL-affine env.close failed: %s", e)
            finally:
                self._env = None

    def _finish_from_env(self) -> None:
        """Finalize a native env terminal with explicit timeout semantics."""
        success = self._env_success()
        self._on_episode_done(
            success=success,
            terminal_reason="normal_success" if success else "env_timeout",
        )

    def _env_success(self) -> bool:
        """Best-effort task-success read from the underlying robosuite env."""
        env = self._env
        if env is None:
            return False
        for getter in ("check_success", "_check_success"):
            fn = getattr(env, getter, None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:  # noqa: BLE001
                    pass
        return False

    def request_complete_terminal(self, request_id: str) -> dict:
        """Idempotently enqueue complete-mode settle/finalization.

        A different request id is rejected once an episode terminal is in flight;
        replaying the same id returns its current state without executing twice.
        """
        rid = str(request_id or "").strip()
        if not rid:
            return {"accepted": False, "state": "rejected", "reason": "missing request_id"}
        with self._lock:
            if self._complete_terminal_id not in (None, rid):
                return {"accepted": False, "state": "conflict",
                        "reason": "another complete request is already bound"}
            if self._complete_terminal_id is None:
                self._complete_terminal_id = rid
                self._complete_terminal_state = "pending"
                self._complete_terminal_req = rid
            return {"accepted": True, "request_id": rid,
                    "state": self._complete_terminal_state}

    def complete_terminal_status(self, request_id: str) -> dict:
        rid = str(request_id or "").strip()
        with self._lock:
            if not rid or rid != self._complete_terminal_id:
                return {"ok": False, "request_id": rid, "state": "unknown"}
            return {"ok": True, "request_id": rid,
                    "state": self._complete_terminal_state}

    def _process_complete_terminal_request(self) -> bool:
        """Process a queued terminal request on the simulator thread."""
        with self._lock:
            request_id = self._complete_terminal_req
            if request_id is None:
                return False
            self._complete_terminal_req = None
            self._complete_terminal_state = "settling"
        try:
            self._settle_and_score()
        finally:
            with self._lock:
                if self._complete_terminal_id == request_id:
                    # Acknowledges settle + benchmark finalization, not policy
                    # success; that remains visible only in the result callback.
                    self._complete_terminal_state = "completed"
        return True

    def request_failed_terminal(self, request_id: str, terminal_reason: str = "cancel") -> dict:
        """Idempotently enqueue a failed episode terminal on the world thread.

        This is distinct from standby cancellation, which freezes the VLA for
        recovery and must never finalize the benchmark episode.
        """
        rid = str(request_id or "").strip()
        if not rid:
            return {"accepted": False, "state": "rejected",
                    "reason": "missing request_id"}
        allowed = {"normal_failure", "cancel", "cap_refusal", "handback_failure", "exception"}
        if terminal_reason not in allowed:
            return {
                "accepted": False,
                "state": "rejected",
                "reason": f"invalid terminal_reason: {terminal_reason}",
            }
        allowed = {
            "normal_failure", "cancel", "cap_refusal",
            "handback_failure", "exception",
        }
        if terminal_reason not in allowed:
            return {
                "accepted": False,
                "state": "rejected",
                "reason": f"invalid terminal_reason: {terminal_reason}",
            }
        with self._lock:
            if self._failed_terminal_id not in (None, rid):
                return {"accepted": False, "state": "conflict",
                        "reason": "another failed terminal is already bound"}
            if self._failed_terminal_id is None:
                self._failed_terminal_id = rid
                self._failed_terminal_state = "pending"
                self._failed_terminal_req = rid
                self._failed_terminal_reason = terminal_reason
            return {"accepted": True, "request_id": rid,
                    "state": self._failed_terminal_state}

    def failed_terminal_status(self, request_id: str) -> dict:
        rid = str(request_id or "").strip()
        with self._lock:
            if not rid or rid != self._failed_terminal_id:
                return {"ok": False, "request_id": rid, "state": "unknown"}
            return {"ok": True, "request_id": rid,
                    "state": self._failed_terminal_state}

    def _process_failed_terminal_request(self) -> bool:
        """Finalize one queued non-standby cancel as benchmark failure."""
        with self._lock:
            request_id = self._failed_terminal_req
            if request_id is None:
                return False
            self._failed_terminal_req = None
            self._failed_terminal_state = "finalizing"
            terminal_reason = self._failed_terminal_reason
        try:
            self._on_episode_done(success=False, terminal_reason=terminal_reason)
        finally:
            with self._lock:
                if self._failed_terminal_id == request_id:
                    self._failed_terminal_state = "completed"
        return True

    def request_pregrasp_resume(self, request_id: str) -> dict:
        """Idempotently bind and enqueue the pre-grasp VLA resume handshake."""
        rid = str(request_id or "").strip()
        if not rid:
            return {"accepted": False, "state": "rejected", "reason": "missing request_id"}
        with self._lock:
            if self._pregrasp_resume_id not in (None, rid):
                return {"accepted": False, "state": "conflict",
                        "reason": "another pregrasp request is already bound"}
            if self._pregrasp_resume_id is None:
                self._pregrasp_resume_id = rid
                self._pregrasp_resume_state = "pending"
                self._pregrasp_resume_until = time.time() + PREGRASP_RESUME_WINDOW_S
                self._pregrasp_resume_req = rid
            return {"accepted": True, "request_id": rid,
                    "state": self._pregrasp_resume_state}

    def pregrasp_resume_status(self, request_id: str) -> dict:
        rid = str(request_id or "").strip()
        with self._lock:
            if not rid or rid != self._pregrasp_resume_id:
                return {"ok": False, "request_id": rid, "state": "unknown"}
            return {"ok": True, "request_id": rid,
                    "state": self._pregrasp_resume_state}

    def _gripper_grasping_any(self):
        """FAIR proprioceptive grasp-detect: are the gripper fingerpads actually
        grasping ANY movable scene object (not the basket / floor / robot)?

        Object-agnostic -- it does NOT use the task-target identity or object
        pose, so it is a tactile 'am I holding something' signal (the modality a
        real gripper has), not privileged localisation. Needed because jaw WIDTH
        alone cannot separate a thin object HELD (width ~ empty) from an empty
        near-miss (width slightly > empty) -- the thin-box assert_holding false
        positive that corrupts chain_ok. Returns True/False, or None if the env
        cannot answer (caller then falls back to the width check alone)."""
        for cand in (self._env, getattr(self._env, "env", None)):
            if cand is None:
                continue
            try:
                gripper = cand.robots[0].gripper
                od = getattr(cand, "objects_dict", None) or {}
                check = getattr(cand, "_check_grasp", None)
                if not od or not callable(check):
                    continue
                for name, obj in od.items():
                    if "basket" in str(name).lower():
                        continue
                    geoms = getattr(obj, "contact_geoms", None)
                    if not geoms:
                        continue
                    if check(gripper=gripper, object_geoms=geoms):
                        return True
                return False
            except Exception:  # noqa: BLE001 - try the other unwrap, else None
                continue
        return None

    def _runtime_action_dim(self) -> int:
        """Resolve the active controller's action dimension through wrappers."""
        outer = self._env
        if outer is None:
            return 0
        cur = outer
        for _ in range(5):
            for attr in ("action_dim", "_action_dim"):
                action_dim = getattr(cur, attr, None)
                if action_dim:
                    return int(action_dim)
            inner = getattr(cur, "env", None)
            if inner is None or inner is cur:
                break
            cur = inner
        action_space = getattr(outer, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if shape:
            return int(shape[0])
        if isinstance(action_space, (tuple, list)) and action_space:
            return int(np.asarray(action_space[0]).reshape(-1).size)
        return 0

    def _settle_and_score(self) -> None:
        """Settle the simulator and compute terminal episode scores."""
        import logging
        env = self._env
        if env is None:
            self._on_episode_done(success=False, terminal_reason="normal_failure")
            return
        success = False
        settle_frames = 60  # ~3s at control_freq=20: ample for a released object to land
        grip = self._last_gripper_action()
        action_dim = self._runtime_action_dim()
        if action_dim < 1:
            logging.info("complete-mode settle: invalid env action_dim=%r", action_dim)
            self._on_episode_done(success=False, terminal_reason="normal_failure")
            return
        logging.info("complete-mode settle: runtime action_dim=%d", action_dim)
        last_obs = None
        for i in range(settle_frames):
            if self._episode_over:
                return
            deadline = getattr(self, "_episode_deadline_wall", None)
            if deadline is None:
                self._on_episode_done(success=False, terminal_reason="exception")
                return
            if time.time() >= deadline:
                self._on_episode_done(success=False, terminal_reason="wall_cap")
                return
            native = self._robo_env()
            elapsed = getattr(native, "timestep", None)
            horizon = getattr(native, "horizon", None)
            if not isinstance(elapsed, (int,float)) or not isinstance(horizon, (int,float)) or horizon <= 0:
                self._on_episode_done(success=False, terminal_reason="exception")
                return
            if elapsed >= horizon:
                break
            try:
                obs, _r, _done, _info = env.step([0.0] * (action_dim - 1) + [grip])
                last_obs = obs
                self._update_snapshots(obs)
            except Exception as e:  # noqa: BLE001
                logging.info("complete-mode settle: env.step stopped at frame %d (%s)", i, e)
                break
            if time.time() >= self._episode_deadline_wall:
                self._on_episode_done(success=False, terminal_reason="wall_cap")
                return
            if self._env_success():
                success = True
                break
            if _done:
                break
        try:
            ee = last_obs.get("robot0_eef_pos") if isinstance(last_obs, dict) else None
            if ee is not None:
                logging.info("complete-mode settle: EE final xyz=%s",
                             [round(float(v), 3) for v in ee[:3]])
        except Exception:  # noqa: BLE001
            pass
        # Diagnostic: where did the object end vs the perceived basket target?
        try:
            import disturbances as _D
            objn = getattr(self.disturbance, "_object_name", None)
            if objn is not None:
                op = _D.get_object_pose(self._env, objn)
                if op is not None:
                    logging.info("complete-mode settle: object final xyz=%s",
                                 [round(float(v), 3) for v in op[:3]])
        except Exception:  # noqa: BLE001
            pass
        logging.info("complete-mode settle+score: success=%s", success)
        self._on_episode_done(
            success=success,
            terminal_reason="normal_success" if success else "normal_failure",
        )

    # -- VLA mode --------------------------------------------------------- #
    def begin_vla_episode(
        self, skip_reset: bool, *, segment_generation: Optional[int] = None
    ) -> None:
        """Switch into VLA consumption. ``skip_reset`` resumes from current pose."""
        with self._lock:
            if segment_generation is None:
                self._next_segment_generation += 1
                segment_generation = self._next_segment_generation
            self._active_segment_generation = int(segment_generation)
            self._first_action_returned.clear()
            self._first_action_returned_ts = None
            self._first_action_returned_generation = None
            self._first_cycle_completed.clear()
            self._first_cycle_completed_ts = None
            self._first_cycle_completed_generation = None
            if self._active_controller_type != "OSC_POSE":
                self._swap_controller("OSC_POSE")
            self._mode = Mode.VLA
            self._cancel_requested = False
            # Recovery handback: stop injecting disturbances so the resumed VLA can
            # actually move the arm (noop_freeze with a huge duration would otherwise
            # keep zeroing actions forever after the one recovery trigger). Crucially,
            # do NOT re-arm via disturbance.reset() on resume -- reset() clears the
            # _disabled flag, which would re-enable the freeze on the next tick and
            # leave the VLA frozen again. Marking the reset as already done keeps the
            # disturbance disabled for the rest of the resumed episode.
            if skip_reset and self.disturbance is not None:
                self.disturbance.disable()
                self._disturbance_reset_done = True
            else:
                self._disturbance_reset_done = False

    def _vla_io_loop(self) -> None:
        """One serial client worker. Never reads or changes the environment."""
        while True:
            with self._vla_io_condition:
                self._vla_io_condition.wait_for(
                    lambda: self._stop.is_set() or (
                        self._vla_request is not None
                        and not self._vla_request["started"]
                    )
                )
                if self._stop.is_set():
                    return
                job = self._vla_request
                job["started"] = True
            started = time.monotonic()
            response, error = None, None
            try:
                with self._vla_client_lock:
                    response = job["client"].step(
                        example=job["example"], step=job["step"]
                    )
            except Exception as exc:  # delivered only if this segment still owns control
                error = exc
            with self._lock:
                job.update(response=response, error=error, done=True,
                           returned_ts=time.time(),
                           inference_s=time.monotonic() - started)

    def _vla_job_current(self, job: dict) -> bool:
        return (
            not self._stop.is_set() and not self._episode_over
            and not job["invalidated"]
            and job["generation"] == self._active_segment_generation
            and self._mode == Mode.VLA
            and self._active_controller_type == "OSC_POSE"
            and self._pending_execute is None
            and self._pregrasp_resume_req is None
            and self._complete_terminal_req is None
            and self._failed_terminal_req is None
        )

    def _apply_joint_takeover(self) -> None:
        """Only the world loop commits the controller swap queued by ZMQ."""
        with self._lock:
            if self._mode == Mode.STANDBY and self._latest_joint_cmd is not None:
                self._mode = Mode.RECOVERY
                self._swap_controller("JOINT_POSITION")

    def _tick_vla(self) -> None:
        # Atomic commit vs command intake; nested snapshot/ACK helpers use RLock.
        with self._lock:
            job = self._vla_request
            if job is not None:
                if not job["done"]:
                    # Do not issue a hold step or refresh the environment while
                    # VLA inference is pending. The loop can still service takeover.
                    self._stop.wait(0.001)
                    return
                self._vla_request = None
                if not self._vla_job_current(job):
                    self._vla_discarded_responses.append({
                        "generation": job["generation"], "step": job["step"],
                        "active_generation": self._active_segment_generation,
                        "invalidated": job["invalidated"], "mode": self._mode.value,
                    })
                    return
                if job["error"] is not None:
                    raise job["error"]
                self._commit_vla_response(job)
                return
            if (self._mode != Mode.VLA or self._episode_over or self._stop.is_set()
                    or self._pending_execute is not None
                    or self._complete_terminal_req is not None
                    or self._failed_terminal_req is not None
                    or self._pregrasp_resume_req is not None):
                return
            env = self._env
            cycle_t0 = time.monotonic()
            obs_state = self._current_obs_for_vla()
            if obs_state is None:
                self._tick_hold()
                return
            img, wrist, state = obs_state
            obs_s = time.monotonic() - cycle_t0
            disturbance_t0 = time.monotonic()
            if self.disturbance is not None and not self._disturbance_reset_done:
                self.disturbance.reset(env, self._raw_obs)
                self._disturbance_reset_done = True
            disturbance_s = time.monotonic() - disturbance_t0
            example = {"image": [img, wrist], "lang": str(self._task_description)}
            if os.environ.get("VLA_BACKEND", "starvla") == "pi05":
                example["state"] = state
            self._vla_request = {
                "client": self.vla_client, "example": copy.deepcopy(example),
                "step": self._step, "generation": self._active_segment_generation,
                "cycle_t0": cycle_t0, "obs_s": obs_s, "disturbance_s": disturbance_s,
                "started": False, "done": False, "invalidated": False,
            }
            if self._vla_io_thread is None:
                self._vla_io_thread = threading.Thread(
                    target=self._vla_io_loop, name="sim-vla-io", daemon=True)
                self._vla_io_thread.start()
            self._vla_io_condition.notify()

    def _commit_vla_response(self, job: dict) -> None:
        # Caller holds the world lock and has checked segment ownership.
        env = self._env
        cycle_t0, obs_s, disturbance_s = (job[k] for k in
                                          ("cycle_t0", "obs_s", "disturbance_s"))
        resp, inference_s = job["response"], job["inference_s"]
        if not self._first_action_returned.is_set():
            first_ts = job["returned_ts"]
            with self._lock:
                self._first_action_returned_ts = first_ts
                self._first_action_returned_generation = self._active_segment_generation
                if self._episode_first_action_returned_ts is None:
                    self._episode_first_action_returned_ts = first_ts
                self._first_action_returned.set()
                self._segment_condition.notify_all()
            import logging
            logging.info(
                "first VLA action returned: step=%d ts=%.6f", self._step, first_ts
            )
        native_action = resp.get("action")
        if native_action is not None:
            action = np.asarray(native_action, dtype=np.float32).reshape(-1)
            if action.size != 7:
                raise RuntimeError(
                    f"backend action is {action.size}-D; LIBERO expects 7-D"
                )
        else:
            ra = resp["raw_action"]
            wv = np.asarray(ra.get("world_vector"), dtype=np.float32).reshape(-1)
            rot = np.asarray(ra.get("rotation_delta"), dtype=np.float32).reshape(-1)
            og = np.asarray(ra.get("open_gripper"), dtype=np.float32).reshape(-1)
            grip = np.asarray(
                [1.0 - 2.0 * (float(og[0]) > 0.5)], dtype=np.float32
            )
            action = np.concatenate([wv, rot, grip], axis=0)
        if self.disturbance is not None:
            action = self.disturbance.filter_action(action, self._step)
        self._last_grip = float(action[-1])

        self._mark_first_robot_motion()
        env_step_t0 = time.monotonic()
        obs, reward, done, info = env.step(action.tolist())
        env_step_s = time.monotonic() - env_step_t0
        on_obs_t0 = time.monotonic()
        if self.disturbance is not None:
            self.disturbance.on_obs(env, obs, self._step)
        on_obs_s = time.monotonic() - on_obs_t0
        snapshots_t0 = time.monotonic()
        self._update_snapshots(obs, state_source="robot_observation_after_physics_step")
        snapshots_s = time.monotonic() - snapshots_t0
        self._step += 1
        if not self._first_cycle_completed.is_set():
            completed_ts = time.time()
            with self._lock:
                self._first_cycle_completed_ts = completed_ts
                self._first_cycle_completed_generation = self._active_segment_generation
                if self._episode_first_cycle_completed_ts is None:
                    self._episode_first_cycle_completed_ts = completed_ts
                if self._pregrasp_resume_state == "waiting_first_action":
                    self._pregrasp_resume_state = "resumed"
                self._first_cycle_completed.set()
                self._segment_condition.notify_all()
            import logging
            logging.info(
                "first VLA cycle completed: step=%d ts=%.6f",
                self._step, completed_ts,
            )


        cycle_s = time.monotonic() - cycle_t0
        if self._step < 3 or max(
            inference_s, env_step_s, on_obs_s, snapshots_s
        ) >= 1.0:
            import logging
            logging.info(
                "[vla-cycle] step=%d obs=%.4fs disturbance=%.4fs "
                "inference=%.4fs env_step=%.4fs on_obs=%.4fs "
                "snapshots=%.4fs total=%.4fs",
                self._step, obs_s, disturbance_s, inference_s, env_step_s,
                on_obs_s, snapshots_s, cycle_s,
            )
        if done:
            # `done` is True for BOTH task success and horizon-timeout (failure);
            # query the env's success check rather than assuming success.
            self._finish_from_env()
        elif self._cancel_requested:
            # a cancel landed; the mode was already flipped by request_cancel
            self._cancel_requested = False

    # -- recovery mode (consume ZMQ joint stream) ------------------------ #
    def _tick_recovery(self) -> None:
        env = self._env
        with self._lock:
            cmd = self._latest_joint_cmd
            recovery_active = (
                self._mode == Mode.RECOVERY
                and self._active_controller_type == "JOINT_POSITION"
            )
        if (
            not recovery_active
            or cmd is None
            or (time.time() - float(cmd.ts)) > self.recovery_cmd_timeout_s
        ):
            # Recovery can spend seconds in perception/planning between joint
            # stream bursts. Do not burn LIBERO horizon (or move the scene) while
            # waiting for the next recovery command. The command is TTL-scoped:
            # do not replay a stale last target forever while SAM3/CGN/cuRobo run.
            self._tick_idle_freeze()
            return
        now = time.time()
        period = 1.0 / max(1e-3, self.recovery_step_hz)
        if now - self._last_recovery_step_s < period:
            # The loop can spin much faster than the bridge stream. Rate-limit
            # recovery stepping so repeated fresh targets help the joint
            # controller settle without burning the LIBERO horizon as fast as CPU.
            self._tick_idle_freeze()
            return
        cur_q = self._arm_qpos()
        target = np.asarray(cmd.arm, dtype=np.float64)
        delta = target - cur_q
        # JOINT_POSITION controller maps input[-1,1] -> output_max rad; normalise.
        scaled = np.clip(delta / max(self.recovery_step_rad, 1e-6), -1.0, 1.0)
        grip_action = 1.0 if cmd.gripper <= self.close_width_m else -1.0
        self._last_grip = float(grip_action)
        action = np.concatenate([scaled, [grip_action]]).tolist()
        self._mark_first_robot_motion()
        # Option B -- freeze the task horizon clock during recovery: snapshot the
        # robosuite timestep before the step and restore it after, so recovery arm
        # motion advances the physics/scene but does NOT burn the VLA's episode
        # horizon. Recovery is an examiner intervention that restores an
        # in-distribution state; the VLA should resume with its remaining task
        # budget intact. Without this, slow recovery (goto_init ~200 steps) drains
        # the ~520-step LIBERO horizon and the episode terminates mid-recovery
        # (mode=RECOVERY) before handback. The physics still steps; only the
        # horizon COUNTER is held.
        robo = self._robo_env()
        ts_before = getattr(robo, "timestep", None) if robo is not None else None
        obs, _reward, done, _info = env.step(action)
        if ts_before is not None:
            try:
                robo.timestep = ts_before
                # env.step's _post_action already latched robo.done from the
                # (now rolled-back) timestep; clear it too, else the NEXT step
                # raises "executing action in terminated episode".
                if getattr(robo, "done", False):
                    robo.done = False
                self._recovery_frozen_steps += 1
                done = False  # horizon-driven done is void while the clock is frozen
            except Exception:  # noqa: BLE001 -- never break recovery on a rollback miss
                pass
        self._last_applied_joint_cmd_ts = float(cmd.ts)
        self._last_recovery_step_s = now
        self._update_snapshots(obs, state_source="robot_observation_after_physics_step")
        if done:
            self._finish_from_env()

    def _refresh_recovery_robot_state(self) -> bool:
        """Re-read robot proprioception on the world thread; never step physics.

        A new wall timestamp denotes a new read, not simulation advancement.
        Failure leaves the last sample and its original timestamp unchanged.
        Camera snapshots are deliberately not refreshed or re-timestamped here.
        """
        try:
            robot = self._env.robots[0]
            data = robot.sim.data
            q = np.array(data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64, copy=True)
            dq = np.array(data.qvel[robot._ref_joint_vel_indexes], dtype=np.float64, copy=True)
            gq = np.array(data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=np.float64, copy=True)
            ee_pos = np.array(data.site_xpos[robot.eef_site_id], dtype=np.float64, copy=True)
            quat_wxyz = np.array(data.get_body_xquat(robot.robot_model.eef_name), dtype=np.float64, copy=True)
            if (q.ndim != 1 or q.size == 0 or dq.shape != q.shape
                    or gq.shape != (2,) or ee_pos.shape != (3,) or quat_wxyz.shape != (4,)
                    or not all(np.all(np.isfinite(x)) for x in (q, dq, gq, ee_pos, quat_wxyz))):
                return False
            read_ts = time.time()
            release_commanded = bool(self._last_gripper_action() < 0.0)
            state = {
                "type": "state", "ts": read_ts,
                "q": q.tolist(), "dq": dq.tolist(),
                "ee_pos": ee_pos.tolist(), "ee_quat": quat_wxyz[[1, 2, 3, 0]].tolist(),
                "ee_vel": [0.0] * 6,
                "w_cmd": 0.08 if release_commanded else self.close_width_m,
                "w_meas": float(abs(gq[0] - gq[1])),
                "release_commanded": release_commanded,
                "state_source": "robot_sim_data_read_no_physics_step",
            }
        except Exception:
            return False
        with self._lock:
            self._state = state
        return True

    # -- idle / standby freeze ------------------------------------------- #
    def _tick_idle_freeze(self) -> None:
        """Advance physics/render snapshots without issuing a VLA action.

        IDLE/STANDBY must expose genuinely new camera frames so Endpoint-R can
        verify release/containment after a settle interval. The step is a
        controller no-op that preserves the current gripper command. As in
        RECOVERY mode, the LIBERO horizon counter is restored afterwards: only
        physics/simulation time advances, not the VLA action budget.

        RECOVERY calls this helper while perception/planning has no fresh joint
        command. This also keeps recovery geometry and grasp planning tied to the
        current simulator state.
        """
        if self._env is None:
            time.sleep(0.02)
            return
        mode = self.get_mode()
        if mode not in (Mode.IDLE, Mode.STANDBY):
            if mode == Mode.RECOVERY:
                self._refresh_recovery_robot_state()
            time.sleep(0.02)
            return
        now = time.monotonic()
        period = 1.0 / max(1e-3, self.recovery_step_hz)
        if now - self._last_idle_refresh_s < period:
            time.sleep(min(0.02, period))
            return

        env = self._env
        if self._active_controller_type == "OSC_POSE":
            action = [0.0] * 6 + [self._last_gripper_action()]
        else:
            action = [0.0] * len(self._arm_qpos()) + [
                self._last_gripper_action()
            ]
        robo = self._robo_env()
        ts_before = getattr(robo, "timestep", None) if robo is not None else None
        obs, _reward, done, _info = env.step(action)
        if ts_before is not None:
            try:
                robo.timestep = ts_before
                if getattr(robo, "done", False):
                    robo.done = False
                done = False
            except Exception:  # noqa: BLE001
                pass
        self._last_idle_refresh_s = now
        self._idle_refresh_steps += 1
        self._update_snapshots(obs, state_source="robot_observation_after_physics_step")
        if done:
            self._finish_from_env()

    # -- hold ------------------------------------------------------------- #
    def _tick_hold(self) -> None:
        env = self._env
        if env is None:
            time.sleep(0.02)
            return
        if self._active_controller_type == "OSC_POSE":
            action = [0.0] * 6 + [self._last_gripper_action()]
        else:
            cur_q = self._arm_qpos()
            action = [0.0] * len(cur_q) + [self._last_gripper_action()]
        obs, _reward, done, _info = env.step(action)
        self._update_snapshots(obs, state_source="robot_observation_after_physics_step")
        if done:
            self._finish_from_env()

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _arm_qpos(self) -> np.ndarray:
        robot = self._env.robots[0]
        return np.asarray(robot._joint_positions, dtype=np.float64)

    def _last_gripper_action(self) -> float:
        return getattr(self, "_last_grip", -1.0)

    def _current_obs_for_vla(self):
        if self._raw_obs is None:
            return None
        obs = self._raw_obs
        raw_agentview = obs[f"{self.camera_name}_image"]
        if raw_agentview.shape[:2] != (self.resolution, self.resolution):
            raise RuntimeError(
                f"VLA observation parity violation: {raw_agentview.shape}"
            )
        img = raw_agentview[::-1, ::-1]
        img = np.ascontiguousarray(img)
        wrist = _resize_rgb(obs["robot0_eye_in_hand_image"], self.resolution)[::-1, ::-1]
        wrist = np.ascontiguousarray(wrist)
        state = np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
             obs["robot0_gripper_qpos"])
        )
        return img, wrist, state

    _raw_obs = None

    def _update_snapshots(self, obs: dict, *, state_source: str = "robot_observation_snapshot") -> None:
        from robosuite.utils import camera_utils

        env = self._env
        self._raw_obs = obs
        pres = self.percep_resolution
        if pres != self.resolution:
            rgb, depth_norm = render_normalized_rgbd(
                env.sim,
                camera_name=self.camera_name,
                width=pres,
                height=pres,
                dump_dir=os.environ.get("P4_DEPTH_FORENSICS_DIR"),
                context=f"{self._mode.value}:{self.camera_name}:{pres}x{pres}",
            )
        else:
            rgb = np.ascontiguousarray(
                obs[f"{self.camera_name}_image"][::-1]
            )
            depth_key = f"{self.camera_name}_depth"
            depth_norm = (
                np.ascontiguousarray(obs[depth_key][::-1])
                if depth_key in obs else None
            )
        if depth_norm is None:
            depth_m = None
        else:
            depth_norm = require_normalized_depth(
                depth_norm,
                dump_dir=os.environ.get("P4_DEPTH_FORENSICS_DIR"),
                context=f"{self._mode.value}:{self.camera_name}:{pres}x{pres}",
            )
            depth_m = np.asarray(
                camera_utils.get_real_depth_map(env.sim, depth_norm)
            )
            if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
                depth_m = depth_m.squeeze(-1)
            if depth_m.ndim != 2:
                raise RuntimeError(
                    f"unexpected metric depth shape {depth_m.shape}"
                )

        # The served overhead frame is at percep_resolution; its intrinsics
        # must be computed at the same size or back-projection is scaled wrong.
        try:
            K = camera_utils.get_camera_intrinsic_matrix(
                env.sim, self.camera_name, pres, pres)
            ext = camera_utils.get_camera_extrinsic_matrix(env.sim, self.camera_name)
            intr = {
                "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                # Recovery detectors accept K/intrinsics or RealSense-style
                # ppx/ppy. Keep cx/cy for readability but expose the canonical
                # keys so the same HTTP camera contract works in sim and on-rig.
                "ppx": float(K[0, 2]), "ppy": float(K[1, 2]),
                "K": K.tolist(),
                "width": pres, "height": pres,
                "depth_scale": self.depth_scale,
                "extrinsics": ext.tolist(),
                "world_from_camera": ext.tolist(),
            }
        except Exception:
            intr = {"width": pres, "height": pres,
                    "depth_scale": self.depth_scale}

        robot = self._env.robots[0]
        q = np.asarray(robot._joint_positions, dtype=np.float64)
        dq = np.asarray(robot._joint_velocities, dtype=np.float64)
        ee_pos = np.asarray(self._raw_obs["robot0_eef_pos"], dtype=np.float64)
        ee_quat_xyzw = np.asarray(self._raw_obs["robot0_eef_quat"], dtype=np.float64)
        gq = np.asarray(self._raw_obs["robot0_gripper_qpos"], dtype=np.float64)
        w_meas = float(abs(gq[0] - gq[1])) if gq.size >= 2 and np.all(np.isfinite(gq[:2])) else None
        last_grip_action = self._last_gripper_action()
        release_commanded = bool(last_grip_action < 0.0)
        w_cmd = 0.08 if release_commanded else self.close_width_m

        producer_ts = time.time()
        try:
            simulation_time_s = float(env.sim.data.time)
        except Exception:
            simulation_time_s = None
        try:
            physics_dt = float(env.sim.model.opt.timestep)
            simulation_step_id = int(round(simulation_time_s / physics_dt))
        except Exception:
            simulation_step_id = None

        with self._lock:
            self._producer_frame_seq += 1
            self._frame = CameraFrame(
                rgb=rgb,
                depth_m=depth_m,
                intrinsics=intr,
                ts=producer_ts,
                producer_frame_seq=self._producer_frame_seq,
                simulation_step_id=simulation_step_id,
                simulation_time_s=simulation_time_s,
                camera_role="external_fixed",
                camera_source=f"sim_world:{self.camera_name}",
            )
            self._qualification_rgb_history.publish(self._frame)
            self._wrist_rgb = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
            self._state = {
                "type": "state", "state_source": state_source, "ts": time.time(),
                "q": q.tolist(), "dq": dq.tolist(),
                "ee_pos": ee_pos.tolist(), "ee_quat": ee_quat_xyzw.tolist(),
                "ee_vel": [0.0] * 6,
                "w_cmd": w_cmd, "w_meas": w_meas,
                "release_commanded": release_commanded,
            }
            # Keep physical measurements across later idle-read snapshot replacement.
            if state_source == 'robot_observation_after_physics_step' and w_meas is not None and np.isfinite(w_meas):
                history = getattr(self, '_physics_width_history', [])
                item = {k: self._state[k] for k in ('ts', 'w_meas', 'state_source')}
                if not history or item['ts'] > history[-1]['ts']:
                    history.append(item)
                    del history[:-240]
                self._physics_width_history = history

        # Sample at most 5 Hz. This is written only into the evaluator result
        # after the episode; the runtime control plane has no endpoint for it.
        now = time.time()
        if (
            not self._offline_gt_timeline
            or now - self._offline_gt_timeline[-1]["ts"] >= 0.2
        ):
            name = getattr(self.disturbance, "_object_name", None)
            px = self.object_target_px(name) if name else None
            if px is not None:
                self._offline_gt_timeline.append({
                    "ts": now,
                    "vla_step": int(self._step),
                    "object": str(name),
                    "normalized_point": [float(px[0]), float(px[1])],
                })
                del self._offline_gt_timeline[:-5000]

    @property
    def obj_of_interest(self):
        """Task target object body names (e.g. ['milk_1', 'basket_1'])."""
        env = self._env
        for obj in (env, getattr(env, "env", None)):
            oi = getattr(obj, "obj_of_interest", None)
            if oi:
                return list(oi)
        return []

    def object_target_px(self, obj_name: str):
        """Project the simulator object position into camera pixel coordinates."""
        from robosuite.utils import camera_utils

        obs = self._raw_obs
        if obs is None:
            return None
        pos_key = f"{obj_name}_pos"
        if pos_key not in obs:
            return None
        world_pos = np.asarray(obs[pos_key], dtype=np.float64).reshape(1, 3)
        pres = self.percep_resolution
        try:
            # world -> pixel in the RAW (un-flipped) render frame.
            w2p = camera_utils.get_camera_transform_matrix(
                self._env.sim, self.camera_name, pres, pres)
            px = camera_utils.project_points_from_world_to_camera(
                world_pos, w2p, pres, pres)  # returns (row, col) = (y, x)
        except Exception:
            return None
        row, col = float(px[0, 0]), float(px[0, 1])
        # Camera projection already uses the upright frame served by RGB/depth.
        # Do not apply an additional vertical flip.
        x_norm = min(1.0, max(0.0, col / float(pres)))
        y_norm = min(1.0, max(0.0, row / float(pres)))
        return (x_norm, y_norm)

    def object_world_xy(self, obj_name: str):
        """Return the simulator object's ground-truth world-frame XY coordinates in metres."""
        obs = self._raw_obs
        if obs is None:
            return None
        pos_key = f"{obj_name}_pos"
        if pos_key not in obs:
            return None
        pos = np.asarray(obs[pos_key], dtype=np.float64).reshape(-1)
        if pos.size < 2:
            return None
        return (float(pos[0]), float(pos[1]))

    def _on_episode_done(self, success: bool, terminal_reason: str) -> None:
        import logging
        logging.info(
            "episode done: success=%s terminal_reason=%s after %d VLA steps "
            "(mode=%s, recovery_frozen_steps=%d)",
            success, terminal_reason, self._step, self._mode, self._recovery_frozen_steps,
        )
        with self._lock:
            self._mode = Mode.IDLE
            self._episode_over = True
        cb = getattr(self, "on_episode_complete", None)
        if cb is not None:
            cb(success, terminal_reason)
