"""Disturbance injectors for LIBERO recovery evaluation.

Injectors use simulator state to perturb the rollout. They implement hold_lost
(object drop), ee_trans_x/y/z (bounded action displacement), xy_offset (object
translation), place_offset (release displacement), and noop_freeze (no-op
window). These perturbations follow the object-drop and translation/no-op
settings of Code-as-Monitor and FailSafe.

For hold_lost, drop_prob is a per-step probability after lift and max_applications
caps the number of drops. Translation offsets are applied over trans_steps;
trans_randomize=False uses the fixed action_bias. Each injector operates on the
environment passed by the simulator runner."""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np

# Action layout used by LIBERO OSC_POSE: [dx, dy, dz, droll, dpitch, dyaw, gripper]
GRIPPER_IDX = 6
ACTION_DIM = 7


# --------------------------------------------------------------------------- #
# Low-level env access (privileged; examiner-only)                            #
# --------------------------------------------------------------------------- #
def _unwrap(env):
    """Return the underlying robosuite env that exposes ``get_object`` / ``sim``.

    The eval code uses ``OffScreenRenderEnv`` (a ``ControlEnv`` wrapper) whose
    ``.env`` is the actual robosuite ``*_Manipulation`` domain. We accept either
    and walk down to the one that has ``get_object``.
    """
    cur = env
    for _ in range(4):
        if hasattr(cur, "get_object") and hasattr(cur, "sim"):
            return cur
        inner = getattr(cur, "env", None)
        if inner is None:
            break
        cur = inner
    # Fall back to the outermost; callers guard for missing attrs.
    return env


def target_object_name(env) -> Optional[str]:
    """The primary object of interest for the task (the thing to manipulate).

    LIBERO's ``obj_of_interest`` lists the task-relevant instances; for
    pick-and-place the first entry is the movable object to grasp.
    """
    base = _unwrap(env)
    ooi = getattr(base, "obj_of_interest", None)
    if not ooi:
        return None
    return ooi[0]


def _object_free_joint(env, object_name: str) -> Optional[str]:
    base = _unwrap(env)
    try:
        obj = base.get_object(object_name)
    except Exception:
        return None
    joints = getattr(obj, "joints", None)
    if not joints:
        return None
    return joints[-1]


def get_object_pose(env, object_name: str) -> Optional[np.ndarray]:
    """Return the object's free-joint qpos ``[x, y, z, qw, qx, qy, qz]`` or None."""
    base = _unwrap(env)
    jnt = _object_free_joint(env, object_name)
    if jnt is None:
        return None
    try:
        return np.asarray(base.sim.data.get_joint_qpos(jnt), dtype=np.float64).copy()
    except Exception:
        return None


def set_object_pose(env, object_name: str, pose7: np.ndarray) -> bool:
    """Teleport the object: write its free-joint qpos and forward the sim."""
    base = _unwrap(env)
    jnt = _object_free_joint(env, object_name)
    if jnt is None:
        return False
    try:
        base.sim.data.set_joint_qpos(jnt, np.asarray(pose7, dtype=np.float64))
        # Zero the object's free-joint velocity so it doesn't fling after a
        # teleport (the 6-dof qvel for this joint).
        try:
            qvel_addr = base.sim.model.get_joint_qvel_addr(jnt)
            lo, hi = qvel_addr
            base.sim.data.qvel[lo:hi] = 0.0
        except Exception:
            pass
        base.sim.forward()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Config + base                                                               #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DisturbanceConfig:
    # none|hold_lost|ee_trans_x|ee_trans_y|ee_trans_z|xy_offset|place_offset|noop_freeze
    kind: str = "none"
    # hold_lost / triggers
    lift_trigger_dz: float = 0.04    # object risen this far above its rest z => "grasped & lifted"
    drop_clearance: float = 0.0      # extra z above table when re-seating the dropped object
    # hold_lost: per-step drop probability after the lift. 1.0 => deterministic
    # drop on the first lifted step (legacy behaviour); <1 => sampled each step.
    drop_prob: float = 1.0
    # ee_trans_{x,y,z}: ONE-SHOT bounded offset (FailSafe stage-target analogue).
    # trans_randomize=True (default): at reset sample total offset =
    #   trans_noise * (rand()-0.5) along the fixed axis (sign + magnitude random,
    #   range [-trans_noise/2, +trans_noise/2]), then spread over trans_steps.
    # trans_randomize=False: fixed per-step action_bias for reproducible demos.
    trans_noise: float = 2.0
    trans_randomize: bool = True
    action_bias: float = 0.20   # per-step increment when trans_randomize=False
    trans_steps: int = 8
    # hold_lost: scatter radius (m) for the drop landing point. 0 = drop straight
    # down at the current xy (legacy). >0 = displace the landing point by a
    # uniformly random direction with this radius, so the object does not sit
    # conveniently under the gripper after the slip.
    drop_scatter_m: float = 0.0
    # Deterministic object relocation (dx, dy) in table-frame metres.
    # A nonzero component overrides random drop scatter.
    drop_offset_xy: tuple = (0.0, 0.0)
    # ee_rot_{roll,pitch,yaw}: same one-shot bounded injection as ee_trans but on
    # the OSC rotation action axes. ``rot_phase`` picks the window: "pre_grasp"
    # (before the object is lifted; a closed-loop VLA is expected to correct it,
    # mirroring the ee_trans negative result) or "carry" (first steps after the
    # lift, twisting the WRIST WHILE HOLDING -- reorients the grasped object and
    # attacks placement alignment).
    rot_phase: str = "pre_grasp"
    # xy_offset / place_offset
    offset_xy: tuple = (0.06, 0.0)   # (dx, dy) meters applied on the table plane
    # trigger by step (absolute, post-wait) if >=0, else on first lift (xy_offset)
    # / from step 0 (ee_trans).
    trigger_step: int = -1
    # noop_freeze
    freeze_start_step: int = 30
    freeze_duration: int = 25
    # general
    max_applications: int = 1        # one-shot by default
    seed: int = 0


class Disturbance:
    """Base injector. Subclasses implement ``_apply_obs`` and/or ``filter_action``."""

    def __init__(self, cfg: DisturbanceConfig):
        self.cfg = cfg
        self._applied = 0
        self._disabled = False
        self._rng = np.random.RandomState(cfg.seed)
        self._object_name: Optional[str] = None
        self._rest_pose: Optional[np.ndarray] = None
        self._was_lifted = False
        self._gripper_was_closed = False
        self.events: list = []

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, env, obs) -> None:
        """Called after ``set_init_state`` + settle, before the policy loop."""
        self._applied = 0
        self._was_lifted = False
        self._gripper_was_closed = False
        self._disabled = False
        self.events = []
        self._object_name = target_object_name(env)
        self._rest_pose = (
            get_object_pose(env, self._object_name) if self._object_name else None
        )

    def disable(self) -> None:
        """Stop applying this disturbance (e.g. after recovery handback resumes VLA)."""
        self._disabled = True

    @property
    def disabled(self) -> bool:
        return bool(getattr(self, "_disabled", False))

    @property
    def done(self) -> bool:
        return self._disabled or self._applied >= self.cfg.max_applications

    # -- hooks ------------------------------------------------------------ #
    def on_obs(self, env, obs, step: int) -> Optional[dict]:
        """Optionally mutate the sim from the latest obs. Returns an event dict
        when a disturbance fired this step, else None."""
        if self.disabled or self._applied >= self.cfg.max_applications or self._object_name is None:
            return None
        return self._apply_obs(env, obs, step)

    def filter_action(self, action: np.ndarray, step: int) -> np.ndarray:
        """Optionally override the VLA action (used by noop_freeze)."""
        if self.disabled:
            return action
        return action

    # -- helpers ---------------------------------------------------------- #
    def _object_lifted(self, env) -> bool:
        if self._rest_pose is None:
            return False
        cur = get_object_pose(env, self._object_name)
        if cur is None:
            return False
        return (cur[2] - self._rest_pose[2]) > self.cfg.lift_trigger_dz

    def _record(self, step: int, kind: str, **extra) -> dict:
        ev = {"step": int(step), "kind": kind, "object": self._object_name,
              "ts_wall": time.time(), **extra}
        self.events.append(ev)
        self._applied += 1
        return ev

    def _apply_obs(self, env, obs, step: int) -> Optional[dict]:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# Concrete injectors                                                          #
# --------------------------------------------------------------------------- #
class NoDisturbance(Disturbance):
    pass


class HoldLostDisturbance(Disturbance):
    """Drop the object after it has been grasped and lifted.

    ``drop_prob`` controls the per-step slip probability once the object is
    lifted: 1.0 drops deterministically on the first lifted step (legacy), while
    a smaller value samples the drop each step. At most ``max_applications`` drops
    occur (the ``on_obs`` guard enforces this).
    """

    def _apply_obs(self, env, obs, step: int) -> Optional[dict]:
        if not self._object_lifted(env):
            return None
        # Per-step slip: with probability (1 - drop_prob) the object stays held
        # this step and may slip on a later step instead.
        drop_prob = float(getattr(self.cfg, "drop_prob", 1.0))
        if drop_prob < 1.0 and self._rng.rand() >= drop_prob:
            return None
        cur = get_object_pose(env, self._object_name)
        if cur is None or self._rest_pose is None:
            return None
        # Re-seat the object on the table at its CURRENT xy (where it was lifted
        # from), at the table-rest height -> simulates it slipping straight down.
        dropped = cur.copy()
        dropped[2] = self._rest_pose[2] + self.cfg.drop_clearance
        # Keep an upright-ish orientation = the rest orientation.
        dropped[3:7] = self._rest_pose[3:7]
        scatter = float(getattr(self.cfg, "drop_scatter_m", 0.0) or 0.0)
        off = getattr(self.cfg, "drop_offset_xy", (0.0, 0.0)) or (0.0, 0.0)
        offx, offy = float(off[0]), float(off[1])
        sdx = sdy = 0.0
        mode = "in_place"
        if offx != 0.0 or offy != 0.0:
            # Apply deterministic object relocation before considering random scatter.
            sdx, sdy = offx, offy
            dropped[0] += offx
            dropped[1] += offy
            mode = "removal"
        elif scatter > 0.0:
            # Uniform random direction, fixed radius: the object tumbles AWAY
            # from the slip point instead of landing conveniently under the
            # gripper (which the VLA trivially re-grasps).
            ang = self._rng.rand() * 2.0 * np.pi
            sdx, sdy = scatter * np.cos(ang), scatter * np.sin(ang)
            dropped[0] += sdx
            dropped[1] += sdy
            mode = "scatter"
        drop_from = cur.copy()
        ok = set_object_pose(env, self._object_name, dropped)
        if not ok:
            return None
        observed = get_object_pose(env, self._object_name)
        applied_delta_xy = (
            None
            if observed is None
            else [
                float(observed[0] - drop_from[0]),
                float(observed[1] - drop_from[1]),
            ]
        )
        return self._record(
            step, "hold_lost", drop_prob=drop_prob, drop_to=dropped[:3].tolist(),
            drop_from=drop_from[:3].tolist(),
            drop_observed_to=(None if observed is None else observed[:3].tolist()),
            applied_delta_xy=applied_delta_xy,
            drop_scatter_m=float(getattr(self.cfg, "drop_scatter_m", 0.0) or 0.0),
            drop_offset_xy=[offx, offy], drop_mode=mode,
            # Applied offset, logged for cross-arm seed-parity verification.
            scatter_dx=float(sdx), scatter_dy=float(sdy),
        )


class XYOffsetDisturbance(Disturbance):
    """Shove the target object by (dx, dy) on the table, mid-episode."""

    def _should_trigger(self, env, step: int) -> bool:
        if self.cfg.trigger_step >= 0:
            return step >= self.cfg.trigger_step
        # else trigger as soon as it's grasped & lifted
        return self._object_lifted(env)

    def _apply_obs(self, env, obs, step: int) -> Optional[dict]:
        if not self._should_trigger(env, step):
            return None
        cur = get_object_pose(env, self._object_name)
        if cur is None:
            return None
        moved = cur.copy()
        moved[0] += float(self.cfg.offset_xy[0])
        moved[1] += float(self.cfg.offset_xy[1])
        ok = set_object_pose(env, self._object_name, moved)
        if not ok:
            return None
        return self._record(
            step, "xy_offset", offset=list(self.cfg.offset_xy), to=moved[:3].tolist()
        )


class PlaceOffsetDisturbance(Disturbance):
    """On release (gripper opens after being closed while lifted), nudge the
    object so it lands off the intended target region."""

    def _apply_obs(self, env, obs, step: int) -> Optional[dict]:
        # Track lift + gripper-close so we can detect a release of a held object.
        lifted = self._object_lifted(env)
        self._was_lifted = self._was_lifted or lifted
        grip = _gripper_closed_from_obs(obs)
        released = self._gripper_was_closed and (grip is False)
        self._gripper_was_closed = bool(grip) if grip is not None else self._gripper_was_closed
        if not (self._was_lifted and released):
            return None
        cur = get_object_pose(env, self._object_name)
        if cur is None:
            return None
        moved = cur.copy()
        moved[0] += float(self.cfg.offset_xy[0])
        moved[1] += float(self.cfg.offset_xy[1])
        ok = set_object_pose(env, self._object_name, moved)
        if not ok:
            return None
        return self._record(
            step, "place_offset", offset=list(self.cfg.offset_xy), to=moved[:3].tolist()
        )


class NoopFreezeDisturbance(Disturbance):
    """Freeze the arm: override the action to a no-op for a window of steps,
    keeping the gripper command as the last VLA value."""

    def __init__(self, cfg: DisturbanceConfig):
        super().__init__(cfg)
        self._last_gripper = -1.0

    def filter_action(self, action: np.ndarray, step: int) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if self.disabled:
            return a
        if a.size >= ACTION_DIM:
            self._last_gripper = float(a[GRIPPER_IDX])
        start = self.cfg.freeze_start_step
        end = start + self.cfg.freeze_duration
        if start <= step < end:
            if self._applied < self.cfg.max_applications and step == start:
                self._record(step, "noop_freeze",
                             window=[start, end])
            frozen = np.zeros(ACTION_DIM, dtype=np.float32)
            frozen[GRIPPER_IDX] = self._last_gripper  # hold gripper, freeze motion
            return frozen
        return a


class EETransDisturbance(Disturbance):
    """FailSafe translation failure on one world-vector axis (pre-grasp).

    Faithful to FailSafe ``target_pose.p += noise * (rand()-0.5) * axis`` on a
    single stage: the axis is fixed by ``kind`` (ee_trans_x/y/z); sign and
    magnitude along that axis are sampled once per episode when
    ``trans_randomize=True`` (range ``[-trans_noise/2, +trans_noise/2]``).

    Closed-loop realisation: the sampled total offset is injected evenly over
    ``trans_steps`` consecutive action steps at the trigger (pre-grasp), then the
    disturbance disables itself and the VLA continues from the displaced pose.

    The acting axis is taken from ``kind``: ``ee_trans_x|ee_trans_y|ee_trans_z``.
    ``trigger_step < 0`` is treated as "start immediately" (step 0).
    """

    _AXIS = {"ee_trans_x": 0, "ee_trans_y": 1, "ee_trans_z": 2}

    def __init__(self, cfg: DisturbanceConfig):
        super().__init__(cfg)
        self._axis_idx = self._AXIS[(cfg.kind or "").lower()]
        self._injected = 0
        self._recorded = False
        self._sampled_offset = 0.0
        self._per_step_bias = 0.0

    def reset(self, env, obs) -> None:
        super().reset(env, obs)
        self._injected = 0
        self._recorded = False
        n_steps = max(1, int(self.cfg.trans_steps))
        if bool(getattr(self.cfg, "trans_randomize", True)):
            noise = float(getattr(self.cfg, "trans_noise", 1.0))
            # FailSafe: noise * (rand()-0.5)  ->  [-noise/2, +noise/2]
            self._sampled_offset = noise * (self._rng.rand() - 0.5)
        else:
            self._sampled_offset = float(self.cfg.action_bias) * n_steps
        self._per_step_bias = self._sampled_offset / n_steps

    def on_obs(self, env, obs, step: int) -> Optional[dict]:
        if self.disabled or self._object_name is None:
            return None
        if not self._was_lifted and self._object_lifted(env):
            self._was_lifted = True
        return None

    def filter_action(self, action: np.ndarray, step: int) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if self.disabled:
            return a
        start = self.cfg.trigger_step if self.cfg.trigger_step >= 0 else 0
        pre_grasp = not self._was_lifted
        n_steps = max(1, int(self.cfg.trans_steps))
        active = (
            step >= start
            and pre_grasp
            and self._injected < n_steps
            and a.size > self._axis_idx
        )
        if active:
            if not self._recorded:
                self._record(
                    step,
                    (self.cfg.kind or "ee_trans").lower(),
                    axis=self._axis_idx,
                    sampled_offset=float(self._sampled_offset),
                    per_step_bias=float(self._per_step_bias),
                    trans_noise=float(getattr(self.cfg, "trans_noise", 0.0)),
                    trans_randomize=bool(getattr(self.cfg, "trans_randomize", True)),
                    trans_steps=n_steps,
                )
                self._recorded = True
            a[self._axis_idx] += float(self._per_step_bias)
            self._injected += 1
        elif self._injected >= n_steps and not self.disabled:
            self.disable()
        return a


class EERotDisturbance(Disturbance):
    """One-shot bounded ROTATION offset on an OSC rotation action axis.

    Mirrors :class:`EETransDisturbance` (same ``trans_noise`` sampling and
    ``trans_steps`` spreading) but on droll/dpitch/dyaw (action indices 3/4/5).
    ``rot_phase`` selects the injection window:

    - ``pre_grasp``: before the object is lifted (expected to be corrected by a
      closed-loop VLA, mirroring the ee_trans negative result);
    - ``carry``: the first ``trans_steps`` steps AFTER the lift -- twists the
      wrist while the object is held, reorienting the grasped object and
      attacking placement alignment (a failure the VLA cannot undo by simply
      re-observing the target).
    """

    _AXIS = {"ee_rot_roll": 3, "ee_rot_pitch": 4, "ee_rot_yaw": 5}

    def __init__(self, cfg: DisturbanceConfig):
        super().__init__(cfg)
        self._axis_idx = self._AXIS[(cfg.kind or "").lower()]
        self._injected = 0
        self._recorded = False
        self._sampled_offset = 0.0
        self._per_step_bias = 0.0

    def reset(self, env, obs) -> None:
        super().reset(env, obs)
        self._injected = 0
        self._recorded = False
        n_steps = max(1, int(self.cfg.trans_steps))
        if bool(getattr(self.cfg, "trans_randomize", True)):
            noise = float(getattr(self.cfg, "trans_noise", 1.0))
            self._sampled_offset = noise * (self._rng.rand() - 0.5)
        else:
            self._sampled_offset = float(self.cfg.action_bias) * n_steps
        self._per_step_bias = self._sampled_offset / n_steps

    def on_obs(self, env, obs, step: int) -> Optional[dict]:
        if self.disabled or self._object_name is None:
            return None
        if not self._was_lifted and self._object_lifted(env):
            self._was_lifted = True
        return None

    def filter_action(self, action: np.ndarray, step: int) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if self.disabled:
            return a
        phase = (getattr(self.cfg, "rot_phase", "pre_grasp") or "pre_grasp").lower()
        start = self.cfg.trigger_step if self.cfg.trigger_step >= 0 else 0
        if phase == "carry":
            phase_ok = self._was_lifted
        else:
            phase_ok = (not self._was_lifted) and step >= start
        n_steps = max(1, int(self.cfg.trans_steps))
        active = phase_ok and self._injected < n_steps and a.size > self._axis_idx
        if active:
            if not self._recorded:
                self._record(
                    step,
                    (self.cfg.kind or "ee_rot").lower(),
                    axis=self._axis_idx,
                    rot_phase=phase,
                    sampled_offset=float(self._sampled_offset),
                    per_step_bias=float(self._per_step_bias),
                    trans_steps=n_steps,
                )
                self._recorded = True
            a[self._axis_idx] += float(self._per_step_bias)
            self._injected += 1
        elif self._injected >= n_steps and not self.disabled:
            self.disable()
        return a


def _gripper_closed_from_obs(obs) -> Optional[bool]:
    """Best-effort: infer gripper-closed from Panda finger qpos in the obs.

    Panda's two prismatic finger joints open to ~0.04 each (~0.08 gap). We treat
    a small gap as 'closed'. Returns None if the signal isn't present.
    """
    q = obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
    if q is None:
        return None
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size < 2:
        return None
    gap = abs(float(q[0]) - float(q[1]))
    return gap < 0.04  # < ~half-open => grasped/closed


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "none": NoDisturbance,
    "hold_lost": HoldLostDisturbance,
    "ee_trans_x": EETransDisturbance,
    "ee_trans_y": EETransDisturbance,
    "ee_trans_z": EETransDisturbance,
    "ee_rot_roll": EERotDisturbance,
    "ee_rot_pitch": EERotDisturbance,
    "ee_rot_yaw": EERotDisturbance,
    "xy_offset": XYOffsetDisturbance,
    "place_offset": PlaceOffsetDisturbance,
    "noop_freeze": NoopFreezeDisturbance,
}


def make_disturbance(cfg: DisturbanceConfig) -> Disturbance:
    kind = (cfg.kind or "none").lower()
    if kind not in _REGISTRY:
        raise ValueError(f"unknown disturbance kind={cfg.kind!r}; options={list(_REGISTRY)}")
    return _REGISTRY[kind](cfg)


def list_disturbances() -> list:
    return [k for k in _REGISTRY if k != "none"]
