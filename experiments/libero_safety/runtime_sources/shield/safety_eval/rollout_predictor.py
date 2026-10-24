#!/usr/bin/env python3
"""Robot-only, in-place MuJoCo rollout for the relational shield.

The predictor deliberately treats ``sim.get_state()`` as an opaque restoration
token. The only state inspected to build predictions is the robot's seven arm
joints, two finger joints, and the robot-observation EE pose. Scene bodies, geoms,
contacts, BDDL predicates, observations, and post-action hooks are never read.
"""
from __future__ import annotations

import copy
import time

import numpy as np


CONTACT_DISABLE_BIT = 16
MAX_ROLLOUT_STEPS = 5

DEFAULT_PROPRIO_LAYOUT = {
    "joint_pos": slice(0, 7),
    "eef_pos": slice(7, 10),
    "eef_quat": slice(10, 14),
    "gripper_qpos": slice(14, 16),
    "size": 16,
}

_CONTROLLER_ARRAY_STATE = (
    "action_scale",
    "action_input_transform",
    "action_output_transform",
    "ee_pos",
    "ee_ori_mat",
    "ee_pos_vel",
    "ee_ori_vel",
    "joint_pos",
    "joint_vel",
    "J_pos",
    "J_ori",
    "J_full",
    "mass_matrix",
    "new_update",
    "goal_pos",
    "goal_ori",
    "relative_ori",
    "ori_ref",
    "torques",
)
_INTERPOLATORS = ("interpolator_pos", "interpolator_ori")


def _copied(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _snapshot_attrs(obj, names=None):
    source = vars(obj)
    keys = tuple(source) if names is None else tuple(name for name in names if hasattr(obj, name))
    return {name: _copied(getattr(obj, name)) for name in keys}


def _restore_attrs(obj, snapshot):
    for name, value in snapshot.items():
        current = getattr(obj, name, None)
        if isinstance(current, np.ndarray) and isinstance(value, np.ndarray) and current.shape == value.shape:
            current[...] = value
        else:
            setattr(obj, name, _copied(value))


def _equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except TypeError:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _as_slice(layout, name, default):
    value = layout.get(name, default)
    if isinstance(value, slice):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return slice(int(value[0]), int(value[1]))
    raise TypeError("proprio_layout[%r] must be a slice or (start, stop)" % name)


class RolloutPredictor:
    """Roll the live robot/controller forward for at most five policy steps.

    ``base_env`` is the underlying robosuite ``MujocoEnv`` (the object exposed
    as ``OffScreenRenderEnv.env`` in the runner). ``fk_spheres_fn`` is the
    signed adapter's ``_fk_spheres`` callable. ``proprio_layout`` is a mapping
    with ``joint_pos``, ``eef_pos``, ``eef_quat``, ``gripper_qpos`` slices and
    optional ``size``; passing ``None`` selects the policy-visible 7+3+4+2
    layout.
    """

    def __init__(self, base_env, fk_spheres_fn, proprio_layout):
        self.base_env = base_env
        self.sim = base_env.sim
        self.robot = base_env.robots[0]
        self.controller = self.robot.controller
        self.fk_spheres_fn = fk_spheres_fn

        layout = dict(DEFAULT_PROPRIO_LAYOUT if proprio_layout is None else proprio_layout)
        self._joint_slice = _as_slice(layout, "joint_pos", DEFAULT_PROPRIO_LAYOUT["joint_pos"])
        self._eef_pos_slice = _as_slice(layout, "eef_pos", DEFAULT_PROPRIO_LAYOUT["eef_pos"])
        self._eef_quat_slice = _as_slice(layout, "eef_quat", DEFAULT_PROPRIO_LAYOUT["eef_quat"])
        self._gripper_slice = _as_slice(layout, "gripper_qpos", DEFAULT_PROPRIO_LAYOUT["gripper_qpos"])
        stops = [
            self._joint_slice.stop,
            self._eef_pos_slice.stop,
            self._eef_quat_slice.stop,
            self._gripper_slice.stop,
        ]
        self._proprio_size = int(layout.get("size", max(stops)))
        self._validate_layout()

        # Match robosuite's eef_pos/eef_quat observation contract used by
        # the held latch: grip-site origin, robot-model end-effector axes.
        self._eef_site_id = int(self.robot.eef_site_id)
        self._eef_body_name = self.robot.robot_model.eef_name

        self.last_restore_fidelity_ok = False
        self.last_latency_ms = None
        self._busy = False

    def _validate_layout(self):
        widths = (
            self._joint_slice.stop - self._joint_slice.start,
            self._eef_pos_slice.stop - self._eef_pos_slice.start,
            self._eef_quat_slice.stop - self._eef_quat_slice.start,
            self._gripper_slice.stop - self._gripper_slice.start,
        )
        if widths != (7, 3, 4, 2):
            raise ValueError("proprio layout must have widths 7+3+4+2, got %r" % (widths,))
        if min(s.start for s in (
                self._joint_slice, self._eef_pos_slice,
                self._eef_quat_slice, self._gripper_slice)) < 0:
            raise ValueError("proprio layout slices must be non-negative")
        if self._proprio_size < max(s.stop for s in (
                self._joint_slice, self._eef_pos_slice,
                self._eef_quat_slice, self._gripper_slice)):
            raise ValueError("proprio layout size is smaller than a field stop")

    @property
    def _model(self):
        return getattr(self.sim.model, "_model", self.sim.model)

    @property
    def _data(self):
        return self.sim.data

    def _save_state(self):
        data = self._data
        controller_state = _snapshot_attrs(self.controller, _CONTROLLER_ARRAY_STATE)
        interpolators = {}
        for name in _INTERPOLATORS:
            value = getattr(self.controller, name, None)
            interpolators[name] = None if value is None else _snapshot_attrs(value)
        recent = {
            name: _snapshot_attrs(value)
            for name, value in vars(self.robot).items()
            if name.startswith("recent_") and hasattr(value, "__dict__")
        }
        return {
            "sim_state": self.sim.get_state(),
            "qacc_warmstart": np.asarray(data.qacc_warmstart).copy(),
            "act": (np.asarray(data.act).copy() if hasattr(data, "act") else None),
            "ctrl": np.asarray(data.ctrl).copy(),
            "qfrc_applied": np.asarray(data.qfrc_applied).copy(),
            "xfrc_applied": np.asarray(data.xfrc_applied).copy(),
            "disableflags": int(self._model.opt.disableflags),
            "controller": controller_state,
            "interpolators": interpolators,
            "recent": recent,
            "robot_torques": _copied(getattr(self.robot, "torques", None)),
            "gripper_current_action": _copied(
                getattr(getattr(self.robot, "gripper", None), "current_action", None)),
            "timestep": _copied(getattr(self.base_env, "timestep", None)),
            "cur_time": _copied(getattr(self.base_env, "cur_time", None)),
        }

    def _restore_state(self, saved):
        data = self._data
        self._model.opt.disableflags = saved["disableflags"]
        self.sim.set_state(saved["sim_state"])
        data.qacc_warmstart[:] = saved["qacc_warmstart"]
        if saved["act"] is not None:
            data.act[:] = saved["act"]
        data.ctrl[:] = saved["ctrl"]
        data.qfrc_applied[:] = saved["qfrc_applied"]
        data.xfrc_applied[:] = saved["xfrc_applied"]
        self.sim.forward()
        _restore_attrs(self.controller, saved["controller"])
        for name, snapshot in saved["interpolators"].items():
            if snapshot is not None:
                _restore_attrs(getattr(self.controller, name), snapshot)
        for name, snapshot in saved["recent"].items():
            _restore_attrs(getattr(self.robot, name), snapshot)
        current = getattr(self.robot, "torques", None)
        if (isinstance(current, np.ndarray) and
                isinstance(saved["robot_torques"], np.ndarray) and
                current.shape == saved["robot_torques"].shape):
            current[...] = saved["robot_torques"]
        else:
            self.robot.torques = _copied(saved["robot_torques"])
        if getattr(self.robot, "gripper", None) is not None:
            current = getattr(self.robot.gripper, "current_action", None)
            value = saved["gripper_current_action"]
            if (isinstance(current, np.ndarray) and isinstance(value, np.ndarray) and
                    current.shape == value.shape):
                current[...] = value
            else:
                self.robot.gripper.current_action = _copied(value)

    def _restored_exactly(self, saved):
        now = self.sim.get_state()
        before = saved["sim_state"]
        checks = [
            _equal(now.time, before.time),
            _equal(now.qpos, before.qpos),
            _equal(now.qvel, before.qvel),
            _equal(np.asarray(self._data.qacc_warmstart), saved["qacc_warmstart"]),
            _equal(np.asarray(self._data.ctrl), saved["ctrl"]),
            int(self._model.opt.disableflags) == saved["disableflags"],
            _equal(_snapshot_attrs(self.controller, _CONTROLLER_ARRAY_STATE), saved["controller"]),
            _equal(getattr(self.base_env, "timestep", None), saved["timestep"]),
            _equal(getattr(self.base_env, "cur_time", None), saved["cur_time"]),
        ]
        for name, snapshot in saved["interpolators"].items():
            current = getattr(self.controller, name, None)
            checks.append(snapshot is None if current is None else _equal(_snapshot_attrs(current), snapshot))
        for name, snapshot in saved["recent"].items():
            checks.append(_equal(_snapshot_attrs(getattr(self.robot, name)), snapshot))
        checks.append(_equal(getattr(self.robot, "torques", None), saved["robot_torques"]))
        checks.append(_equal(
            getattr(getattr(self.robot, "gripper", None), "current_action", None),
            saved["gripper_current_action"]))
        return bool(all(checks))

    def _robot_proprio(self):
        from robosuite.utils import transform_utils as transform

        data = self._data
        joint_pos = np.asarray(data.qpos)[self.robot._ref_joint_pos_indexes].copy()
        gripper_qpos = np.asarray(data.qpos)[self.robot._ref_gripper_joint_pos_indexes].copy()
        ee_pos = np.asarray(data.site_xpos[self._eef_site_id], dtype=np.float64).copy()
        ee_quat = transform.convert_quat(
            np.asarray(data.get_body_xquat(self._eef_body_name), dtype=np.float64),
            to="xyzw",
        )
        proprio = np.zeros(self._proprio_size, dtype=np.float64)
        proprio[self._joint_slice] = joint_pos
        proprio[self._eef_pos_slice] = ee_pos
        proprio[self._eef_quat_slice] = ee_quat
        proprio[self._gripper_slice] = gripper_qpos
        return proprio, ee_pos, np.asarray(ee_quat, dtype=np.float64).copy()

    def _gripper_spheres(self, proprio):
        result = self.fk_spheres_fn(proprio)
        spheres = result[1] if isinstance(result, tuple) and len(result) == 2 else result
        return [(np.asarray(center, dtype=np.float64).copy(), float(radius)) for center, radius in spheres]

    def predict(self, chunk_rows):
        """Return post-step robot geometry for ``chunk_rows[:5]``."""
        if self._busy:
            raise RuntimeError("RolloutPredictor.predict is not re-entrant")
        chunk = np.asarray(chunk_rows, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError("chunk_rows must be a 2-D action array")
        chunk = chunk[:MAX_ROLLOUT_STEPS]
        if len(chunk) < 1:
            raise ValueError("chunk_rows must contain at least one action")
        if chunk.shape[1] != int(self.robot.action_dim):
            raise ValueError(
                "action width %d does not match robot.action_dim %d" %
                (chunk.shape[1], int(self.robot.action_dim)))

        started = time.perf_counter_ns()
        self._busy = True
        saved = None
        predictions = []
        try:
            saved = self._save_state()
            self._model.opt.disableflags = saved["disableflags"] | CONTACT_DISABLE_BIT
            if not (int(self._model.opt.disableflags) & CONTACT_DISABLE_BIT):
                raise RuntimeError("MuJoCo contact-disable bit did not latch")

            n_substeps = int(self.base_env.control_timestep / self.base_env.model_timestep)
            if n_substeps <= 0:
                raise RuntimeError("control_timestep/model_timestep produced no substeps")
            for k, action in enumerate(chunk):
                policy_step = True
                for _ in range(n_substeps):
                    self.sim.forward()
                    self.base_env._pre_action(action, policy_step)
                    self.sim.step()
                    policy_step = False
                self.sim.forward()
                proprio, ee_pos, ee_quat = self._robot_proprio()
                predictions.append({
                    "k": int(k),
                    "proprio_k": proprio,
                    "ee_pos_k": ee_pos,
                    "ee_quat_k": ee_quat,
                    "spheres_k": self._gripper_spheres(proprio),
                })
        finally:
            if saved is not None:
                self._restore_state(saved)
                self.last_restore_fidelity_ok = self._restored_exactly(saved)
            else:
                self.last_restore_fidelity_ok = False
            self.last_latency_ms = (time.perf_counter_ns() - started) / 1.0e6
            self._busy = False
        if not self.last_restore_fidelity_ok:
            raise RuntimeError("RolloutPredictor state restoration fidelity check failed")
        return predictions
