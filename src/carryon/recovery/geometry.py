"""Geometry backends for motion planning, IK and grasp perception.

GeometryValidator checks recorded grasps and rewind paths. GeoCheck.ok is True
for a pass, False for failure and None when the check cannot run.
KinematicBoundsValidator checks reachability bounds and joint limits without
collision checking or IK. cuRobo, torch and perception dependencies are loaded
lazily. Unavailable backends leave the corresponding operation disabled."""



from __future__ import annotations



import atexit

import contextlib

import hashlib

import json

import math

import os

import re

import select

import subprocess

import sys

import tempfile

import time

from dataclasses import dataclass, field, replace

from pathlib import Path

from typing import Optional, Protocol, Sequence



from .primitives import grasp_proposal_hand_pose

from .schema import EpisodeRecord, GraspProposal





# --------------------------------------------------------------------------- #

# result + interface                                                          #

# --------------------------------------------------------------------------- #

@dataclass

class GeoCheck:

    """Outcome of one offline geometry check. ``ok=None`` means 'cannot determine

    with this backend' -> the gate reports PENDING_IMPL (not a failure)."""



    ok: Optional[bool]

    detail: str = ""



    @classmethod

    def pending(cls, detail: str) -> "GeoCheck":

        return cls(None, detail)





class GeometryValidator(Protocol):

    def grasp_reachable(self, rec: EpisodeRecord) -> GeoCheck: ...

    def rewind_path_collision_free(self, rec: EpisodeRecord) -> GeoCheck: ...

    def perception_repeatable(self, rec: EpisodeRecord) -> GeoCheck: ...





# --------------------------------------------------------------------------- #

# robot kinematic constants (FR3)                                             #

# --------------------------------------------------------------------------- #

@dataclass

class FR3Kinematics:

    """Coarse kinematic envelope for Franka Research 3.



    NOTE: joint limits and reach radius below are the commonly-cited Panda/FR3

    values and MUST be confirmed against the deployed robot's URDF.

    Base frame assumed at the world origin; override ``base_xyz`` if your

    hand-eye/world frame differs."""



    q_min: list[float] = field(default_factory=lambda: [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])

    q_max: list[float] = field(default_factory=lambda: [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    reach_max_m: float = 0.855       # Panda/FR3 max reach (sphere radius from base)

    reach_min_m: float = 0.05        # inside this, the arm self-collides / singular

    z_min_m: float = 0.0             # below table top is unreachable/invalid

    base_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])



    def joint_limits_ok(self, q: Sequence[float]) -> tuple[bool, str]:

        if len(q) != len(self.q_min):

            return False, f"q has length {len(q)}, expected {len(self.q_min)}"

        for i, v in enumerate(q):

            if not (self.q_min[i] <= v <= self.q_max[i]):

                return False, f"joint {i} = {v:.3f} out of limits [{self.q_min[i]:.3f}, {self.q_max[i]:.3f}]"

        return True, "all joints within limits"



    def position_in_envelope(self, pos: Sequence[float]) -> tuple[bool, str]:

        dx = [pos[i] - self.base_xyz[i] for i in range(3)]

        r = math.sqrt(sum(c * c for c in dx))

        if pos[2] < self.z_min_m:

            return False, f"z {pos[2]:.3f} below z_min {self.z_min_m:.3f}"

        if r > self.reach_max_m:

            return False, f"radius {r:.3f} m exceeds max reach {self.reach_max_m:.3f} m"

        if r < self.reach_min_m:

            return False, f"radius {r:.3f} m inside min reach {self.reach_min_m:.3f} m"

        return True, f"within reach envelope (r={r:.3f} m)"





def _nearest_q_at(rec: EpisodeRecord, ts: Optional[float]) -> Optional[list[float]]:

    if ts is None or not rec.trajectory:

        return None

    s = min(rec.trajectory, key=lambda x: abs(x.ts - ts))

    return list(s.q)





# --------------------------------------------------------------------------- #

# zero-dependency validator (coarse but real)                                 #

# --------------------------------------------------------------------------- #

class KinematicBoundsValidator:

    """Dependency-free geometry validator: reach-envelope + joint-limit bounds.



    This is a *necessary-condition* gate, not full IK/collision. It can FAIL an

    obviously-bad grasp/target and otherwise defers

    the precise verdict (collision / IK convergence / perception) to a heavier

    backend by reporting ``ok=None`` (pending)."""



    def __init__(self, kin: Optional[FR3Kinematics] = None) -> None:

        self.kin = kin or FR3Kinematics()



    def grasp_reachable(self, rec: EpisodeRecord) -> GeoCheck:

        grasp = rec.recovery.perceived_grasp if rec.recovery else None

        if grasp is None:

            return GeoCheck.pending("no perceived grasp")

        ok, detail = self.kin.position_in_envelope(grasp.position)

        if not ok:

            return GeoCheck(False, f"grasp unreachable: {detail}")

        # Within the envelope is a necessary condition; full IK is the upgrade.

        return GeoCheck(True, f"grasp within reach envelope ({detail}); full IK pending curobo")



    def rewind_path_collision_free(self, rec: EpisodeRecord) -> GeoCheck:

        q_goal = rec.recovery.rewind_target_q if rec.recovery else None

        if q_goal is None:

            return GeoCheck.pending("no rewind target joint state")

        ok, detail = self.kin.joint_limits_ok(q_goal)

        if not ok:

            return GeoCheck(False, f"rewind target out of joint limits: {detail}")

        q_start = _nearest_q_at(rec, rec.first_fail_ts)

        if q_start is not None:

            ok2, detail2 = self.kin.joint_limits_ok(q_start)

            if not ok2:

                return GeoCheck(False, f"failed-state q out of joint limits: {detail2}")

        # Joint limits OK, but collision along the path is NOT evaluated here.

        return GeoCheck.pending("joint limits ok; collision-free path needs curobo")



    def perception_repeatable(self, rec: EpisodeRecord) -> GeoCheck:

        return GeoCheck.pending("needs a perception backend to re-run on recorded RGB-D")





def infer_object_geometry(

    mask,

    depth_m=None,

    *,

    intrinsics=None,

    debug_out=None,

) -> str:

    """Classify geometry from calibrated physical extents, with a pixel-based fallback."""

    try:

        import numpy as np  # type: ignore

    except Exception:  # noqa: BLE001

        return "unknown"



    m = np.asarray(mask)

    if m.dtype != bool:

        m = m > 0

    if m.ndim != 2 or not np.any(m):

        return "unknown"



    rows = np.where(np.any(m, axis=1))[0]

    cols = np.where(np.any(m, axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:

        return "unknown"



    bbox_h = float(rows[-1] - rows[0] + 1)

    bbox_w = float(cols[-1] - cols[0] + 1)

    aspect = bbox_h / max(bbox_w, 1.0)



    # ---- calibrated world-frame path (preferred) ------------------------- #

    world_pts = _masked_points_world(np, m, depth_m, intrinsics)

    if world_pts is not None and len(world_pts) >= 32:

        z = world_pts[:, 2]

        # Height must be measured against the SUPPORT PLANE, not within the

        # mask: from a near-overhead camera a can's mask is mostly its top

        # face, so the masked z-spread is ~0 and every upright object would

        # read as low_object. Estimate the table as a low percentile of the

        # FULL scene's world z (dominant plane on a tabletop rig) and take

        # object top minus table; keep the masked spread as a lower bound

        # (side-view rigs, missing scene depth).

        height = float(np.percentile(z, 98) - np.percentile(z, 2))

        table_z = _scene_table_z(np, depth_m, intrinsics)

        if table_z is not None:

            height = max(height, float(np.percentile(z, 98)) - table_z)

        if isinstance(debug_out, dict):

            debug_out.update({

                "n_pts": int(len(world_pts)),

                "z_top": round(float(np.percentile(z, 98)), 4),

                "z_bot": round(float(np.percentile(z, 2)), 4),

                "table_z": None if table_z is None else round(table_z, 4),

                "height": round(height, 4),

            })

        fx_ = float(np.percentile(world_pts[:, 0], 98) - np.percentile(world_pts[:, 0], 2))

        fy_ = float(np.percentile(world_pts[:, 1], 98) - np.percentile(world_pts[:, 1], 2))

        fp_max = max(fx_, fy_)

        fp_ratio = (min(fx_, fy_) / fp_max) if fp_max > 1e-6 else 0.0

        if isinstance(debug_out, dict):

            debug_out.update({"fp_max": round(fp_max, 4), "fp_ratio": round(fp_ratio, 3)})

        # Physical thresholds (metres), category-level only:

        #   < 4.5 cm tall            -> low_object (flat boxes / pucks)

        #   >= 12.5 cm tall          -> tall upright: wide footprint (>= 6.2 cm)

        #                               reads as a carton (tall_box); a narrower

        #                               footprint is a bottle silhouette, which

        #                               routes through the clutter/bottle bucket

        #   in between               -> short upright: near-round footprint

        #                               (minor/major >= 0.72) is a can (cylinder),

        #                               else default to the clutter bucket

        # A mask with no physical height is the TABLE/BACKGROUND, not a low

        # object: grounding missed. Returning low_object here poisons routing



        # mask classified a tall salad-dressing bottle as low_object). Fail

        # open to unknown so the canonical/diagnosis fallback applies.

        if height < 0.006:

            return "unknown"

        if height < 0.045:

            return "low_object"

        if height >= 0.125:

            return "tall_box" if fp_max >= 0.062 else "clutter"

        return "cylinder" if fp_ratio >= 0.72 else "clutter"



    # ---- uncalibrated fallback: pixel-aspect heuristic ------------------- #



    z_extent_m: float | None = None

    if depth_m is not None:

        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        if depth.shape == m.shape:

            zs = depth[m & np.isfinite(depth) & (depth > 0.0)]

            if zs.size >= 8:

                z_extent_m = float(np.percentile(zs, 95) - np.percentile(zs, 5))



    if aspect >= 2.2 or (aspect >= 1.8 and (z_extent_m is None or z_extent_m >= 0.07)):

        return "tall_box"

    # Wide shallow boxes (cream cheese, pudding cups): low aspect or thin depth spread.

    if aspect < 0.90 or (z_extent_m is not None and z_extent_m < 0.030):

        return "low_object"

    if 1.25 <= aspect < 2.2:

        return "cylinder"

    if aspect >= 0.85:

        return "clutter"

    return "unknown"





def _record_zspan(detector, pts_world, np):

    """Record the perceived object's world z-span on the detector.



    ``last_object_zspan_world = [z_p2, z_p98]`` of the masked object cloud --

    a calibrated, object-agnostic height signal consumed by the ``depth_frac``

    grasp-depth primitive (grasp at a fraction of the object's own height).

    Fails silent: consumers treat a missing span as "no adjustment".

    """

    try:

        if pts_world is None or len(pts_world) < 16:

            detector.last_object_zspan_world = None

            return

        z = np.asarray(pts_world)[:, 2]

        detector.last_object_zspan_world = [

            float(np.percentile(z, 2)), float(np.percentile(z, 98))

        ]

    except Exception:  # noqa: BLE001

        detector.last_object_zspan_world = None





def _masked_points_world(np, mask, depth_m, intrinsics):

    """Back-project masked depth pixels to world-frame points ``(N,3)``.



    Returns ``None`` when depth/calibration is missing or unusable, so

    ``infer_object_geometry`` falls back to the pixel heuristic. Subsampled for

    speed; category inference only needs coarse extents."""

    if depth_m is None or not isinstance(intrinsics, dict):

        return None

    try:

        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        if depth.shape != mask.shape:

            return None

        if "K" in intrinsics:

            k = intrinsics["K"]

        elif "intrinsics" in intrinsics:

            k = intrinsics["intrinsics"]

        elif {"fx", "fy", "ppx", "ppy"}.issubset(intrinsics):

            k = [

                [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],

                [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],

                [0.0, 0.0, 1.0],

            ]

        else:

            return None

        K = np.asarray(k, dtype=np.float32).reshape(3, 3)

        wfc = intrinsics.get("world_from_camera")

        if wfc is None:

            wfc = intrinsics.get("extrinsics")

        if wfc is None:

            return None

        world_from_camera = np.asarray(wfc, dtype=np.float32).reshape(4, 4)

        vs, us = np.where(mask)

        if len(us) == 0:

            return None

        if len(us) > 4000:

            idx = np.random.default_rng(0).choice(len(us), 4000, replace=False)

            us, vs = us[idx], vs[idx]

        zs = depth[vs, us]

        ok = np.isfinite(zs) & (zs > 0.0)

        us, vs, zs = us[ok], vs[ok], zs[ok]

        if len(us) < 8:

            return None

        xs = (us.astype(np.float32) - K[0, 2]) * zs / K[0, 0]

        ys = (vs.astype(np.float32) - K[1, 2]) * zs / K[1, 1]

        pts_cam = np.stack([xs, ys, zs, np.ones_like(zs)], axis=1)

        pts_world = (world_from_camera @ pts_cam.T).T[:, :3]

        return pts_world

    except Exception:  # noqa: BLE001 - inference must fail open to the fallback

        return None





def _scene_table_z(np, depth_m, intrinsics):

    """Support-plane world z: a low percentile of the full-frame scene points.



    Tabletop assumption -- the dominant plane under/around the objects is the

    support surface. Returns ``None`` when scene back-projection is unusable."""

    try:

        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        full = np.ones(depth.shape, dtype=bool)

        pts = _masked_points_world(np, full, depth, intrinsics)

        if pts is None or len(pts) < 64:

            return None

        return float(np.percentile(pts[:, 2], 15))

    except Exception:  # noqa: BLE001

        return None





def infer_geometry_from_backend(backend, object_hint, target_px=None):

    """Perception-only geometry category for live strategy routing.



    Snapshot the backend's camera, ground the target with the SAME cascade the

    grasp pipeline uses (a bare point-prompt at a judge/GT anchor segments the

    table -- see object_bucket_0707), then classify with the calibrated

    ``infer_object_geometry``. Returns ``"unknown"`` on ANY failure so callers

    fall back to the canonical strategy. No object-name priors.

    """

    try:

        import numpy as np  # type: ignore

        import cv2  # type: ignore



        fetch = getattr(backend, "_fetch_perception_inputs", None)

        if not callable(fetch):

            return "unknown"

        inputs = fetch()

        if inputs is None:

            return "unknown"

        rgb_jpeg, depth_m, intr = inputs

        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

        if rgb is None:

            return "unknown"

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        det = getattr(backend, "detector", None)

        fb = getattr(det, "fallback", None) or det

        mask = None

        if target_px is not None and hasattr(fb, "_disambiguate_mask_px"):

            mask = fb._disambiguate_mask_px(

                np, rgb=rgb, depth_m=depth_m,

                object_hint=object_hint, target_px=target_px, intrinsics=intr,

            )

        if mask is None:

            seg = getattr(det, "segmenter", None) or getattr(fb, "segmenter", None)

            if seg is None:

                return "unknown"

            mask = seg.segment(rgb, prompt=object_hint)

        if mask is None:

            return "unknown"

        return infer_object_geometry(mask, depth_m, intrinsics=intr)

    except Exception:  # noqa: BLE001 - routing must fail open to canonical

        return "unknown"





def scene_pointcloud_world(depth_m, intrinsics, *, subsample: int = 6, max_points: int = 40000):

    """Back-project a FULL-FRAME depth image into WORLD-frame points ``(N,3)``.



    Unlike the detectors' *masked* clouds, this returns the whole observed scene

    (table + all objects) so the motion planner can treat clutter as collision

    geometry at recovery time (see CuroboMotionPlanner.set_scene_pointcloud). The

    target object is carved out later, in the base frame, by a radius filter.



    ``intrinsics`` must carry ``K``/``fx..`` and ``world_from_camera``/``extrinsics``.

    Returns ``None`` if inputs are unusable (fails open -> planner keeps table-only).

    """

    try:

        import numpy as np  # type: ignore

    except Exception:  # noqa: BLE001

        return None

    try:

        if "K" in intrinsics:

            k = np.asarray(intrinsics["K"], dtype=np.float32).reshape(3, 3)

        elif "intrinsics" in intrinsics:

            k = np.asarray(intrinsics["intrinsics"], dtype=np.float32).reshape(3, 3)

        elif {"fx", "fy", "ppx", "ppy"}.issubset(intrinsics):

            k = np.array([

                [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],

                [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],

                [0.0, 0.0, 1.0],

            ], dtype=np.float32)

        else:

            return None

        mat = intrinsics.get("world_from_camera") or intrinsics.get("extrinsics")

        if mat is None:

            return None

        world_from_camera = np.asarray(mat, dtype=np.float32).reshape(4, 4)



        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        if depth.ndim != 2:

            return None

        s = max(1, int(subsample))

        depth = depth[::s, ::s]

        h, w = depth.shape

        ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0.0))

        if len(xs) == 0:

            return None

        z = depth[ys, xs]

        # xs/ys are in the subsampled grid; scale back to original pixel coords.

        xpix = xs.astype(np.float32) * s

        ypix = ys.astype(np.float32) * s

        x = (xpix - k[0, 2]) * z / k[0, 0]

        y = (ypix - k[1, 2]) * z / k[1, 1]

        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)

        pts_world = (world_from_camera @ pts_cam)[:3].T.astype(np.float32)

        if max_points > 0 and len(pts_world) > max_points:

            rng = np.random.default_rng(0)

            pts_world = pts_world[rng.choice(len(pts_world), size=int(max_points), replace=False)]

        return pts_world

    except Exception:  # noqa: BLE001

        return None





# --------------------------------------------------------------------------- #

# curobo-backed motion planner (runtime) + validator (offline)                #

# --------------------------------------------------------------------------- #

@dataclass

class CuroboConfig:

    """Paths/params for curobo. ``robot_config`` is a curobo robot YAML (or the

    bundled ``franka.yml``); ``world_config`` is an optional static scene model

    (a scene YAML under curobo's scene configs). Use the paths supplied by the experiment configuration."""



    robot_config: str = "franka.yml"

    world_config: Optional[object] = None   # cuRobo scene: path/name (str) or dict

    interpolation_dt: float = 0.02          # kept for reference; set in the trajopt yml

    collision_activation_distance_m: float = 0.01

    num_seeds: int = 4

    # cap-x-style grasp execution uses a goalset-sized cuRobo warmup and
    # sufficient IK seeds for the candidate poses.

    num_ik_seeds: int = 128

    max_goalset: int = 100

    # RNG seed for cuRobo's IK/trajopt seed sampling. Fixed so a given grasp/scene

    # replays to the same plan run-to-run (otherwise the eval is irreproducible:

    # identical perception can yield different reachability/trajectories). Set at

    # planner construction before warmup.

    seed: int = 123

    # Optional table/ground collision plane so planned paths avoid the table even

    # though the bundled scene is empty. When ``world_config`` is unset and

    # ``table_top_z`` is not None, a ground cuboid is synthesised with its TOP

    # surface at world z == ``table_top_z`` (the robot base/world frame; the FR3

    # workspace table top is ~z=0). Ignored if ``world_config`` is given.

    table_top_z: Optional[float] = None

    table_extent_m: float = 4.0

    table_thickness_m: float = 0.2

    # Robot base pose in the WORLD frame that perception/records use. cuRobo's

    # robot config puts the base at the origin, so when the perception "world"

    # frame is NOT the robot base (e.g. LIBERO sim: robosuite world origin is

    # ~0.6 m in front of the Panda base) every Cartesian goal must be expressed

    # relative to the base before planning -- otherwise the goal lands in the

    # wrong place / is reported unreachable. On the real rig the calibrated world

    # IS the base, so the default identity makes this a no-op. CarryOn keeps all

    # of perception/disambiguation/records in the world frame; only the cuRobo

    # boundary (plan_to_pose / FK out) applies this transform.

    robot_base_xyz_m: tuple = (0.0, 0.0, 0.0)

    robot_base_quat_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)

    # Obstacle-cache headroom so a runtime SCENE mesh (from the recovery-time depth

    # point cloud) can be swapped in via ``update_world`` on top of the static

    # table. See CuroboMotionPlanner.set_scene_pointcloud.

    collision_cache_cuboid: int = 4

    collision_cache_mesh: int = 2





class CuroboMotionPlanner:

    """Collision-free planner using curobo.motion_planner.MotionPlanner.

    Imports cuRobo and torch lazily. plan_cspace handles joint goals and plan_pose
    handles tool-frame pose goals. Successful results provide an interpolated
    JointState trajectory through get_interpolated_plan()."""



    def __init__(self, config: Optional[CuroboConfig] = None) -> None:

        self.config = config or CuroboConfig()

        self.last_error: Optional[str] = None

        # Lazy, guarded imports so module import never pulls torch/CUDA.

        import torch  # type: ignore

        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # type: ignore

        from curobo.types import GoalToolPose, JointState  # type: ignore



        self._torch = torch

        self._JointState = JointState

        self._GoalToolPose = GoalToolPose



        # Determinism: cuRobo's IK/trajopt draw many random seeds (num_ik_seeds,

        # num_trajopt_seeds). Without a fixed RNG the same grasp/scene can plan to

        # a different solution -- or none -- run-to-run, so a fixed-perception eval

        # is not reproducible. Seed once BEFORE warmup so an identical episode call

        # sequence replays deterministically.

        try:

            seed = int(getattr(self.config, "seed", 123))

            torch.manual_seed(seed)

            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True

            torch.backends.cudnn.benchmark = False

        except Exception:  # noqa: BLE001 - never let determinism setup break planning

            pass



        scene_model = self.config.world_config

        if scene_model is None and self.config.table_top_z is not None:

            scene_model = self._table_scene(

                self.config.table_top_z,

                self.config.table_extent_m,

                self.config.table_thickness_m,

            )

        # Preallocate an obstacle cache big enough to swap in a runtime SCENE mesh

        # (built from the recovery-time depth point cloud) on top of the static

        # table. Without this, ``update_world`` cannot add a mesh obstacle -- the

        # collision checker is sized to the initial (table-only) scene. This is

        # what lets the goalset planner reason about clutter it would otherwise

        # descend straight into. See ``set_scene_pointcloud``.

        collision_cache = {

            "cuboid": max(4, int(self.config.collision_cache_cuboid)),

            "mesh": max(2, int(self.config.collision_cache_mesh)),

        }

        cfg = MotionPlannerCfg.create(

            robot=self.config.robot_config,

            scene_model=scene_model,

            collision_cache=collision_cache,

            num_ik_seeds=self.config.num_ik_seeds,

            num_trajopt_seeds=self.config.num_seeds,

            optimizer_collision_activation_distance=self.config.collision_activation_distance_m,

            max_goalset=self.config.max_goalset,

        )

        self._planner = MotionPlanner(cfg)

        self._planner.warmup()

        self._device = self._planner.default_joint_state.position.device

        self._init_base_transform()

        # Remember the static table so scene updates can re-assert it alongside the

        # runtime mesh (update_world REPLACES the whole collision model).

        self._table_cfg = (

            (self.config.table_top_z, self.config.table_extent_m, self.config.table_thickness_m)

            if (self.config.world_config is None and self.config.table_top_z is not None)

            else None

        )

        self._scene_supported = True



    def _init_base_transform(self) -> None:

        """Precompute the world<->robot-base transform (see CuroboConfig).



        ``world_from_base`` = (R_wb, t_wb): the base pose expressed in the

        perception/world frame. cuRobo plans in the base frame (origin at the

        base), so ``plan_to_pose`` maps world->base on input and the FK helpers

        map base->world on output. Identity (the real-rig default) short-circuits

        to a no-op via :attr:`_has_base_tf`."""

        import numpy as np  # local: module import stays torch/numpy-free



        t = np.asarray(self.config.robot_base_xyz_m, dtype=float).reshape(3)

        q = np.asarray(self.config.robot_base_quat_wxyz, dtype=float).reshape(4)

        q = q / (np.linalg.norm(q) or 1.0)

        w, x, y, z = (float(v) for v in q)

        R = np.array([

            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],

            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],

            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],

        ], dtype=float)

        self._np = np

        self._R_wb = R

        self._t_wb = t

        self._q_wb = [w, x, y, z]

        self._has_base_tf = bool(

            np.linalg.norm(t) > 1e-9 or abs(1.0 - w) > 1e-9

            or abs(x) > 1e-9 or abs(y) > 1e-9 or abs(z) > 1e-9

        )



    @staticmethod

    def _quat_mul(a, b):

        aw, ax, ay, az = a

        bw, bx, by, bz = b

        return [

            aw * bw - ax * bx - ay * by - az * bz,

            aw * bx + ax * bw + ay * bz - az * by,

            aw * by - ax * bz + ay * bw + az * bx,

            aw * bz + ax * by - ay * bx + az * bw,

        ]



    def _world_to_base(self, position, quaternion_wxyz):

        """Map a world-frame tool pose into the cuRobo base frame."""

        if not self._has_base_tf:

            return list(position), list(quaternion_wxyz)

        np = self._np

        p = self._R_wb.T @ (np.asarray(position, dtype=float).reshape(3) - self._t_wb)

        q_conj = [self._q_wb[0], -self._q_wb[1], -self._q_wb[2], -self._q_wb[3]]

        q = self._quat_mul(q_conj, list(quaternion_wxyz))

        return [float(v) for v in p], [float(v) for v in q]



    def _base_to_world(self, position, quaternion_wxyz):

        """Map a cuRobo base-frame tool pose back into the world frame."""

        if not self._has_base_tf:

            return list(position), list(quaternion_wxyz)

        np = self._np

        p = self._R_wb @ np.asarray(position, dtype=float).reshape(3) + self._t_wb

        q = self._quat_mul(self._q_wb, list(quaternion_wxyz))

        return [float(v) for v in p], [float(v) for v in q]



    @staticmethod

    def _table_scene(top_z: float, extent: float, thickness: float) -> dict:

        """cuRobo scene dict for a ground/table slab whose TOP is at world ``top_z``.

        Pose quaternion order is ``qx,qy,qz,qw`` (cuRobo convention)."""

        center_z = top_z - thickness / 2.0

        return {

            "cuboid": {

                "table": {

                    "dims": [extent, extent, thickness],

                    "pose": [0.0, 0.0, center_z, 0.0, 0.0, 0.0, 1.0],

                }

            }

        }



    def _world_points_to_base(self, points_world):

        """Vectorised world->base transform for an ``(N,3)`` point array."""

        np = self._np

        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)

        if not self._has_base_tf:

            return pts

        return (self._R_wb.T @ (pts - self._t_wb.reshape(1, 3)).T).T



    def set_scene_pointcloud(

        self,

        points_world,

        *,

        exclude_center_world=None,

        exclude_radius_m: float = 0.06,

        pitch_m: float = 0.02,

        table_clear_m: float = 0.005,

        workspace_half_m: float = 0.30,

        max_points: int = 6000,

    ) -> bool:  # pragma: no cover - needs CUDA

        """Install a runtime collision scene from world-frame depth points.

        Exclude points within exclude_radius_m of the grasp target and turn the remaining
        clutter into mesh obstacles. Return True when installed. On error, restore the
        table-only collision model."""

        if not getattr(self, "_scene_supported", False):

            return False

        np = self._np

        try:

            from curobo.scene import Scene, Cuboid, Mesh  # type: ignore

        except Exception as e:  # noqa: BLE001

            self.last_error = f"scene update import failed: {type(e).__name__}: {e}"

            return False

        try:

            pts_base = self._world_points_to_base(points_world)

            table_z_base = None

            if self._table_cfg is not None:

                top_z, _, _ = self._table_cfg

                # table_top_z is a world-z; map to base-z (translation only along z

                # for the identity-rot rigs we support). Clip points at/under the

                # table so the table cuboid (not stray depth) owns that surface.

                table_z_base = float((self._R_wb.T @ (np.array([0.0, 0.0, top_z]) - self._t_wb))[2])

            pts_base = self._filter_scene_points(

                np,

                pts_base,

                table_z_base=table_z_base,

                table_clear_m=table_clear_m,

                exclude_center_base=(

                    self._world_points_to_base(exclude_center_world)[0]

                    if exclude_center_world is not None

                    else None

                ),

                exclude_radius_m=exclude_radius_m,

                workspace_half_m=workspace_half_m,

                max_points=max_points,

            )

            cuboids = []

            if self._table_cfg is not None:

                top_z, extent, thickness = self._table_cfg

                cz = float((self._R_wb.T @ (np.array([0.0, 0.0, top_z - thickness / 2.0]) - self._t_wb))[2])

                cuboids = [Cuboid(name="table", dims=[extent, extent, thickness], pose=[0.0, 0.0, cz, 1.0, 0.0, 0.0, 0.0])]

            if pts_base is None or len(pts_base) == 0:

                # Nothing but the target in view -> just table.

                self._planner.update_world(Scene(cuboid=cuboids or None))

                return False

            mesh = Mesh.from_pointcloud(np.asarray(pts_base, dtype=np.float64), pitch=float(pitch_m), name="scene_clutter")

            self._planner.update_world(Scene(cuboid=cuboids or None, mesh=[mesh]))

            self.last_error = None

            return True

        except Exception as e:  # noqa: BLE001 - fail open to table-only

            self.last_error = f"set_scene_pointcloud failed: {type(e).__name__}: {e}"

            try:

                self.clear_scene()

            except Exception:  # noqa: BLE001

                pass

            return False



    def clear_scene(self) -> None:  # pragma: no cover - needs CUDA

        """Reset the collision model back to the static table (or empty)."""

        if not getattr(self, "_scene_supported", False):

            return

        from curobo.scene import Scene, Cuboid  # type: ignore



        np = self._np

        cuboids = None

        if self._table_cfg is not None:

            top_z, extent, thickness = self._table_cfg

            cz = float((self._R_wb.T @ (np.array([0.0, 0.0, top_z - thickness / 2.0]) - self._t_wb))[2])

            cuboids = [Cuboid(name="table", dims=[extent, extent, thickness], pose=[0.0, 0.0, cz, 1.0, 0.0, 0.0, 0.0])]

        self._planner.update_world(Scene(cuboid=cuboids))



    @staticmethod

    def _filter_scene_points(

        np,

        pts_base,

        *,

        table_z_base=None,

        table_clear_m: float = 0.005,

        z_ceiling_m: float = 0.6,

        exclude_center_base=None,

        exclude_radius_m: float = 0.06,

        workspace_half_m: float = 0.0,

        max_points: int = 6000,

    ):

        """Pure numpy filter for the runtime clutter cloud (base frame).



        - drops points at/under the table (owned by the table cuboid),

        - drops a tall ceiling of stray far points,

        - (optionally) keeps only a WORKSPACE BOX around the target so the robot

          arm (offset toward the base) and far background never enter the mesh --

          the over-constraint that made the whole goalset infeasible,

        - carves out a sphere/column around the grasp target so the gripper can

          still descend onto and contact it,

        - subsamples to ``max_points`` for a bounded-size mesh.

        Kept static + numpy-only so it is unit-testable without CUDA."""

        pts = np.asarray(pts_base, dtype=float).reshape(-1, 3)

        if len(pts) == 0:

            return pts

        keep = np.isfinite(pts).all(axis=1)

        if table_z_base is not None:

            keep &= pts[:, 2] > (float(table_z_base) + float(table_clear_m))

            keep &= pts[:, 2] < (float(table_z_base) + float(z_ceiling_m))

        if exclude_center_base is not None:

            c = np.asarray(exclude_center_base, dtype=float).reshape(3)

            if workspace_half_m and workspace_half_m > 0.0:

                keep &= np.abs(pts[:, 0] - c[0]) < float(workspace_half_m)

                keep &= np.abs(pts[:, 1] - c[1]) < float(workspace_half_m)

            d2 = ((pts[:, :2] - c[:2].reshape(1, 2)) ** 2).sum(axis=1)

            keep &= d2 > (float(exclude_radius_m) ** 2)

        pts = pts[keep]

        if max_points > 0 and len(pts) > max_points:

            rng = np.random.default_rng(0)

            pts = pts[rng.choice(len(pts), size=int(max_points), replace=False)]

        return pts



    def _js(self, q):

        t = self._torch.tensor([list(q)], device=self._device, dtype=self._torch.float32)

        return self._JointState.from_position(t, joint_names=self._planner.joint_names)



    def _goal_pose(self, position, quaternion_wxyz):

        # GoalToolPose expects 5D tensors [batch, horizon, num_links, num_goalset, 3/4].

        torch = self._torch

        pos = torch.tensor([[[[list(position)]]]], device=self._device, dtype=torch.float32)

        quat = torch.tensor([[[[list(quaternion_wxyz)]]]], device=self._device, dtype=torch.float32)

        return self._GoalToolPose(

            tool_frames=self._planner.tool_frames,

            position=pos,

            quaternion=quat,

        )



    def _goalset_pose(self, poses):  # pragma: no cover - needs CUDA

        """Build a cuRobo GoalToolPose from ordered world-frame poses.

        Each pose is (position, quaternion_wxyz); the tensor shape is
        [batch=1, horizon=1, link=1, goalset=N, xyz/quat]."""

        torch = self._torch

        positions = [list(p) for p, _q in poses]

        quats = [list(q) for _p, q in poses]

        pos = torch.tensor([[positions]], device=self._device, dtype=torch.float32).unsqueeze(1)

        quat = torch.tensor([[quats]], device=self._device, dtype=torch.float32).unsqueeze(1)

        return self._GoalToolPose(

            tool_frames=self._planner.tool_frames,

            position=pos,

            quaternion=quat,

        )



    def _jointstate_to_waypoints(self, traj) -> Optional[list[list[float]]]:  # pragma: no cover - needs CUDA

        if traj is None or getattr(traj, "position", None) is None:

            return None

        try:

            traj = self._planner.kinematics.get_active_js(traj)

        except Exception:  # noqa: BLE001 - fall back to raw positions

            pass

        pos = traj.position

        while pos.ndim > 2:

            pos = pos[0]

        return [list(map(float, q)) for q in pos.detach().cpu().numpy()]



    def _result_to_waypoints(self, result) -> Optional[list[list[float]]]:

        if result is None or result.success is None or not bool(result.success.any()):

            self.last_error = (

                "planner returned no successful trajectory"

                if result is None

                else f"planner status={getattr(result, 'status', None)}"

            )

            return None

        self.last_error = None

        traj = result.get_interpolated_plan()

        # The interpolated trajectory carries the full kinematic-tree dof (arm +

        # gripper fingers); ``get_active_js`` trims it to the actuated arm joints

        # so the waypoints match CarryOn's 7-dof arm command contract.

        return self._jointstate_to_waypoints(traj)



    def plan_to_q(self, q_start, q_goal):  # pragma: no cover - needs CUDA

        self.last_error = None

        res = self._planner.plan_cspace(self._js(q_goal), self._js(q_start))

        return self._result_to_waypoints(res)



    def plan_to_pose(self, q_start, position, quaternion_wxyz):  # pragma: no cover - needs CUDA

        self.last_error = None

        # ``position``/``quaternion`` are WORLD-frame (perception convention); cuRobo

        # plans in the base frame, so map the goal into the base frame first.

        bpos, bquat = self._world_to_base(position, quaternion_wxyz)

        res = self._planner.plan_pose(self._goal_pose(bpos, bquat), self._js(q_start))

        return self._result_to_waypoints(res)



    @staticmethod

    def _fingertip_to_hand_pose(grasp: GraspProposal, tcp_offset_m: float):

        """Convert a fingertip/contact-centre grasp to the panda_hand/tool pose.

        As in cap-x's _grasp_pose_fingertip_to_hand, the conversion is
        hand = fingertip - tcp * R(quat)[:, 2]."""

        return grasp_proposal_hand_pose(grasp, tcp_offset_m)



    def plan_grasp_goalset(

        self,

        q_start,

        grasps: Sequence[GraspProposal],

        *,

        tcp_offset_m: float = 0.1034,

        pre_grasp_dis: float = 0.10,

        plan_grasp_to_lift: bool = False,

    ):  # pragma: no cover - needs CUDA

        """cap-x-style grasp planning over a top-K goalset.



        Uses the current cuRobo refactor's native ``MotionPlanner.plan_grasp``:

        1. select a reachable goal from the full top-K goalset,

        2. plan to an approach pose offset along tool local -z,

        3. plan a linear approach from standoff to grasp (and optional lift).



        Returns a dict with ``approach_path``, ``grasp_path``, ``lift_path``,

        ``goalset_index`` and ``status``. Returns ``None`` if cuRobo fails.

        """

        if not grasps:

            return None

        # Hand poses come out in the WORLD frame; map each into the base frame

        # cuRobo plans in (no-op on the real rig where world == base).

        poses = [

            self._world_to_base(*self._fingertip_to_hand_pose(g, tcp_offset_m))

            for g in grasps

        ]

        goalset = self._goalset_pose(poses)

        res = self._planner.plan_grasp(

            grasp_poses=goalset,

            current_state=self._js(q_start),

            grasp_approach_axis="z",

            grasp_approach_offset=-float(pre_grasp_dis),

            grasp_approach_in_tool_frame=True,

            grasp_lift_axis="z",

            grasp_lift_offset=-float(pre_grasp_dis),

            grasp_lift_in_tool_frame=True,

            plan_approach_to_grasp=True,

            plan_grasp_to_lift=bool(plan_grasp_to_lift),

        )

        ok = res is not None and res.success is not None and bool(res.success.any())

        if not ok:

            return {

                "ok": False,

                "status": getattr(res, "status", "plan_grasp returned None") if res is not None else "plan_grasp returned None",

                "goalset_index": None,

                "approach_path": None,

                "grasp_path": None,

                "lift_path": None,

            }

        goal_idx = None

        if getattr(res, "goalset_index", None) is not None:

            goal_idx = int(res.goalset_index.view(-1)[0].item())

        return {

            "ok": True,

            "status": getattr(res, "status", None),

            "goalset_index": goal_idx,

            "approach_path": self._jointstate_to_waypoints(getattr(res, "approach_interpolated_trajectory", None)),

            "grasp_path": self._jointstate_to_waypoints(getattr(res, "grasp_interpolated_trajectory", None)),

            "lift_path": self._jointstate_to_waypoints(getattr(res, "lift_interpolated_trajectory", None)),

        }



    def _kin_state(self, q):  # pragma: no cover - needs CUDA

        return self._planner.kinematics.compute_kinematics(self._js(q))



    def ee_position(self, q):  # pragma: no cover - needs CUDA

        """Best-effort forward kinematics: world-frame tool position ``[x,y,z]`` for

        a joint vector, or ``None`` if the cuRobo kinematics API differs."""

        try:

            pos = self._kin_state(q).tool_poses.position

            arr = pos.detach().cpu().numpy().reshape(-1)

            world_pos, _ = self._base_to_world(

                [float(arr[0]), float(arr[1]), float(arr[2])], [1.0, 0.0, 0.0, 0.0]

            )

            return world_pos

        except Exception:  # noqa: BLE001

            return None



    def ee_pose(self, q):  # pragma: no cover - needs CUDA

        """Forward kinematics returning ``(position[x,y,z], quaternion_wxyz)`` for the

        tool frame, in the **same convention** :meth:`plan_to_pose` consumes (cuRobo

        ``Pose`` is wxyz). Used by ``lift`` to hold the current orientation exactly --

        deriving it from the planner's FK avoids any bridge quaternion-convention

        mismatch (which would otherwise send a wrong target orientation). ``None`` on

        an API mismatch."""

        try:

            tp = self._kin_state(q).tool_poses

            pos = tp.position.detach().cpu().numpy().reshape(-1)

            quat = tp.quaternion.detach().cpu().numpy().reshape(-1)

            world_pos, world_quat = self._base_to_world(

                [float(pos[0]), float(pos[1]), float(pos[2])],

                [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])],

            )

            return (world_pos, world_quat)

        except Exception:  # noqa: BLE001

            return None



    def lowest_body_z(self, q):  # pragma: no cover - needs CUDA

        """Best-effort world-frame height of the *lowest point of the whole arm*:

        ``min(z - r)`` over cuRobo's robot collision spheres for joint vector ``q``.



        This is the table-strike guard's core signal -- it catches elbow / wrist /

        link collisions with the table, not just the tool tip. Returns ``None`` if

        the sphere model is unavailable."""

        try:

            spheres = self._kin_state(q).robot_spheres  # [..., N, 4] = [x,y,z,r]

            arr = spheres.detach().cpu().numpy().reshape(-1, 4)

            return float((arr[:, 2] - arr[:, 3]).min())

        except Exception:  # noqa: BLE001

            return None





class CuroboGeometryValidator:

    """Reuse the runtime motion planner for recorded geometry checks."""



    def __init__(self, planner: CuroboMotionPlanner) -> None:

        self.planner = planner

        self.kin = FR3Kinematics()



    def grasp_reachable(self, rec: EpisodeRecord) -> GeoCheck:  # pragma: no cover - needs CUDA

        grasp = rec.recovery.perceived_grasp if rec.recovery else None

        if grasp is None:

            return GeoCheck.pending("no perceived grasp")

        try:

            q_seed = _nearest_q_at(rec, rec.first_fail_ts) or self.kin.q_min

            wp = self.planner.plan_to_pose(q_seed, grasp.position, grasp.quaternion)

        except Exception as e:  # noqa: BLE001

            return GeoCheck(False, f"curobo plan_to_pose raised: {type(e).__name__}: {e}")

        if not wp:

            return GeoCheck(False, "no IK / motion solution to grasp pose")

        return GeoCheck(True, f"grasp reachable ({len(wp)} waypoints)")



    def rewind_path_collision_free(self, rec: EpisodeRecord) -> GeoCheck:  # pragma: no cover - needs CUDA

        q_goal = rec.recovery.rewind_target_q if rec.recovery else None

        if q_goal is None:

            return GeoCheck.pending("no rewind target joint state")

        q_start = _nearest_q_at(rec, rec.first_fail_ts)

        if q_start is None:

            return GeoCheck.pending("no failed-state q in trajectory")

        try:

            wp = self.planner.plan_to_q(q_start, q_goal)

        except Exception as e:  # noqa: BLE001

            return GeoCheck(False, f"curobo plan_to_q raised: {type(e).__name__}: {e}")

        if not wp:

            return GeoCheck(False, "no collision-free path from failed state to rewind target")

        return GeoCheck(True, f"collision-free rewind path found ({len(wp)} waypoints)")



    def perception_repeatable(self, rec: EpisodeRecord) -> GeoCheck:

        return GeoCheck.pending("perception determinism needs a GraspDetector, not the planner")





# --------------------------------------------------------------------------- #

# SAM3 text-grounded segmentation (runtime) -> mask for the grasp target       #

# --------------------------------------------------------------------------- #

@dataclass

class Sam3Config:

    """Paths/params for the external SAM3 segmenter. See ENV doc §3."""



    model_path: str = field(default_factory=lambda: os.environ.get("CARRYON_SAM3_MODEL_PATH", ""))

    python_exe: str = field(default_factory=lambda: os.environ.get("CARRYON_SAM3_PYTHON", ""))

    runner_path: str = ""

    threshold: float = 0.5

    # Adaptive-cascade retry: a RELAXED text-grounding threshold used only when the

    # strict ``threshold`` pass returns no usable mask. Keeps everyday grounding at

    # high precision while recovering weak-but-correct detections (e.g. the small

    # LIBERO "ketchup" bottle scores ~0.30 and dips under 0.25 frame-to-frame).

    # < 0 disables the retry (single-pass at ``threshold``). The persistent server

    # is launched at min(threshold, retry_threshold) so a single loaded model can

    # serve both passes via per-request score gating.

    retry_threshold: float = -1.0

    mask_threshold: float = 0.5

    seed: int = 123

    timeout_s: float = 120.0

    verbose: bool = False

    # Persistent server: load the model ONCE in a long-lived subprocess and feed it

    # one request per call (stdin/stdout protocol). Removes the per-call torch-import

    # + from_pretrained cost (~6-10 s) that dominates every perceive_grasp. Falls

    # back to the one-shot subprocess if the server cannot be started.

    persistent: bool = True

    server_ready_timeout_s: float = 180.0

    # "auto" | "cuda" | "cpu". Use cpu when no spare GPU is available.

    device: str = "auto"

    # Physical GPU id(s) for this subprocess only (sets CUDA_VISIBLE_DEVICES).

    # Lets SAM3 run on card 1 while cuRobo holds card 0 in the parent process.

    cuda_visible_devices: str = ""

    # "native" (sam3 package: promptable point -> full object, cap-x aligned) or

    # "transformers" (HF Sam3Model: point approximated by a small box). Native

    # fixes the table-height sliver that the box approximation produced for

    # point-prompted grounding (Molmo / ground-truth anchor paths).

    backend: str = "native"





class Sam3Segmenter:

    """Run SAM3 (text -> mask) in its own conda env via a subprocess runner.



    The transformers/SAM3 stack lives outside CarryOn's process; ``segment``

    writes the RGB frame to a temp file, invokes :mod:`sam3_runner`, and returns a

    boolean target mask aligned to the input image size. Fails closed (``None``)

    and records :attr:`last_error` / :attr:`last_result` for debugging."""



    def __init__(self, config: Optional[Sam3Config] = None) -> None:

        self.config = config or Sam3Config()

        if not str(self.config.model_path).strip() or not str(self.config.python_exe).strip():

            raise FileNotFoundError("Configure CARRYON_SAM3_MODEL_PATH and CARRYON_SAM3_PYTHON or pass explicit resource paths")

        self.last_error: Optional[str] = None

        self.last_result: Optional[dict] = None

        self.last_timing: dict = {}

        self.last_grounding_via: Optional[str] = None

        self.last_grounding_via: Optional[str] = None

        self.model_path = Path(self.config.model_path)

        self.python_exe = Path(self.config.python_exe)

        self.runner_path = Path(

            self.config.runner_path or Path(__file__).with_name("sam3_runner.py")

        )

        missing = [

            str(p)

            for p in (self.model_path, self.python_exe, self.runner_path)

            if not p.exists()

        ]

        if missing:

            raise FileNotFoundError("missing SAM3 resource(s): " + ", ".join(missing))

        self._proc: Optional[subprocess.Popen] = None  # persistent server

        self._server_log = None                        # child stderr sink (file)

        self._server_failed = False                    # one-shot fallback latch



    def _subprocess_env(self) -> dict:

        """Child env for SAM3: optional isolated GPU via CUDA_VISIBLE_DEVICES."""

        env = os.environ.copy()

        cvd = (self.config.cuda_visible_devices or "").strip()

        if cvd:

            env["CUDA_VISIBLE_DEVICES"] = cvd

        return env



    def _fail(self, reason: str):

        self.last_error = reason

        if self.config.verbose:

            print(f"[Sam3Segmenter] {reason}")

        return None



    def _write_image(self, np, td: Path, rgb) -> tuple[dict, Optional[str]]:

        """Return (image_request, error). ``image_request`` is a dict

        ``{"image"|"image_npy": path}`` shared by the persistent and one-shot

        paths. ``rgb`` may be encoded bytes, an HxWx3 array, or a path."""

        if isinstance(rgb, (bytes, bytearray)):

            img_path = td / "image.jpg"

            img_path.write_bytes(bytes(rgb))

            return {"image": str(img_path)}, None

        if isinstance(rgb, (str, Path)):

            p = Path(rgb)

            if not p.exists():

                return {}, f"image path does not exist: {p}"

            return {"image": str(p)}, None

        try:

            arr = np.asarray(rgb)

        except Exception as e:  # noqa: BLE001

            return {}, f"unsupported rgb type {type(rgb).__name__}: {e}"

        if arr.ndim != 3 or arr.shape[2] != 3:

            return {}, f"rgb array must be HxWx3, got shape {arr.shape}"

        npy_path = td / "image.npy"

        np.save(npy_path, arr.astype(np.uint8))

        return {"image_npy": str(npy_path)}, None



    # ---- persistent server (load model once; one request per call) --------- #

    def _read_line(self, stream, timeout_s: float) -> Optional[str]:

        """Read one non-empty line from ``stream`` within ``timeout_s`` (select-

        based; Linux pipe). Returns None on timeout or EOF."""

        end = time.monotonic() + float(timeout_s)

        while True:

            remaining = end - time.monotonic()

            if remaining <= 0:

                return None

            try:

                ready, _, _ = select.select([stream], [], [], remaining)

            except Exception:  # noqa: BLE001

                return None

            if not ready:

                return None

            line = stream.readline()

            if line == "":

                return None  # EOF -> server died

            line = line.strip()

            if line:

                return line



    def _close_server_log(self) -> None:

        log, self._server_log = self._server_log, None

        if log is not None:

            with contextlib.suppress(Exception):

                log.close()

            with contextlib.suppress(Exception):

                os.unlink(log.name)



    def _server_log_tail(self, n: int = 400) -> str:

        log = getattr(self, "_server_log", None)

        if log is None:

            return ""

        with contextlib.suppress(Exception):

            return Path(log.name).read_text(encoding="utf-8", errors="replace")[-n:].strip()

        return ""



    def _stop_server(self) -> None:

        proc, self._proc = self._proc, None

        if proc is None:

            self._close_server_log()

            return

        with contextlib.suppress(Exception):

            if proc.stdin and not proc.stdin.closed:

                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")

                proc.stdin.flush()

        with contextlib.suppress(Exception):

            proc.terminate()

        with contextlib.suppress(Exception):

            proc.wait(timeout=5.0)

        with contextlib.suppress(Exception):

            if proc.poll() is None:

                proc.kill()

        self._close_server_log()



    def _server_threshold_floor(self) -> float:

        """Lowest text-grounding threshold the server must be able to serve: the

        min of the strict primary and (if enabled) the relaxed retry threshold.

        The model is loaded at this floor; each request re-gates upward."""

        floor = float(self.config.threshold)

        rt = float(getattr(self.config, "retry_threshold", -1.0))

        if rt >= 0.0:

            floor = min(floor, rt)

        return floor



    def _ensure_server(self) -> bool:

        """Start the persistent SAM3 server if needed. Returns True if usable."""

        if self._proc is not None and self._proc.poll() is None:

            return True

        self._stop_server()

        cmd = [

            str(self.python_exe), str(self.runner_path),

            "--model", str(self.model_path), "--serve",

            # Launch at the FLOOR so one loaded model serves both cascade passes;

            # each request re-gates by its own threshold (native filters per call).

            "--threshold", str(self._server_threshold_floor()),

            "--mask-threshold", str(self.config.mask_threshold),

            "--seed", str(self.config.seed),

            "--device", str(self.config.device),

            "--backend", str(self.config.backend),

        ]

        # CRITICAL: the model load dumps hundreds of KB to stderr. A PIPE we never

        # drain would fill (~64KB) and BLOCK the child before it can print the

        # ready line -> the parent then waits out the full ready timeout and falls

        # back to one-shot. Send stderr to a file (never blocks; keeps diagnostics).

        try:

            self._server_log = tempfile.NamedTemporaryFile(

                prefix="carryon_sam3_server_", suffix=".log", delete=False

            )

        except Exception:  # noqa: BLE001

            self._server_log = None

        try:

            self._proc = subprocess.Popen(

                cmd,

                stdin=subprocess.PIPE, stdout=subprocess.PIPE,

                stderr=(self._server_log or subprocess.DEVNULL),

                text=True, encoding="utf-8", errors="replace", bufsize=1,

                env=self._subprocess_env(),

            )

        except Exception as e:  # noqa: BLE001

            self._proc = None

            self._close_server_log()

            self.last_error = f"SAM3 server launch failed: {type(e).__name__}: {e}"

            return False

        line = self._read_line(self._proc.stdout, self.config.server_ready_timeout_s)

        ok = False

        if line is not None:

            with contextlib.suppress(Exception):

                ok = bool(json.loads(line).get("ready"))

        if not ok:

            tail = self._server_log_tail()

            self.last_error = (

                f"SAM3 server failed to become ready: {line!r}"

                + (f" | server log: ...{tail}" if tail else "")

            )

            self._stop_server()

            return False

        atexit.register(self._stop_server)

        print("[Sam3Segmenter] persistent server ready "

              f"pid={self._proc.pid} cuda_visible_devices="

              f"{self.config.cuda_visible_devices or '<inherited>'} "

              "(model loaded once)")

        return True



    def _finish(self, np, mask_path: Path, meta_path: Path, *, multi: bool = False):

        """Parse the meta + load the mask(s) written by either runner mode.



        ``multi`` returns a list of per-instance bool masks (ordered by score

        desc) from the stacked ``(N, H, W)`` array; otherwise a single bool mask.

        """

        if not meta_path.exists():

            return self._fail("SAM3 produced no result meta")

        try:

            result = json.loads(meta_path.read_text(encoding="utf-8"))

        except Exception as e:  # noqa: BLE001

            return self._fail(f"SAM3 result parse failed: {type(e).__name__}: {e}")

        self.last_result = result

        if not result.get("ok"):

            return self._fail(f"SAM3 rejected prompt: {result.get('reason', 'unknown')}")

        if not mask_path.exists():

            return self._fail("SAM3 reported ok but mask file missing")

        try:

            arr = np.load(mask_path)

        except Exception as e:  # noqa: BLE001

            return self._fail(f"SAM3 mask load failed: {type(e).__name__}: {e}")

        if multi:

            if arr.ndim == 2:

                arr = arr[None, ...]

            return [arr[i].astype(bool) for i in range(arr.shape[0])]

        return arr.astype(bool)



    def _segment_persistent(self, np, image_req: dict, mask_path: Path, meta_path: Path, prompt: str, *, all_masks: bool = False, threshold: Optional[float] = None):

        """Send one request to the persistent server and read the result. Returns

        the mask, or raises/returns None (signalling the caller to fall back)."""

        if not self._ensure_server():

            return self._fail(self.last_error or "SAM3 server unavailable")

        req = {

            "prompt": str(prompt),

            "output": str(mask_path),

            "meta": str(meta_path),

            "threshold": self.config.threshold if threshold is None else float(threshold),

            "mask_threshold": self.config.mask_threshold,

            "all_masks": bool(all_masks),

            **image_req,

        }

        t0 = time.monotonic()

        try:

            self._proc.stdin.write(json.dumps(req) + "\n")

            self._proc.stdin.flush()

        except Exception as e:  # noqa: BLE001

            self._stop_server()

            return self._fail(f"SAM3 server write failed: {type(e).__name__}: {e}")

        line = self._read_line(self._proc.stdout, self.config.timeout_s)

        if self.config.verbose:

            print(f"[Sam3Segmenter] persistent inference round-trip "

                  f"{time.monotonic() - t0:.2f}s")

        if line is None:

            self._stop_server()  # timed out or died -> next call restarts it

            return self._fail("SAM3 server timed out / died")

        try:

            ack = json.loads(line)

        except Exception:  # noqa: BLE001

            ack = {"req_ok": False, "error": line}

        if not ack.get("req_ok"):

            # the meta file may carry a structured reason; prefer it.

            return self._finish(np, mask_path, meta_path, multi=all_masks)

        return self._finish(np, mask_path, meta_path, multi=all_masks)



    def _segment_one_shot(self, np, image_req: dict, mask_path: Path, meta_path: Path, prompt: str, *, all_masks: bool = False, threshold: Optional[float] = None):

        # A successful (fallback) one-shot must not surface a stale server error.

        self.last_error = None

        image_args: list[str] = []

        for k, v in image_req.items():

            image_args += [f"--{k.replace('_', '-')}", str(v)]

        cmd = [

            str(self.python_exe), str(self.runner_path),

            "--model", str(self.model_path),

            "--prompt", str(prompt),

            *image_args,

            "--output", str(mask_path),

            "--meta", str(meta_path),

            "--threshold", str(self.config.threshold if threshold is None else float(threshold)),

            "--mask-threshold", str(self.config.mask_threshold),

            "--seed", str(self.config.seed),

            "--device", str(self.config.device),

            *(["--all-masks"] if all_masks else []),

        ]

        try:

            proc = subprocess.run(

                cmd, check=False, text=True, capture_output=True,

                encoding="utf-8", errors="replace", timeout=self.config.timeout_s,

                env=self._subprocess_env(),

            )

        except Exception as e:  # noqa: BLE001

            return self._fail(f"SAM3 runner launch failed: {type(e).__name__}: {e}")

        if not meta_path.exists():

            detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]

            return self._fail(f"SAM3 runner produced no result rc={proc.returncode}: {detail[0]}")

        return self._finish(np, mask_path, meta_path, multi=all_masks)



    @staticmethod

    def _prompt_fallbacks(prompt: str) -> list:

        """Ordered SAM3 prompts to try: the full phrase first, then a simpler

        head-noun fallback for awkward multi-word names. SAM3 text-grounding

        often misses a full phrase ("alphabet soup", "cream cheese", "the

        orange juice") but reliably grounds its head noun ("soup", "cheese",

        "juice"). We drop a leading article and, failing the full phrase, retry

        on the last token. De-duplicated, order-preserving."""

        base = (prompt or "").strip()

        cands = [base]

        words = base.lower().split()

        if words and words[0] in {"the", "a", "an"} and len(words) > 1:

            cands.append(" ".join(base.split()[1:]))

        # head noun (last token) as the most permissive grounding prompt

        tokens = [w for w in base.split() if w.lower() not in {"the", "a", "an"}]

        if len(tokens) > 1:

            cands.append(tokens[-1])

        seen, out = set(), []

        for c in cands:

            k = c.strip().lower()

            if c.strip() and k not in seen:

                seen.add(k)

                out.append(c.strip())

        return out



    def _run(self, rgb, *, prompt: str, all_masks: bool, threshold: Optional[float] = None):

        """Try the full prompt, then simpler head-noun fallbacks (fast-validation

        path for hard LIBERO-plus object names). First prompt that yields a mask

        wins; if all miss, the last failure reason is kept."""

        last = None

        # An empty/whitespace prompt has no usable fallback -- run once anyway so

        # ``_run_once`` records the "empty SAM3 text prompt" rejection instead of

        # silently returning None with a stale ``last_error``.

        for i, p in enumerate(self._prompt_fallbacks(prompt) or [prompt]):

            res = self._run_once(rgb, prompt=p, all_masks=all_masks, threshold=threshold)

            ok = res is not None and (not all_masks or len(res) > 0)

            if ok:

                if i > 0 and self.config.verbose:

                    print(f"[Sam3Segmenter] prompt fallback hit: {p!r} "

                          f"(original {prompt!r})")

                return res

            last = res

        return last



    def _run_once(self, rgb, *, prompt: str, all_masks: bool, threshold: Optional[float] = None):

        """Shared body for :meth:`segment` / :meth:`segment_candidates`."""

        self.last_error = None

        self.last_result = None

        if not prompt or not str(prompt).strip():

            return self._fail("empty SAM3 text prompt")

        try:

            import numpy as np  # type: ignore

        except Exception:

            return self._fail("numpy unavailable")



        with tempfile.TemporaryDirectory(prefix="carryon_sam3_") as tmp:

            td = Path(tmp)

            image_req, err = self._write_image(np, td, rgb)

            if err is not None:

                return self._fail(f"bad rgb input: {err}")

            mask_path = td / "mask.npy"

            meta_path = td / "meta.json"



            if self.config.persistent and not self._server_failed:

                mask = self._segment_persistent(

                    np, image_req, mask_path, meta_path, prompt, all_masks=all_masks,

                    threshold=threshold,

                )

                # Only fall back to one-shot when the SERVER itself is unusable

                # (launch/ready failure), not on a legitimate "no mask" result.

                if mask is not None or self._proc is not None:

                    return mask

                self._server_failed = True

                if self.config.verbose:

                    print("[Sam3Segmenter] persistent server unusable; "

                          "falling back to one-shot subprocess")



            return self._segment_one_shot(

                np, image_req, mask_path, meta_path, prompt, all_masks=all_masks,

                threshold=threshold,

            )



    def _segment_point_persistent(

        self, np, image_req: dict, mask_path: Path, meta_path: Path, point_px: tuple

    ):

        call_started = time.perf_counter()

        ensure_started = time.perf_counter()

        if not self._ensure_server():

            return self._fail(self.last_error or "SAM3 server unavailable")

        ensure_ms = (time.perf_counter() - ensure_started) * 1000.0

        req = {

            "point_coords": [float(point_px[0]), float(point_px[1])],

            "output": str(mask_path),

            "meta": str(meta_path),

            **image_req,

        }

        t0 = time.monotonic()

        write_started = time.perf_counter()

        try:

            self._proc.stdin.write(json.dumps(req) + "\n")

            self._proc.stdin.flush()

        except Exception as e:  # noqa: BLE001

            self._stop_server()

            return self._fail(f"SAM3 server write failed: {type(e).__name__}: {e}")

        write_ms = (time.perf_counter() - write_started) * 1000.0

        wait_started = time.perf_counter()

        line = self._read_line(self._proc.stdout, self.config.timeout_s)

        wait_ms = (time.perf_counter() - wait_started) * 1000.0

        if self.config.verbose:

            print(

                f"[Sam3Segmenter] persistent point round-trip "

                f"{time.monotonic() - t0:.2f}s"

            )

        if line is None:

            self._stop_server()

            return self._fail("SAM3 server timed out / died")

        parse_started = time.perf_counter()

        try:

            ack = json.loads(line)

        except Exception:  # noqa: BLE001

            ack = {"req_ok": False, "error": line}

        ack_parse_ms = (time.perf_counter() - parse_started) * 1000.0

        finish_started = time.perf_counter()

        result = self._finish(np, mask_path, meta_path, multi=False)

        finish_parse_ms = (time.perf_counter() - finish_started) * 1000.0

        self.last_timing = {

            "server_start_model_load_ms": round(ensure_ms, 3),

            "request_write_ms": round(write_ms, 3),

            "warm_inference_wait_ms": round(wait_ms, 3),

            "response_parse_ms": round(ack_parse_ms + finish_parse_ms, 3),

            "total_ms": round((time.perf_counter() - call_started) * 1000.0, 3),

        }

        if not ack.get("req_ok"):

            return result

        return result



    def _segment_point_one_shot(

        self, np, image_req: dict, mask_path: Path, meta_path: Path, point_px: tuple

    ):

        self.last_error = None

        image_args: list[str] = []

        for k, v in image_req.items():

            image_args += [f"--{k.replace('_', '-')}", str(v)]

        cmd = [

            str(self.python_exe), str(self.runner_path),

            "--model", str(self.model_path),

            "--point-x", str(float(point_px[0])),

            "--point-y", str(float(point_px[1])),

            *image_args,

            "--output", str(mask_path),

            "--meta", str(meta_path),

            "--seed", str(self.config.seed),

            "--device", str(self.config.device),

        ]

        try:

            proc = subprocess.run(

                cmd, check=False, text=True, capture_output=True,

                encoding="utf-8", errors="replace", timeout=self.config.timeout_s,

                env=self._subprocess_env(),

            )

        except Exception as e:  # noqa: BLE001

            return self._fail(f"SAM3 point runner launch failed: {type(e).__name__}: {e}")

        if not meta_path.exists():

            detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]

            return self._fail(

                f"SAM3 point runner produced no result rc={proc.returncode}: {detail[0]}"

            )

        return self._finish(np, mask_path, meta_path, multi=False)



    def _run_point_once(self, rgb, *, point_px: tuple):

        self.last_error = None

        self.last_result = None

        try:

            import numpy as np  # type: ignore

        except Exception:

            return self._fail("numpy unavailable")

        try:

            px, py = float(point_px[0]), float(point_px[1])

        except (TypeError, ValueError, IndexError):

            return self._fail(f"bad point_px {point_px!r}")



        with tempfile.TemporaryDirectory(prefix="carryon_sam3_point_") as tmp:

            td = Path(tmp)

            image_req, err = self._write_image(np, td, rgb)

            if err is not None:

                return self._fail(f"bad rgb input: {err}")

            mask_path = td / "mask.npy"

            meta_path = td / "meta.json"

            coords = (px, py)



            if self.config.persistent and not self._server_failed:

                mask = self._segment_point_persistent(

                    np, image_req, mask_path, meta_path, coords

                )

                if mask is not None or self._proc is not None:

                    return mask

                self._server_failed = True

                if self.config.verbose:

                    print("[Sam3Segmenter] persistent server unusable; "

                          "falling back to one-shot point subprocess")

            return self._segment_point_one_shot(

                np, image_req, mask_path, meta_path, coords

            )



    def segment_at_point(self, rgb, *, point_px):

        """Return a bool HxW mask from a foreground point prompt (pixel coords).



        ``point_px`` is ``(x, y)`` in the same pixel frame as ``rgb``."""

        self.last_grounding_via = None

        mask = self._run_point_once(rgb, point_px=point_px)

        if mask is not None:

            self.last_grounding_via = "point-prompt"

        return mask



    def segment(self, rgb, *, prompt: str, threshold: Optional[float] = None):

        """Return a boolean HxW target mask for ``prompt`` (or ``None``).



        Uses the persistent server when enabled (model stays loaded across calls);

        falls back to a one-shot subprocess if the server cannot be started.

        ``threshold`` overrides the config's primary text-grounding threshold for

        this call (adaptive-cascade retry); ``None`` uses ``config.threshold``."""

        self.last_grounding_via = "text"

        return self._run(rgb, prompt=prompt, all_masks=False, threshold=threshold)



    def segment_candidates(self, rgb, *, prompt: str, threshold: Optional[float] = None):

        """Return a list of per-instance bool masks for ``prompt`` (or ``None``).



        SAM3 enumerates every instance matching the (typically generic, e.g.

        "can") prompt; masks come back ordered by detection score (desc). The

        caller disambiguates spatially -- picking the instance nearest the VLA's

        pre-failure intent -- instead of trusting the argmax-score mask, which is

        exactly what mis-grounds visually-similar objects. ``threshold`` overrides

        the config's primary threshold for this call (adaptive-cascade retry)."""

        return self._run(rgb, prompt=prompt, all_masks=True, threshold=threshold)





def build_sam3_segmenter(config: Optional[Sam3Config] = None) -> Optional["Sam3Segmenter"]:

    try:

        return Sam3Segmenter(config)

    except Exception as e:  # noqa: BLE001

        print(f"[recovery.geometry] SAM3 segmenter unavailable: {type(e).__name__}: {e}")

        return None





# --------------------------------------------------------------------------- #

# Molmo2 open-vocabulary pointing (live grounding point source)               #

# --------------------------------------------------------------------------- #

@dataclass

class MolmoConfig:

    """Paths/params for the external Molmo2 pointer. See ENV doc.



    Runs in the ``molmoact`` python with a ``transformers==4.57.1`` overlay

    prepended to ``PYTHONPATH`` (Molmo2 remote code needs 4.57.1 while the base

    env pins 4.52.1). ``pythonpath_prepend`` is that overlay directory.

    """



    model_path: str = field(default_factory=lambda: os.environ.get("CARRYON_MOLMO_MODEL_PATH", ""))

    python_exe: str = field(default_factory=lambda: os.environ.get("CARRYON_MOLMO_PYTHON", ""))

    pythonpath_prepend: str = field(default_factory=lambda: os.environ.get("CARRYON_MOLMO_PYTHONPATH", ""))

    runner_path: str = ""

    max_new_tokens: int = 128

    seed: int = 123

    timeout_s: float = 120.0

    server_ready_timeout_s: float = 300.0

    # Physical GPU id(s) for this subprocess only (Molmo2-8B needs ~18 GB); keep

    # it off the cards cuRobo/GraspGen/SAM3 already hold.

    cuda_visible_devices: str = ""

    verbose: bool = False





class MolmoPointer:

    """Run Molmo2 (text -> pixel point) in its own conda env via a subprocess.



    Mirrors :class:`Sam3Segmenter`'s persistent-server pattern: load the model

    once in a long-lived child, then feed one JSON request per call over

    stdin/stdout. :meth:`point_at` returns a ``(x_px, y_px)`` tuple on the input

    image grid, or ``None`` (fails closed, records :attr:`last_error`)."""



    def __init__(self, config: Optional[MolmoConfig] = None) -> None:

        self.config = config or MolmoConfig()

        if not str(self.config.model_path).strip() or not str(self.config.python_exe).strip():

            raise FileNotFoundError("Configure CARRYON_MOLMO_MODEL_PATH and CARRYON_MOLMO_PYTHON or pass explicit resource paths")

        self.last_error: Optional[str] = None

        self.last_result: Optional[dict] = None

        self.last_timing: dict = {}

        self.model_path = Path(self.config.model_path)

        self.python_exe = Path(self.config.python_exe)

        self.runner_path = Path(

            self.config.runner_path or Path(__file__).with_name("molmo_runner.py")

        )

        missing = [

            str(p)

            for p in (self.model_path, self.python_exe, self.runner_path)

            if not p.exists()

        ]

        if missing:

            raise FileNotFoundError("missing Molmo resource(s): " + ", ".join(missing))

        self._proc: Optional[subprocess.Popen] = None

        self._server_log = None



    def _subprocess_env(self) -> dict:

        env = os.environ.copy()

        cvd = (self.config.cuda_visible_devices or "").strip()

        if cvd:

            env["CUDA_VISIBLE_DEVICES"] = cvd

        overlay = (self.config.pythonpath_prepend or "").strip()

        if overlay:

            prev = env.get("PYTHONPATH", "")

            env["PYTHONPATH"] = overlay + (os.pathsep + prev if prev else "")

        return env



    def _fail(self, reason: str):

        self.last_error = reason

        if self.config.verbose:

            print(f"[MolmoPointer] {reason}")

        return None



    def _read_line(self, stream, timeout_s: float) -> Optional[str]:

        end = time.monotonic() + float(timeout_s)

        while True:

            remaining = end - time.monotonic()

            if remaining <= 0:

                return None

            try:

                ready, _, _ = select.select([stream], [], [], remaining)

            except Exception:  # noqa: BLE001

                return None

            if not ready:

                return None

            line = stream.readline()

            if line == "":

                return None

            line = line.strip()

            if line:

                return line



    def _close_server_log(self) -> None:

        log, self._server_log = self._server_log, None

        if log is not None:

            with contextlib.suppress(Exception):

                log.close()

            with contextlib.suppress(Exception):

                os.unlink(log.name)



    def _server_log_tail(self, n: int = 400) -> str:

        log = getattr(self, "_server_log", None)

        if log is None:

            return ""

        with contextlib.suppress(Exception):

            return Path(log.name).read_text(encoding="utf-8", errors="replace")[-n:].strip()

        return ""



    def _stop_server(self) -> None:

        proc, self._proc = self._proc, None

        if proc is None:

            self._close_server_log()

            return

        with contextlib.suppress(Exception):

            if proc.stdin and not proc.stdin.closed:

                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")

                proc.stdin.flush()

        with contextlib.suppress(Exception):

            proc.terminate()

        with contextlib.suppress(Exception):

            proc.wait(timeout=5.0)

        with contextlib.suppress(Exception):

            if proc.poll() is None:

                proc.kill()

        self._close_server_log()



    def _ensure_server(self) -> bool:

        if self._proc is not None and self._proc.poll() is None:

            return True

        self._stop_server()

        cmd = [

            str(self.python_exe), str(self.runner_path),

            "--model", str(self.model_path), "--serve",

            "--max-new-tokens", str(self.config.max_new_tokens),

        ]

        try:

            self._server_log = tempfile.NamedTemporaryFile(

                prefix="carryon_molmo_server_", suffix=".log", delete=False

            )

        except Exception:  # noqa: BLE001

            self._server_log = None

        try:

            self._proc = subprocess.Popen(

                cmd,

                stdin=subprocess.PIPE, stdout=subprocess.PIPE,

                stderr=(self._server_log or subprocess.DEVNULL),

                text=True, encoding="utf-8", errors="replace", bufsize=1,

                env=self._subprocess_env(),

            )

        except Exception as e:  # noqa: BLE001

            self._proc = None

            self._close_server_log()

            self.last_error = f"Molmo server launch failed: {type(e).__name__}: {e}"

            return False

        line = self._read_line(self._proc.stdout, self.config.server_ready_timeout_s)

        ok = False

        if line is not None:

            with contextlib.suppress(Exception):

                ok = bool(json.loads(line).get("ready"))

        if not ok:

            tail = self._server_log_tail()

            self.last_error = (

                f"Molmo server failed to become ready: {line!r}"

                + (f" | server log: ...{tail}" if tail else "")

            )

            self._stop_server()

            return False

        atexit.register(self._stop_server)

        if self.config.verbose:

            print("[MolmoPointer] persistent server ready (model loaded once)")

        return True



    def point_at(

        self, rgb, obj: str, *, request_id: str = ""

    ) -> Optional[tuple[int, int]]:

        """Return Molmo2's ``(x_px, y_px)`` for ``obj`` on ``rgb``, else None."""

        call_started = time.perf_counter()

        self.last_error = None

        self.last_result = None

        self.last_timing = {}

        if not obj:

            return self._fail("empty object phrase")

        ensure_started = time.perf_counter()

        if not self._ensure_server():

            return self._fail(self.last_error or "Molmo server unavailable")

        ensure_ms = (time.perf_counter() - ensure_started) * 1000.0

        try:

            import numpy as np  # type: ignore

        except Exception as e:  # noqa: BLE001

            return self._fail(f"numpy unavailable: {e}")

        with tempfile.TemporaryDirectory(prefix="carryon_molmo_") as tmp:

            td = Path(tmp)

            image_req: dict

            if isinstance(rgb, (str, Path)):

                p = Path(rgb)

                if not p.exists():

                    return self._fail(f"image path does not exist: {p}")

                image_req = {"image": str(p)}

            elif isinstance(rgb, (bytes, bytearray)):

                img_path = td / "image.jpg"

                img_path.write_bytes(bytes(rgb))

                image_req = {"image": str(img_path)}

            else:

                try:

                    arr = np.asarray(rgb)

                except Exception as e:  # noqa: BLE001

                    return self._fail(f"unsupported rgb type {type(rgb).__name__}: {e}")

                if arr.ndim != 3 or arr.shape[2] != 3:

                    return self._fail(f"rgb array must be HxWx3, got {arr.shape}")

                npy_path = td / "image.npy"

                np.save(npy_path, arr.astype(np.uint8))

                image_req = {"image_npy": str(npy_path)}

            req = {**image_req, "request_id": str(request_id or ""),

                   "objects": [str(obj)],

                   "max_new_tokens": self.config.max_new_tokens}

            write_started = time.perf_counter()

            try:

                self._proc.stdin.write(json.dumps(req) + "\n")

                self._proc.stdin.flush()

            except Exception as e:  # noqa: BLE001

                self._stop_server()

                return self._fail(f"Molmo request write failed: {type(e).__name__}: {e}")

            write_ms = (time.perf_counter() - write_started) * 1000.0

            wait_started = time.perf_counter()

            line = self._read_line(self._proc.stdout, self.config.timeout_s)

            wait_ms = (time.perf_counter() - wait_started) * 1000.0

            if line is None:

                tail = self._server_log_tail()

                self._stop_server()

                return self._fail(

                    "Molmo request timed out / server died"

                    + (f" | server log: ...{tail}" if tail else "")

                )

            parse_started = time.perf_counter()

            try:

                resp = json.loads(line)

            except Exception as e:  # noqa: BLE001

                return self._fail(f"Molmo response parse failed: {type(e).__name__}: {e}")

            parse_ms = (time.perf_counter() - parse_started) * 1000.0

            self.last_result = resp

            if not resp.get("req_ok"):

                return self._fail(f"Molmo error: {resp.get('error', 'unknown')}")

            pt = (resp.get("points") or {}).get(str(obj))

            if pt is None:

                return self._fail(f"Molmo produced no point for {obj!r}")

            try:

                result = (int(pt[0]), int(pt[1]))

            except (TypeError, ValueError, IndexError):

                return self._fail(f"Molmo returned bad point {pt!r}")

            self.last_timing = {

                "server_start_model_load_ms": round(ensure_ms, 3),

                "request_write_ms": round(write_ms, 3),

                "warm_inference_wait_ms": round(wait_ms, 3),

                "response_parse_ms": round(parse_ms, 3),

                "total_ms": round((time.perf_counter() - call_started) * 1000.0, 3),

            }

            return result





def build_molmo_pointer(config: Optional[MolmoConfig] = None) -> Optional["MolmoPointer"]:

    try:

        return MolmoPointer(config)

    except Exception as e:  # noqa: BLE001

        print(f"[recovery.geometry] Molmo pointer unavailable: {type(e).__name__}: {e}")

        return None





# --------------------------------------------------------------------------- #

# contact-graspnet detector (runtime) + perception determinism                #

# --------------------------------------------------------------------------- #

@dataclass

class GraspNetConfig:

    """Paths/params for Contact-GraspNet (+ a segmentation source). See ENV doc."""



    checkpoint_dir: str = ""

    repo_dir: str = field(default_factory=lambda: os.environ.get("CARRYON_CONTACT_GRASPNET_REPO", ""))

    python_exe: str = field(default_factory=lambda: os.environ.get("CARRYON_CONTACT_GRASPNET_PYTHON", ""))

    runner_path: str = ""

    seg_source: str = "sam3"          # how the object mask is produced

    score_threshold: float = 0.4

    local_regions: bool = True

    filter_grasps: bool = True

    forward_passes: int = 2

    z_min_m: float = 0.2

    z_max_m: float = 2.0

    remove_table: bool = False

    table_dist_thresh_m: float = 0.01

    table_min_inlier_frac: float = 0.3

    max_retries: int = 1

    viewpoint_mode: str = "even"   # "random" or "even"

    num_viewpoints: int = 40

    max_angle_deg: float = 45.0

    viewpoint_xy_extent_m: float = 0.25

    # Contact-GraspNet poses use the gripper base. Shift along tool +z by
    # the configured TCP offset to obtain the fingertip contact point. Zero disables the shift.

    cgn_base_to_contact_m: float = 0.1034

    mask_dilation_px: int = 0

    min_mask_pixels: int = 25

    min_valid_depth_pixels: int = 25

    determinism_pos_tol_m: float = 0.01

    determinism_quat_tol_rad: float = math.radians(5)

    seed: int = 123

    timeout_s: float = 30.0

    world_from_camera: Optional[list[list[float]]] = None

    debug_dir: str = ""

    verbose: bool = False





@dataclass

class GraspGenConfig:

    """Client-side params for a running GraspGen ZMQ server.



    The heavy GraspGen model stays out-of-process. This detector only builds an

    object point cloud from the existing mask/depth perception seam, sends it to

    the server, and converts returned gripper poses into CarryOn

    :class:`GraspProposal` objects.

    """



    host: str = "127.0.0.1"

    port: int = 5557

    request_timeout_ms: int = 5000

    num_points: int = 2048

    num_grasps: int = 200

    grasp_threshold: float = -1.0

    min_grasps: int = 1

    max_tries: int = 1

    remove_outliers: bool = False

    gripper_base_to_contact_m: float = 0.1034



    mask_dilation_px: int = 0

    min_mask_pixels: int = 25

    min_valid_depth_pixels: int = 25

    seed: int = 123

    source: str = "graspgen"

    spatial_disambig: bool = True

    disambig_prompt: str = ""

    disambig_max_dist_m: float = 0.20

    point_disambig_prompt: str = "object"

    point_disambig_max_frac: float = 0.15

    point_disambig_max_area_frac: float = 0.20

    # Consistency gate for the Molmo point source: reject a Molmo point that sits

    # farther than this fraction of the image diagonal from the Judge anchor

    # (``target_px``) and fall back to the anchor point-prompt. Guards against

    # Molmo pointing at a look-alike neighbour object. <= 0 disables the gate.

    molmo_anchor_gate_frac: float = 0.08

    # Depth pop-out gate (m) for the bare anchor point-prompt: the masked median

    # depth must be at least this much nearer the camera than the surrounding

    # ring, else the mask is the tabletop and the cascade jitters/fails closed.

    # Ray-depth pop of a 2 cm object exceeds 2 cm under any oblique view, so

    # 4 mm rejects only true table masks. 0 disables.

    point_prompt_min_pop_m: float = 0.004

    # World-Z floor for the object point cloud fed to GraspGen. When

    # ``table_z_world`` is set, points with world z < table_z_world +

    # ``pc_table_clearance_m`` are dropped BEFORE inference, so an over-segmented

    # mask that leaked table pixels (e.g. a low Judge-anchor point-prompt) cannot

    # pull grasps down to table height. Fail-open: if the floor would leave fewer

    # than ``min_valid_depth_pixels`` points, the unfiltered cloud is kept.

    # ``None`` disables the floor (default; real-rig table height is per-scene).

    table_z_world: Optional[float] = None

    pc_table_clearance_m: float = 0.02

    # Sensor-noise band above the table (m). Points below ``table_z + this`` are

    # table returns; points between here and ``pc_table_clearance_m`` are the BODY

    # of a thin object (a ~2 cm cream-cheese-style slab lives entirely inside that

    # band). The z-floor uses this to tell "mask leaked the table" (dropped points

    # hug table_z -> keep the strict floor) from "object is thinner than the

    # clearance" (dropped points sit above the noise band -> relax the floor to

    # the noise band so the object body survives). Physical constant, not a

    # per-object tune.

    pc_table_noise_m: float = 0.004

    # Minimum height (m) the masked cloud must rise above table_z to count as an

    # object at all. Below this the mask IS the tabletop (grounding failure) and

    # the detector fails closed with a grounding diagnosis rather than proposing

    # table-hugging grasps. Physical constant (thinnest LIBERO objects are ~2 cm;

    # table returns sit within ~4 mm noise). 0 disables.

    pc_min_object_pop_m: float = 0.006

    world_from_camera: Optional[list[list[float]]] = None

    verbose: bool = False





@dataclass

class MaskDepthGraspConfig:

    """Params for the mask/depth top-down PCA grasp (the deterministic fallback).



    This is the analog of cap-x's OBB-pose path: it never calls a grasp network.

    The fingertip target comes straight from the segmented RGB-D point cloud, and

    the wrist yaw from PCA of the mask in world XY. It works on flat-lying produce

    (e.g. a carrot) precisely because it makes no learned assumption about

    graspable geometry -- it just pinches across the object's short axis from above.

    """



    z_percentile: float = 50.0       # mask world-z percentile for the fingertip target

    xy_stat: str = "median"          # "median" or "mean" centroid in world XY

    # Top-band centroid (angled-overhead correction): an agentview-style camera

    # looks DOWN AT AN ANGLE, so a tall object's visible front/side faces

    # back-project to their true (camera-facing) world XY -- pulling the median of

    # ALL mask points toward the front face, off the object's real footprint

    # centre. A top-down grasp centred there hits the face and tips the object

    # (observed on the LIBERO milk carton). When >0, compute center_xy from only

    # the mask points whose world-z is within this band of the object top

    # (``center_top_band_percentile``), i.e. the TOP-FACE footprint centre, which

    # is unbiased. 0 = legacy (all points). Short/flat objects: the band spans the

    # whole object, so center_xy is unchanged.

    center_top_band_m: float = 0.0

    center_top_band_percentile: float = 90.0

    center_top_band_min_pixels: int = 20

    # Property-driven grasp descent (generalises the per-object z knob). A top-down

    # pinch must land on the object's WIDE body, not its top surface / tapered top

    # (e.g. a gable-top carton's ~0.5 cm peak reads jaws-empty and slips). Instead

    # of a fixed world-z offset (a magic constant that over-descends short objects),

    # anchor on the empirically-good mask-median height and descend a FRACTION of

    # the object's own perceived height, capped and floored, so the same rule fits a

    # tall milk carton (larger descent onto the wide body) and a short can (tiny

    # descent, won't hit the table). When enabled,

    # target_z = z_median - clamp(frac*height, min, max), height =

    # z_top(percentile) - z_base(percentile). Still clamped by the table floor.

    # 0/False keeps the legacy median-percentile behaviour (rig unchanged).

    grasp_below_top_enabled: bool = False

    # Grasp-height anchor: table uses a fraction of object height above its base;
    # median uses z_med minus a fraction of the object height.

    grasp_anchor: str = "median"

    grasp_height_frac: float = 0.45

    grasp_below_top_frac: float = 0.32    # descend this fraction of object height below the median

    grasp_below_top_min_m: float = 0.0

    grasp_below_top_max_m: float = 0.05

    grasp_top_percentile: float = 90.0    # z_top: robust object top

    grasp_base_percentile: float = 10.0   # z_base: robust object base (visible)

    # Spatial disambiguation: when a ``target_xy`` prior (the VLA's pre-failure

    # end-effector XY) is supplied, enumerate ALL instances matching the prompt

    # and pick the one whose world centroid is nearest the prior, instead of

    # trusting SAM3's argmax mask. ``disambig_prompt`` is the enumeration prompt

    # (a generic category noun like "can" makes SAM3 list every candidate). When

    # empty, the runtime ``object_hint`` is used -- but a specific phrase often

    # collapses to a single (possibly wrong) instance, so set this per rig.

    spatial_disambig: bool = True

    disambig_prompt: str = ""

    # Reject a disambiguation pick whose nearest-instance centroid is still

    # farther than this from the prior (m); falls back to the plain mask path.

    disambig_max_dist_m: float = 0.20

    # Point-disambiguation (Judge anchor): when a ``target_px`` (normalized x,y)

    # is supplied, enumerate instance masks GEOMETRICALLY and pick the one whose

    # PIXEL centroid is nearest the point. ``point_disambig_prompt`` is the broad

    # enumeration prompt used when the object_hint-derived category yields nothing

    # ("object" lists every blob; the point then selects the right one). This is

    # what lets a food name SAM3 can't ground ("cream cheese") still resolve.

    point_disambig_prompt: str = "object"

    # Reject a point-disambig pick whose nearest-instance pixel centroid is still

    # farther than this fraction of the image diagonal from the anchor; falls

    # back to the world-prior / plain mask path.

    point_disambig_max_frac: float = 0.25

    # Drop enumeration masks larger than this fraction of the image before the

    # anchor pick. The generic "object" prompt also returns big containers

    # (basket, table) whose centroid can sit closer to the anchor than the small

    # target beside it -- so a nearest-centroid pick grabs the basket. Grasp

    # targets in libero_object are small; this gate removes the container. If it

    # would drop every candidate, it is ignored (fail-open).

    point_disambig_max_area_frac: float = 0.15

    # Consistency gate for the Molmo point source: reject a Molmo point that sits

    # farther than this fraction of the image diagonal from the Judge anchor

    # (``target_px``) and fall back to the anchor point-prompt. This routes final

    # authority to the reflector/Judge anchor and guards against Molmo pointing at

    # a look-alike neighbour object (the wrong-ketchup-neighbour bug). <= 0

    # disables the gate (accept any Molmo point). Reflector/operator-tunable.

    molmo_anchor_gate_frac: float = 0.08

    # Depth pop-out gate (m) for the bare anchor point-prompt: the masked median

    # depth must be at least this much nearer the camera than the surrounding

    # ring, else the mask is the tabletop and the cascade jitters/fails closed.

    # Ray-depth pop of a 2 cm object exceeds 2 cm under any oblique view, so

    # 4 mm rejects only true table masks. 0 disables.

    point_prompt_min_pop_m: float = 0.004

    # Sensor-noise band (m) above the support plane; masked points below

    # table_z + this are table returns and are dropped before centroid /

    # height percentiles (below-top descent hygiene). 0 disables.

    mask_table_noise_m: float = 0.004

    # Tool-mount calibration (NOT per-object tuning): which panda_hand tool axis is

    # the PHYSICAL gripper opening axis, plus an optional fixed yaw offset.

    physical_open_axis: str = "y"    # "x" or "y"

    physical_open_axis_angle_deg: Optional[float] = None

    tool_yaw_offset_deg: float = 0.0

    flip_opening: bool = False

    offset_world_m: tuple = (0.0, 0.0, 0.0)

    # Optional tabletop safety floor for top-down mask grasps. When enabled, the

    # detector estimates the local table height from a ring around the object and

    # prevents the fingertip target from going below ``table_high + clearance``.

    # This protects against thin objects whose mask-z percentile lies close to the

    # table and against gripper collision geometry not captured by the fingertip.

    local_table_clearance_m: float = 0.0

    local_table_ring_inner_m: float = 0.02

    local_table_ring_outer_m: float = 0.10

    local_table_z_percentile: float = 90.0

    local_table_min_points: int = 200

    local_table_max_above_object_m: float = 0.03

    local_table_max_below_object_m: float = 0.08

    min_valid_depth_pixels: int = 50

    mask_dilation_px: int = 0

    # Erode segmentation-mask edges before centroid and point-cloud extraction.
    # A zero erosion radius leaves the mask unchanged.

    mask_erode_px: int = 0

    world_from_camera: Optional[list[list[float]]] = None

    verbose: bool = False





# Generic category nouns for spatial disambiguation when ``grasp_disambig_prompt``

# is empty: SAM3 lists every instance of the category instead of argmax on the

# full phrase ("alphabet soup can" -> "can").

_DISAMBIG_CATEGORY_WORDS = frozenset({

    "can", "bowl", "plate", "bottle", "mug", "cup", "box", "jar", "pot", "pan",

})





def resolve_disambig_prompt(config_prompt: str, object_hint: str) -> str:

    """Pick the SAM3 enumeration prompt for spatial disambiguation.



    An explicit ``config_prompt`` wins. Otherwise derive a generic category noun

    from ``object_hint`` (last token if it is a known category, else the full hint).

    """

    explicit = (config_prompt or "").strip()

    if explicit:

        return explicit

    hint = (object_hint or "").strip().lower()

    if not hint:

        return ""

    words = hint.split()

    if words and words[-1] in _DISAMBIG_CATEGORY_WORDS:

        return words[-1]

    return hint





@dataclass

class HybridGraspConfig:

    """Knobs for the hybrid grasp-source heuristic shell.



    The reflection LLM tunes *these* (per DESIGN_OVERVIEW_CN §4.1) -- never the

    learned grasp-network weights. ``prefer`` lets a policy force one source.



    Default is ``prefer="fallback"`` (mask top-down) because for our use case --

    pick objects standing on a clean-to-cluttered table from above -- the

    deterministic mask path is more reliable than CGN. CGN's only real edge is

    6-DoF grasps for handles / non-top-down geometry (a hose handle, a tool

    grip), and even there our single-best-grasp pipeline can't reproduce

    cap-x's robustness because we don't pass a top-K goalset to cuRobo nor use

    a grasp-approach metric (sub-cap-x:plan_to_grasp_poses) -- so a CGN grasp

    that's e.g. 20° tilted has the wrong standoff direction for our world-z

    standoff scheme. Until those motion-side pieces are added, CGN is opt-in

    via ``prefer="primary"`` (LLM policies asking for it) or by a future

    LLM-side rule that recognises "I'm grasping a handle".

    """



    # Trust gate on Contact-GraspNet's confidence -- a LOW garbage floor only.

    # CGN scores are stochastic (point-cloud downsampling uses np.random.choice)

    # and do NOT separate object classes on our rig: a cup oscillates ~0.16-0.27

    # run-to-run and a flat carrot is ~0.22, so a tight score gate (e.g. 0.2) just

    # randomly rejects good cup grasps. The real flat-vs-graspable discriminator is

    # the geometric table gate below (sub-table z => distrust); the score gate is

    # kept low (0.1) so it only drops outright-garbage grasps and lets the table

    # gate do the actual routing. This is still a reflection-diff target.

    min_primary_score: float = 0.1

    prefer: str = "fallback"         # "auto" | "primary" | "fallback"

    # Geometric trust gate (robust to CGN's fragile score): distrust a primary

    # grasp whose fingertip z sits at/below the LOCAL table plane (the signature of

    # a thin/flat object CGN cannot really grasp -- it returns a sub-table contact).

    # When the local table cannot be estimated (e.g. no depth in unit tests) the

    # gate is skipped and only the score gate applies. 0 keeps score-only behaviour.

    min_primary_z_above_table_m: float = 0.005

    table_ring_inner_m: float = 0.02

    table_ring_outer_m: float = 0.10

    table_z_percentile: float = 90.0

    table_min_points: int = 200

    table_max_above_object_m: float = 0.03

    table_max_below_object_m: float = 0.08

    verbose: bool = False





def _sam_text_prompt_candidates(

    object_hint: str,

    extra_prompts: Optional[list[str]] = None,

) -> list[str]:

    """Ordered SAM3 text prompts: task hint, generic fallbacks, reflector extras."""

    base = (object_hint or "").strip()

    cands = list(Sam3Segmenter._prompt_fallbacks(base)) if base else []

    seen = {c.strip().lower() for c in cands if c.strip()}

    for prompt in extra_prompts or []:

        text = (prompt or "").strip()

        key = text.lower()

        if text and key not in seen:

            seen.add(key)

            cands.append(text)

    return cands or ([base] if base else [])





def _segment_text_with_retry(

    segmenter,

    rgb,

    *,

    object_hint: str,

    extra_prompts: Optional[list[str]] = None,

    verbose: bool = False,

):

    """Run SAM3 text grounding at the strict threshold, then configured retry.



    ``threshold=None`` means "use the segmenter's primary threshold"; an explicit

    retry threshold lets the persistent SAM3 server keep a low model floor while

    each request re-gates detections at the desired score.

    """

    if segmenter is None:

        return None

    thresholds = [None]

    seg_cfg = getattr(segmenter, "config", None)

    rt = float(getattr(seg_cfg, "retry_threshold", -1.0))

    primary = float(getattr(seg_cfg, "threshold", 0.5))

    if rt >= 0.0 and rt < primary:

        thresholds.append(rt)

    for prompt in _sam_text_prompt_candidates(object_hint, extra_prompts):

        for thr in thresholds:

            try:

                mask = segmenter.segment(rgb, prompt=prompt, threshold=thr)

            except TypeError:

                # Older test doubles / segmenters predate the threshold kwarg; use

                # their single-pass behavior rather than failing the whole detector.

                mask = segmenter.segment(rgb, prompt=prompt)

            if mask is None:

                continue

            if thr is not None:

                try:

                    segmenter.last_grounding_via = f"text@{float(thr):.2f}"

                except Exception:  # noqa: BLE001 - diagnostic only

                    pass

                if verbose:

                    try:

                        px = int(mask.sum())

                    except Exception:  # noqa: BLE001 - diagnostic only

                        px = -1

                    print(

                        f"[SAM3] grounding_via=text@{float(thr):.2f} "

                        f"prompt={prompt!r} px={px}"

                    )

            return mask

    return None





class ContactGraspNetDetector:

    """Runtime perception (implements ``GraspDetector``) via an external runner.



    The PyTorch graspnet/SAM stacks live in their own conda env, so CarryOn never

    imports torch or contact-graspnet in-process. ``detect`` writes a compact RGB-D

    packet to a temp file, invokes the runner, and converts the best camera-frame

    grasp into a world-frame :class:`GraspProposal`.



    Segmentation is intentionally explicit: pass ``intrinsics["segmap"]`` or

    ``intrinsics["mask"]`` plus ``object_hint="seg:<id>"``. Text-to-mask via SAM3

    should feed that same seam; without a mask/segmap the detector fails closed."""



    def __init__(

        self,

        config: Optional[GraspNetConfig] = None,

        *,

        segmenter: Optional["Sam3Segmenter"] = None,

    ) -> None:

        self.config = config or GraspNetConfig()

        if not str(self.config.repo_dir).strip() or not str(self.config.python_exe).strip():

            raise FileNotFoundError("Configure CARRYON_CONTACT_GRASPNET_REPO and CARRYON_CONTACT_GRASPNET_PYTHON or pass explicit resource paths")

        self.segmenter = segmenter

        self.last_error: Optional[str] = None

        self.last_runner_result: Optional[dict] = None

        self.repo_dir = Path(self.config.repo_dir)

        self.python_exe = Path(self.config.python_exe)

        self.runner_path = Path(

            self.config.runner_path

            or Path(__file__).with_name("graspnet_pytorch_runner.py")

        )

        self.checkpoint_dir = Path(

            self.config.checkpoint_dir

            or self.repo_dir / "checkpoints" / "contact_graspnet"

        )

        missing = [

            str(p)

            for p in (self.repo_dir, self.python_exe, self.runner_path, self.checkpoint_dir)

            if not p.exists()

        ]

        model_pt = self.checkpoint_dir / "checkpoints" / "model.pt"

        if not model_pt.exists():

            missing.append(str(model_pt))

        if missing:

            raise FileNotFoundError("missing graspnet resource(s): " + ", ".join(missing))



    def _fail(self, reason: str):

        self.last_error = reason

        if self.config.verbose:

            print(f"[ContactGraspNetDetector] {reason}")

        return None



    def _intrinsics_matrix(self, intrinsics: dict) -> list[list[float]]:

        if "K" in intrinsics:

            k = intrinsics["K"]

        elif "intrinsics" in intrinsics:

            k = intrinsics["intrinsics"]

        elif {"fx", "fy", "ppx", "ppy"}.issubset(intrinsics):

            k = [

                [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],

                [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],

                [0.0, 0.0, 1.0],

            ]

        else:

            raise ValueError("intrinsics must include K/intrinsics or fx/fy/ppx/ppy")

        flat = [float(v) for row in k for v in row]

        if len(flat) != 9:

            raise ValueError("camera intrinsics must be 3x3")

        return [flat[0:3], flat[3:6], flat[6:9]]



    def _world_from_camera(self, intrinsics: dict) -> Optional[list[list[float]]]:

        mat = (

            intrinsics.get("world_from_camera")

            or intrinsics.get("extrinsics")

            or self.config.world_from_camera

        )

        if mat is None:

            return None

        flat = [float(v) for row in mat for v in row]

        if len(flat) != 16:

            raise ValueError("world_from_camera/extrinsics must be 4x4")

        return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]



    @staticmethod

    def _looks_like_seg_id(object_hint: str) -> bool:

        """True for explicit ``seg:<id>`` / ``id:<n>`` style hints (not free text)."""

        return bool(re.fullmatch(r"\s*(?:seg(?:ment)?|id)\s*:?\s*\d+\s*", object_hint or "", re.IGNORECASE))



    def _segmap_and_id(self, intrinsics: dict, object_hint: str):

        segmap = intrinsics.get("segmap")

        if segmap is None and "mask" in intrinsics:

            mask = intrinsics["mask"]

            try:

                import numpy as np  # type: ignore



                segmap = np.asarray(mask).astype(np.int32)

            except Exception:  # noqa: BLE001

                segmap = mask

        if segmap is None:

            return None, None

        match = re.search(r"(?:seg(?:ment)?|id)?\s*:?\s*(\d+)", object_hint or "")

        segmap_id = int(match.group(1)) if match else 1

        return segmap, segmap_id



    @staticmethod

    def _dilate_mask(mask, iterations: int):

        if iterations <= 0:

            return mask

        import numpy as np  # type: ignore



        out = mask.astype(bool)

        for _ in range(iterations):

            p = np.pad(out, 1, mode="constant", constant_values=False)

            out = (

                p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]

                | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]

                | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]

            )

        return out



    @staticmethod

    def _erode_mask(mask, iterations: int):

        if iterations <= 0:

            return mask

        import numpy as np  # type: ignore



        out = mask.astype(bool)

        for _ in range(iterations):

            p = np.pad(out, 1, mode="constant", constant_values=False)

            out = (

                p[1:-1, 1:-1]

                & p[:-2, 1:-1]

                & p[2:, 1:-1]

                & p[1:-1, :-2]

                & p[1:-1, 2:]

            )

        return out



    def _prepare_depth_and_segmap(self, np, depth_m, segmap, segmap_id: int):

        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        if depth.ndim != 2:

            raise ValueError(f"depth must be HxW, got shape {depth.shape}")



        seg = np.asarray(segmap)

        if seg.ndim == 3 and 1 in seg.shape:

            seg = np.squeeze(seg)

        if seg.ndim != 2:

            raise ValueError(f"segmap/mask must be HxW, got shape {seg.shape}")

        if seg.shape != depth.shape:

            raise ValueError(f"segmap shape {seg.shape} does not match depth shape {depth.shape}")



        target = seg.astype(np.int32) == int(segmap_id)

        target = self._dilate_mask(target, int(self.config.mask_dilation_px))

        mask_pixels = int(target.sum())

        if mask_pixels < int(self.config.min_mask_pixels):

            raise ValueError(f"target mask too small: {mask_pixels} px")



        valid_depth = np.isfinite(depth) & (depth > 0.0) & target

        valid_pixels = int(valid_depth.sum())

        if valid_pixels < int(self.config.min_valid_depth_pixels):

            raise ValueError(f"target has too few valid depth pixels: {valid_pixels} px")



        seg_out = np.zeros_like(depth, dtype=np.int32)

        seg_out[target] = int(segmap_id)

        return depth, seg_out



    @staticmethod

    def _matmul4(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:

        return [

            [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]

            for i in range(4)

        ]



    @staticmethod

    def _quat_wxyz_from_rot(r: list[list[float]]) -> list[float]:

        tr = r[0][0] + r[1][1] + r[2][2]

        if tr > 0.0:

            s = math.sqrt(tr + 1.0) * 2.0

            return [0.25 * s, (r[2][1] - r[1][2]) / s, (r[0][2] - r[2][0]) / s, (r[1][0] - r[0][1]) / s]

        if r[0][0] > r[1][1] and r[0][0] > r[2][2]:

            s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0

            return [(r[2][1] - r[1][2]) / s, 0.25 * s, (r[0][1] + r[1][0]) / s, (r[0][2] + r[2][0]) / s]

        if r[1][1] > r[2][2]:

            s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0

            return [(r[0][2] - r[2][0]) / s, (r[0][1] + r[1][0]) / s, 0.25 * s, (r[1][2] + r[2][1]) / s]

        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0

        return [(r[1][0] - r[0][1]) / s, (r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s, 0.25 * s]



    def _grasp_cam_to_proposal(

        self, grasp_cam, score: float, world_from_camera, segmap_id: int

    ) -> GraspProposal:

        """Camera-frame 4x4 + score -> world-frame :class:`GraspProposal` with

        the panda_hand → fingertip TCP shift baked in."""

        grasp_world = self._matmul4(world_from_camera, grasp_cam)

        rot = [row[:3] for row in grasp_world[:3]]

        quat = self._quat_wxyz_from_rot(rot)

        norm = math.sqrt(sum(v * v for v in quat)) or 1.0

        quat = [v / norm for v in quat]

        approach = [grasp_world[i][2] for i in range(3)]

        anorm = math.sqrt(sum(a * a for a in approach)) or 1.0

        approach = [a / anorm for a in approach]

        d = float(self.config.cgn_base_to_contact_m)

        position = [float(grasp_world[i][3] + d * approach[i]) for i in range(3)]

        return GraspProposal(

            position=position,

            quaternion=quat,

            score=float(score),

            source=f"contact_graspnet_pytorch:{segmap_id}",

        )



    def _run_inference(self, *, rgb, depth_m, intrinsics, object_hint, top_k: int, sam_text_prompts=None):

        """Shared subprocess-runner pipeline. Returns

        ``(result_dict, world_from_camera, segmap_id)`` on success or ``None`` on

        any failure (with ``self.last_error`` set). ``top_k`` is forwarded to the

        runner so a single subprocess invocation can populate either the legacy

        single-best output or the goalset-sized top-K used by ``detect_many``."""

        self.last_error = None

        self.last_runner_result = None

        try:

            import numpy as np  # type: ignore

        except Exception:

            self._fail("numpy unavailable")

            return None



        try:

            k = self._intrinsics_matrix(intrinsics)

            world_from_camera = self._world_from_camera(intrinsics)

            segmap, segmap_id = self._segmap_and_id(intrinsics, object_hint)

        except Exception as e:

            self._fail(f"bad grasp detector inputs: {type(e).__name__}: {e}")

            return None

        if world_from_camera is None:

            self._fail("missing world_from_camera/extrinsics")

            return None



        # SAM3 fallback: no explicit mask/segmap but a free-text object hint + an

        # RGB frame + a wired segmenter -> ground the text to a target mask.

        if segmap is None:

            if self.segmenter is None:

                self._fail("no segmap/mask provided and no SAM3 segmenter wired")

                return None

            if rgb is None:

                self._fail("no segmap/mask provided and no rgb for SAM3 grounding")

                return None

            if self._looks_like_seg_id(object_hint):

                self._fail(f"object_hint {object_hint!r} expects a provided mask/segmap")

                return None

            mask = _segment_text_with_retry(

                self.segmenter,

                rgb,

                object_hint=object_hint,

                extra_prompts=sam_text_prompts,

                verbose=self.config.verbose,

            )

            if mask is None:

                self._fail(f"SAM3 grounding failed: {self.segmenter.last_error}")

                return None

            segmap, segmap_id = mask, 1

        if segmap is None or segmap_id is None:

            self._fail("missing world_from_camera or segmap/mask")

            return None



        try:

            depth_arr, segmap_arr = self._prepare_depth_and_segmap(np, depth_m, segmap, segmap_id)

        except Exception as e:

            self._fail(f"invalid RGB-D segmentation packet: {type(e).__name__}: {e}")

            return None



        with tempfile.TemporaryDirectory(prefix="carryon_graspnet_") as td:

            in_path = Path(td) / "input.npz"

            out_path = Path(td) / "output.json"

            np.savez(

                in_path,

                depth=depth_arr,

                K=np.asarray(k, dtype=np.float32),

                segmap=segmap_arr,

            )

            cmd = [

                str(self.python_exe),

                str(self.runner_path),

                "--repo-dir", str(self.repo_dir),

                "--ckpt-dir", str(self.checkpoint_dir),

                "--input", str(in_path),

                "--output", str(out_path),

                "--segmap-id", str(segmap_id),

                "--seed", str(self.config.seed),

                "--score-threshold", str(self.config.score_threshold),

                "--local-regions", "1" if self.config.local_regions else "0",

                "--filter-grasps", "1" if self.config.filter_grasps else "0",

                "--forward-passes", str(self.config.forward_passes),

                "--z-min", str(self.config.z_min_m),

                "--z-max", str(self.config.z_max_m),

                "--remove-table", "1" if self.config.remove_table else "0",

                "--table-dist-thresh", str(self.config.table_dist_thresh_m),

                "--table-min-inlier-frac", str(self.config.table_min_inlier_frac),

                "--max-retries", str(self.config.max_retries),

                "--viewpoint-mode", str(self.config.viewpoint_mode),

                "--num-viewpoints", str(self.config.num_viewpoints),

                "--max-angle-deg", str(self.config.max_angle_deg),

                "--viewpoint-xy-extent", str(self.config.viewpoint_xy_extent_m),

                "--top-k", str(max(1, int(top_k))),

            ]

            env = os.environ.copy()

            env["PYTHONPATH"] = f"{self.repo_dir}:{self.repo_dir / 'Pointnet_Pointnet2_pytorch'}:" + env.get("PYTHONPATH", "")

            try:

                proc = subprocess.run(

                    cmd,

                    cwd=str(self.repo_dir),

                    env=env,

                    check=False,

                    text=True,

                    capture_output=True,

                    encoding="utf-8",

                    errors="replace",

                    timeout=self.config.timeout_s,

                )

            except Exception as e:

                self._fail(f"runner launch failed: {type(e).__name__}: {e}")

                return None

            if proc.returncode != 0 or not out_path.exists():

                detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]

                self._fail(f"runner failed rc={proc.returncode}: {detail[0]}")

                return None

            try:

                result = json.loads(out_path.read_text(encoding="utf-8"))

            except Exception as e:

                self._fail(f"runner result parse failed: {type(e).__name__}: {e}")

                return None

            self.last_runner_result = result

            if self.config.debug_dir:

                debug_dir = Path(self.config.debug_dir)

                debug_dir.mkdir(parents=True, exist_ok=True)

                stamp = f"graspnet_{os.getpid()}_{int(time.time() * 1000)}"

                np.savez(

                    debug_dir / f"{stamp}.npz",

                    depth=depth_arr,

                    K=np.asarray(k, dtype=np.float32),

                    segmap=segmap_arr,

                )

                (debug_dir / f"{stamp}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

            if not result.get("ok"):

                self._fail(f"runner rejected grasp: {result.get('reason', 'unknown')}")

                return None

            return result, world_from_camera, segmap_id



    def detect(self, *, rgb, depth_m, intrinsics, object_hint, sam_text_prompts=None, **kwargs):  # pragma: no cover

        # CGN has no anchored-grounding path: a judge/reflector target point

        # arriving here is DROPPED. Warn loudly -- on the real rig (default

        # grasp_primary: cgn) this is a wrong-object-grasp risk, and silence



        _dropped = [k for k in ("target_px", "target_xy", "grounding") if kwargs.get(k) is not None]

        if _dropped:

            print(

                "[ContactGraspNetDetector] WARNING: anchored-grounding inputs "

                f"{_dropped} are not supported by CGN and were IGNORED "

                "(use the graspgen/hybrid detector to consume them)",

                flush=True,

            )

        out = self._run_inference(

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            top_k=1,

            sam_text_prompts=sam_text_prompts,

        )

        if out is None:

            return None

        result, world_from_camera, segmap_id = out

        try:

            return self._grasp_cam_to_proposal(

                result["grasp_cam"], float(result.get("score", 0.0)), world_from_camera, segmap_id

            )

        except Exception as e:

            return self._fail(f"runner result parse failed: {type(e).__name__}: {e}")



    def detect_many(

        self,

        *,

        rgb,

        depth_m,

        intrinsics,

        object_hint,

        top_k: int = 15,

        sam_text_prompts=None,

        **kwargs,

    ):  # pragma: no cover - needs GPU/service env

        """Return top-K grasps for cap-x-style cuRobo goalset planning.

        Candidates are ordered by descending score. The motion layer tries IK and
        collision-free planning in that order. A deterministic top-down detector
        returns one candidate; perception failure returns an empty list."""

        out = self._run_inference(

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            top_k=max(1, int(top_k)),

            sam_text_prompts=sam_text_prompts,

        )

        if out is None:

            return []

        result, world_from_camera, segmap_id = out

        try:

            grasps_cam = result.get("grasps_cam") or [result["grasp_cam"]]

            scores = result.get("scores") or [float(result.get("score", 0.0))]

        except Exception as e:

            self._fail(f"runner result parse failed: {type(e).__name__}: {e}")

            return []

        proposals: list[GraspProposal] = []

        for g_cam, sc in zip(grasps_cam, scores):

            try:

                proposals.append(

                    self._grasp_cam_to_proposal(g_cam, float(sc), world_from_camera, segmap_id)

                )

            except Exception:

                continue

        return proposals





class GraspGenDetector:

    """Runtime top-K grasp perception via a running GraspGen ZMQ server.



    GraspGen takes an object point cloud and returns 6-DoF gripper poses. This

    adapter deliberately sends **world-frame** object points, so returned poses are

    treated as world-frame poses and can flow directly into cuRobo goalset

    planning after the same gripper-base -> fingertip/contact shift used for CGN.

    """



    def __init__(

        self,

        config: Optional[GraspGenConfig] = None,

        *,

        segmenter: Optional["Sam3Segmenter"] = None,

        pointer: Optional["MolmoPointer"] = None,

        client_factory=None,

    ) -> None:

        self.config = config or GraspGenConfig()

        self.segmenter = segmenter

        self.pointer = pointer

        self.client_factory = client_factory

        self.last_error: Optional[str] = None

        self.last_server_result: Optional[dict] = None

        self.last_debug: dict = {}



    def _fail(self, reason: str):

        self.last_error = reason

        if self.config.verbose:

            print(f"[GraspGenDetector] {reason}")

        return None



    def _world_from_camera(self, intrinsics: dict) -> Optional[list[list[float]]]:

        mat = (

            intrinsics.get("world_from_camera")

            or intrinsics.get("extrinsics")

            or self.config.world_from_camera

        )

        if mat is None:

            return None

        flat = [float(v) for row in mat for v in row]

        if len(flat) != 16:

            raise ValueError("world_from_camera/extrinsics must be 4x4")

        return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]



    @staticmethod

    def _looks_like_seg_id(object_hint: str) -> bool:

        return ContactGraspNetDetector._looks_like_seg_id(object_hint)



    def _segmap_and_id(self, intrinsics: dict, object_hint: str):

        segmap = intrinsics.get("segmap")

        if segmap is None and "mask" in intrinsics:

            mask = intrinsics["mask"]

            try:

                import numpy as np  # type: ignore



                segmap = np.asarray(mask).astype(np.int32)

            except Exception:  # noqa: BLE001

                segmap = mask

        if segmap is None:

            return None, None

        match = re.search(r"(?:seg(?:ment)?|id)?\s*:?\s*(\d+)", object_hint or "")

        segmap_id = int(match.group(1)) if match else 1

        return segmap, segmap_id



    def _object_point_cloud_world(

        self,

        *,

        depth_m,

        intrinsics,

        object_hint,

        rgb=None,

        target_xy=None,

        target_px=None,

        sam_text_prompts=None,

        scene_reference=None,

        exclude_masks=None,

    ):

        try:

            import numpy as np  # type: ignore

        except Exception as e:  # noqa: BLE001

            return self._fail(f"numpy unavailable: {type(e).__name__}: {e}")



        try:

            k = np.asarray(ContactGraspNetDetector._intrinsics_matrix(self, intrinsics), dtype=np.float32)

            wfc = self._world_from_camera(intrinsics)

            segmap, segmap_id = self._segmap_and_id(intrinsics, object_hint)

        except Exception as e:

            return self._fail(f"bad GraspGen detector inputs: {type(e).__name__}: {e}")

        if wfc is None:

            return self._fail("missing world_from_camera/extrinsics")

        world_from_camera = np.asarray(wfc, dtype=np.float32)



        if segmap is None:

            if self.segmenter is None:

                return self._fail("no segmap/mask provided and no SAM3 segmenter wired")

            if rgb is None:

                return self._fail("no segmap/mask provided and no rgb for SAM3 grounding")

            if self._looks_like_seg_id(object_hint):

                return self._fail(f"object_hint {object_hint!r} expects a provided mask/segmap")

            chosen = None

            if self.config.spatial_disambig:

                disambig = MaskDepthGraspDetector(

                    MaskDepthGraspConfig(

                        spatial_disambig=True,

                        disambig_prompt=self.config.disambig_prompt,

                        disambig_max_dist_m=self.config.disambig_max_dist_m,

                        point_disambig_prompt=self.config.point_disambig_prompt,

                        point_disambig_max_frac=self.config.point_disambig_max_frac,

                        point_disambig_max_area_frac=self.config.point_disambig_max_area_frac,

                        molmo_anchor_gate_frac=self.config.molmo_anchor_gate_frac,

                        min_valid_depth_pixels=self.config.min_valid_depth_pixels,

                        verbose=self.config.verbose,

                    ),

                    segmenter=self.segmenter,

                    pointer=self.pointer,

                )

                disambig.object_catalog = getattr(self, "object_catalog", None)

                if target_px is not None:

                    chosen = disambig._disambiguate_mask_px(

                        np, rgb=rgb, depth_m=depth_m, object_hint=object_hint,

                        target_px=target_px, intrinsics=intrinsics,

                        scene_reference=scene_reference,

                        extra_prompts=sam_text_prompts,

                        target_xy=target_xy,

                        exclude_masks=exclude_masks,

                    )

                elif target_xy is not None:

                    chosen = disambig._disambiguate_mask(

                        np,

                        rgb=rgb,

                        depth_m=depth_m,

                        K=k,

                        world_from_camera=world_from_camera,

                        object_hint=object_hint,

                        target_xy=target_xy,

                    )

            if chosen is not None:

                mask = chosen

            elif target_px is not None:

                via = getattr(self.segmenter, "last_grounding_via", None) or "failed"

                return self._fail(f"grounding failed (via={via}): {self.segmenter.last_error}")

            else:

                mask = _segment_text_with_retry(

                    self.segmenter,

                    rgb,

                    object_hint=object_hint,

                    extra_prompts=sam_text_prompts,

                    verbose=self.config.verbose,

                )

            if mask is None:

                return self._fail(f"SAM3 grounding failed: {self.segmenter.last_error}")

            segmap, segmap_id = mask, 1

            try:  # persisted for the pre-grasp identity check (fails open)

                self.segmenter.last_grounding_mask = np.asarray(mask) > 0

                self.segmenter.last_grounding_rgb = rgb

            except Exception:  # noqa: BLE001

                pass



        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        if depth.ndim != 2:

            return self._fail(f"depth must be HxW, got shape {depth.shape}")

        seg = np.asarray(segmap)

        if seg.ndim == 3 and 1 in seg.shape:

            seg = np.squeeze(seg)

        if seg.ndim != 2:

            return self._fail(f"segmap/mask must be HxW, got shape {seg.shape}")

        if seg.shape != depth.shape:

            return self._fail(f"segmap shape {seg.shape} does not match depth shape {depth.shape}")



        target = seg.astype(np.int32) == int(segmap_id)

        target = ContactGraspNetDetector._dilate_mask(target, int(self.config.mask_dilation_px))

        mask_pixels = int(target.sum())

        if mask_pixels < int(self.config.min_mask_pixels):

            return self._fail(f"target mask too small: {mask_pixels} px")



        valid = np.isfinite(depth) & (depth > 0.0) & target

        valid_pixels = int(valid.sum())

        if valid_pixels < int(self.config.min_valid_depth_pixels):

            return self._fail(f"target has too few valid depth pixels: {valid_pixels} px")



        ys, xs = np.nonzero(valid)

        z = depth[ys, xs].astype(np.float32)

        x = (xs.astype(np.float32) - k[0, 2]) * z / k[0, 0]

        y = (ys.astype(np.float32) - k[1, 2]) * z / k[1, 1]

        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)

        pts_world = (world_from_camera @ pts_cam)[:3].T.astype(np.float32)



        # World-Z floor: drop table-height points so an over-segmented mask (e.g. a

        # low Judge-anchor point-prompt that leaked the tabletop) cannot pull the

        # grasp down to the table. Fail-open if it would remove too much.

        table_z = getattr(self.config, "table_z_world", None)

        if table_z is not None:

            # Contract gate: a masked cloud with NO points meaningfully above the

            # table means grounding masked the tabletop itself (bare point-prompt



            # 'cream cheese', the anchor point-prompt segmented the table, and

            # GraspGen happily proposed z~0.006 table grasps that cuRobo refused).

            # Fail with a grounding diagnosis instead of handing garbage downstream.

            pop = float(getattr(self.config, "pc_min_object_pop_m", 0.0) or 0.0)

            if pop > 0.0:

                z_max = float(pts_world[:, 2].max())

                if z_max < float(table_z) + pop:

                    return self._fail(

                        "masked region is table-height only "

                        f"(max z={z_max:.3f} < table_z+{pop:.3f}); grounding "

                        "missed the object, not a grasp/motion failure"

                    )

            floor = float(table_z) + float(getattr(self.config, "pc_table_clearance_m", 0.0) or 0.0)

            keep = pts_world[:, 2] >= floor

            kept = int(keep.sum())

            # Thin-object guard: when the strict floor would eat most of the mask

            # AND the dropped points sit ABOVE the table-noise band, the mask is a

            # thin object (top near/below the clearance), not a table leak -- relax

            # the floor to the noise band so GraspGen sees the object body instead

            # of a 0.4 mm top sliver (depth_gate_v3: cream cheese kept 76 pts in a

            # z=[0.0200,0.0204] skin -> bogus grasps closing above the real box).

            noise = float(getattr(self.config, "pc_table_noise_m", 0.0) or 0.0)

            if noise > 0.0 and kept < 0.5 * len(pts_world):

                dropped_z = pts_world[~keep][:, 2]

                body = dropped_z >= (float(table_z) + noise)

                if int(body.sum()) > kept:

                    relaxed = float(table_z) + noise

                    keep_relaxed = pts_world[:, 2] >= relaxed

                    if self.config.verbose:

                        print(

                            f"[GraspGenDetector] pc z-floor relaxed to z={relaxed:.3f}: "

                            f"strict floor z={floor:.3f} kept {kept}/{len(pts_world)} but "

                            f"{int(body.sum())} dropped pts sit above the table-noise band "

                            "(thin object, not a table leak)"

                        )

                    floor, keep, kept = relaxed, keep_relaxed, int(keep_relaxed.sum())

            if kept >= int(self.config.min_valid_depth_pixels):

                if self.config.verbose and kept < len(pts_world):

                    print(

                        f"[GraspGenDetector] pc z-floor: kept {kept}/{len(pts_world)} "

                        f"pts above z={floor:.3f} (table_z={float(table_z):.3f})"

                    )

                pts_world = pts_world[keep]

            elif self.config.verbose:

                print(

                    f"[GraspGenDetector] pc z-floor skipped: only {kept} pts above "

                    f"z={floor:.3f} (< min {int(self.config.min_valid_depth_pixels)}); keeping unfiltered"

                )



        n = int(self.config.num_points)

        if n > 0 and len(pts_world) > n:

            rng = np.random.default_rng(int(self.config.seed))

            idx = rng.choice(len(pts_world), size=n, replace=False)

            pts_world = pts_world[idx]

        pts_dbg = np.ascontiguousarray(pts_world.astype(np.float32, copy=False))

        try:

            self.last_debug["point_cloud"] = {

                "shape": [int(v) for v in pts_dbg.shape],

                "sha256": hashlib.sha256(pts_dbg.tobytes()).hexdigest(),

                "xyz_min": [round(float(v), 6) for v in pts_dbg.min(axis=0)],

                "xyz_max": [round(float(v), 6) for v in pts_dbg.max(axis=0)],

                "xyz_mean": [round(float(v), 6) for v in pts_dbg.mean(axis=0)],

                "mask_pixels": int(mask_pixels),

                "valid_pixels": int(valid_pixels),

                "config_seed": int(self.config.seed),

            }

        except Exception:  # noqa: BLE001 - diagnostics only

            pass

        _record_zspan(self, pts_world, np)

        return pts_world, int(segmap_id)



    def _query_server(self, point_cloud, *, top_k: int):

        params = dict(

            num_grasps=int(self.config.num_grasps),

            grasp_threshold=float(self.config.grasp_threshold),

            topk_num_grasps=max(1, int(top_k)),

            min_grasps=int(self.config.min_grasps),

            max_tries=int(self.config.max_tries),

            remove_outliers=bool(self.config.remove_outliers),

            seed=int(self.config.seed),

        )

        self.last_debug["request_params"] = dict(params)

        if self.client_factory is not None:

            try:

                client = self.client_factory(self.config)

                if callable(client) and not hasattr(client, "infer"):

                    return client(point_cloud, **params)

                if hasattr(client, "__enter__"):

                    with client as c:

                        return c.infer(point_cloud, **params)

                return client.infer(point_cloud, **params)

            except Exception as e:  # noqa: BLE001

                return self._fail(f"GraspGen client failed: {type(e).__name__}: {e}")



        try:

            import msgpack  # type: ignore

            import msgpack_numpy  # type: ignore

            import zmq  # type: ignore

        except Exception as e:  # noqa: BLE001

            return self._fail(f"GraspGen ZMQ client deps unavailable: {type(e).__name__}: {e}")



        msgpack_numpy.patch()

        ctx = zmq.Context.instance()

        socket = ctx.socket(zmq.REQ)

        socket.setsockopt(zmq.RCVTIMEO, int(self.config.request_timeout_ms))

        socket.setsockopt(zmq.SNDTIMEO, int(self.config.request_timeout_ms))

        socket.setsockopt(zmq.LINGER, 0)

        try:

            socket.connect(f"tcp://{self.config.host}:{int(self.config.port)}")

            request = {"action": "infer", "point_cloud": point_cloud, **params}

            socket.send(msgpack.packb(request, use_bin_type=True))

            response = msgpack.unpackb(socket.recv(), raw=False)

        except Exception as e:  # noqa: BLE001

            return self._fail(f"GraspGen ZMQ request failed: {type(e).__name__}: {e}")

        finally:

            socket.close(linger=0)



        self.last_server_result = response if isinstance(response, dict) else None

        if not isinstance(response, dict):

            return self._fail(f"GraspGen response must be dict, got {type(response).__name__}")

        if response.get("error"):

            return self._fail(f"GraspGen server error: {response['error']}")

        return response.get("grasps", []), response.get("confidences", [])



    def _grasp_world_to_proposal(

        self, grasp_world, score: float, segmap_id: int, point_cloud

    ) -> GraspProposal:

        rot = [[float(v) for v in row[:3]] for row in grasp_world[:3]]

        quat = ContactGraspNetDetector._quat_wxyz_from_rot(rot)

        norm = math.sqrt(sum(v * v for v in quat)) or 1.0

        quat = [v / norm for v in quat]

        approach = [float(grasp_world[i][2]) for i in range(3)]

        anorm = math.sqrt(sum(a * a for a in approach)) or 1.0

        approach = [a / anorm for a in approach]

        d = float(self.config.gripper_base_to_contact_m)

        position = [float(grasp_world[i][3] + d * approach[i]) for i in range(3)]

        # Project the selected mask using the gripper closing-axis convention.

        import numpy as np  # type: ignore



        opening_axis = np.asarray(grasp_world, dtype=float)[:3, 0]

        axis_norm = float(np.linalg.norm(opening_axis))

        if not math.isfinite(axis_norm) or axis_norm <= 0.0:

            raise ValueError("GraspGen candidate has invalid gripper-X opening axis")

        opening_axis = opening_axis / axis_norm

        points = np.asarray(point_cloud, dtype=float)[:, :3]

        projections = points @ opening_axis

        predicted_width_m = float(projections.max() - projections.min())

        if not math.isfinite(predicted_width_m):

            raise ValueError("invalid predicted grasp width")

        return GraspProposal(

            position=position,

            quaternion=quat,

            score=float(score),

            source=f"{self.config.source}:{segmap_id}",

            predicted_width_m=predicted_width_m,

        )



    def detect_many(

        self,

        *,

        rgb,

        depth_m,

        intrinsics,

        object_hint,

        top_k: int = 15,

        target_xy=None,

        target_px=None,

        grounding=None,

        sam_text_prompts=None,

        **kwargs,

    ):

        self.last_error = None

        self.last_server_result = None

        self.last_debug = {}

        if grounding is not None:

            if getattr(grounding, "normalized_point", None) is not None:

                target_px = tuple(grounding.normalized_point)

            if getattr(grounding, "world_point", None) is not None and len(grounding.world_point) >= 2:

                target_xy = tuple(grounding.world_point[:2])

        pc_out = self._object_point_cloud_world(

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            target_xy=target_xy,

            target_px=target_px,

            sam_text_prompts=sam_text_prompts,

            scene_reference=kwargs.get("scene_reference"),

            exclude_masks=kwargs.get("exclude_masks"),

        )

        if pc_out is None:

            return []

        point_cloud, segmap_id = pc_out

        server_out = self._query_server(point_cloud, top_k=max(1, int(top_k)))

        if server_out is None:

            return []

        try:

            import numpy as np  # type: ignore



            grasps, confidences = server_out

            grasps_arr = np.asarray(grasps, dtype=np.float32)

            conf_arr = np.asarray(confidences, dtype=np.float32).reshape(-1)

            if grasps_arr.ndim != 3 or grasps_arr.shape[1:] != (4, 4):

                return self._fail(f"GraspGen grasps must be (N,4,4), got {grasps_arr.shape}") or []

            if len(conf_arr) != len(grasps_arr):

                return self._fail(

                    f"GraspGen confidence count {len(conf_arr)} != grasp count {len(grasps_arr)}"

                ) or []

            order = list(np.argsort(-conf_arr))

        except Exception as e:  # noqa: BLE001

            self._fail(f"GraspGen result parse failed: {type(e).__name__}: {e}")

            return []



        proposals: list[GraspProposal] = []

        for idx in order[: max(1, int(top_k))]:

            try:

                proposals.append(

                    self._grasp_world_to_proposal(

                        grasps_arr[int(idx)],

                        float(conf_arr[int(idx)]),

                        segmap_id,

                        point_cloud,

                    )

                )

            except Exception:

                continue

        if not proposals:

            self._fail("GraspGen returned no valid grasp candidates")

            return []

        # Preserve descending GraspGen confidence order.

        return proposals



    def detect(self, *, rgb, depth_m, intrinsics, object_hint, target_xy=None, target_px=None, sam_text_prompts=None, **kwargs):

        grasps = self.detect_many(

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            top_k=1,

            target_xy=target_xy,

            target_px=target_px,

            sam_text_prompts=sam_text_prompts,

        )

        return grasps[0] if grasps else None





# --------------------------------------------------------------------------- #

# Mask/depth top-down grasp (deterministic fallback; cap-x's OBB-pose analog)  #

# --------------------------------------------------------------------------- #

class MaskDepthGraspDetector:

    """Runtime perception (implements ``GraspDetector``) with **no grasp network**.



    The fingertip target is the masked RGB-D point cloud's world-XY centroid at a

    chosen world-z percentile; the wrist yaw is PCA of the mask in world XY, so the

    gripper pinches across the object's short axis from straight above. This is the

    portable analog of cap-x's ``get_object_pose`` (OBB) path and is exactly what

    rescues flat-lying produce that Contact-GraspNet cannot place contacts on.



    Same seam as :class:`ContactGraspNetDetector`: pass ``intrinsics['segmap']`` or

    ``intrinsics['mask']`` plus ``object_hint='seg:<id>'``, or wire a SAM3

    ``segmenter`` to ground free text. Without a mask source it fails closed.

    """



    def __init__(

        self,

        config: Optional[MaskDepthGraspConfig] = None,

        *,

        segmenter: Optional["Sam3Segmenter"] = None,

        pointer: Optional["MolmoPointer"] = None,

        positionfix_memory=None,

    ) -> None:

        self.config = config or MaskDepthGraspConfig()

        self.segmenter = segmenter

        # Optional live point source (Molmo2). Used only when SAM3 text grounding

        # yields no usable mask, replacing the sim's low ground-truth anchor with

        # a visible-surface point before the SAM3 point-prompt.

        self.pointer = pointer

        self.positionfix_memory = positionfix_memory

        self.last_error: Optional[str] = None

        self.last_debug: dict = {}



    def _fail(self, reason: str):

        self.last_error = reason

        if self.config.verbose:

            print(f"[MaskDepthGraspDetector] {reason}")

        return None



    @staticmethod

    def _mask_world_centroid_xy(np, mask, depth, K, world_from_camera, min_pixels: int):

        """World-XY centroid (median) of a single bool mask's valid depth pixels,

        or ``None`` if too few valid points."""

        ys, xs = np.nonzero(mask)

        if xs.size == 0:

            return None

        z = depth[ys, xs].astype(float)

        v = np.isfinite(z) & (z > 0.0)

        if int(v.sum()) < int(min_pixels):

            return None

        xs = xs[v].astype(float)

        ys = ys[v].astype(float)

        z = z[v]

        X = (xs - K[0, 2]) * z / K[0, 0]

        Y = (ys - K[1, 2]) * z / K[1, 1]

        cam = np.stack([X, Y, z, np.ones_like(z)], axis=0)

        w = (world_from_camera @ cam)[:3].T

        return np.median(w[:, :2], axis=0)



    def _scene_diff_component(

        self,

        np,

        depth,

        intrinsics,

        scene_reference,

        tx,

        ty,

        diag,

        prefail_xy=None,

        object_category=None,

        failure_category="hold_lost",

    ):

        """Find above-table geometry that appeared since the episode-start frame.

        Compare current depth with a reference RGB-D frame from the same fixed camera.
        Return None if required inputs are unavailable."""

        if scene_reference is None or intrinsics is None:

            return None

        try:

            import cv2  # type: ignore

        except Exception:  # noqa: BLE001

            return None

        try:

            positionfix_rule = None

            if self.positionfix_memory is not None and object_category:

                positionfix_rule = self.positionfix_memory.select_rule(

                    object_category=str(object_category),

                    failure_category=str(failure_category),

                    source_type="scene_diff",

                )

                if positionfix_rule is None:

                    return None

            min_pixels = int(positionfix_rule.min_component_pixels) if positionfix_rule else 40

            max_height = float(positionfix_rule.max_height_above_table_m) if positionfix_rule else 0.25

            max_area_frac = (

                float(positionfix_rule.max_component_area_frac)

                if positionfix_rule else float(self.config.point_disambig_max_area_frac)

            )

            ref_depth = scene_reference[1] if isinstance(scene_reference, (tuple, list)) else None

            if ref_depth is None:

                return None



            def _zmap(d_raw):

                d = np.asarray(d_raw, dtype=np.float32)

                if d.ndim == 3 and 1 in d.shape:

                    d = np.squeeze(d)

                k = intrinsics.get("K") or intrinsics.get("intrinsics")

                wfc = intrinsics.get("world_from_camera") or intrinsics.get("extrinsics")

                if k is None or wfc is None:

                    return None, None

                k = np.asarray(k, dtype=np.float32)

                wfc = np.asarray(wfc, dtype=np.float32)

                valid = np.isfinite(d) & (d > 0)

                vs, us = np.nonzero(valid)

                zs = d[vs, us]

                xs_ = (us.astype(np.float32) - k[0, 2]) * zs / k[0, 0]

                ys_ = (vs.astype(np.float32) - k[1, 2]) * zs / k[1, 1]

                zw = (wfc @ np.stack([xs_, ys_, zs, np.ones_like(zs)], axis=0))[2]

                zmap = np.full(d.shape, -np.inf, dtype=np.float32)

                zmap[vs, us] = zw

                return zmap, d



            z_now, d_now = _zmap(depth)

            z_ref, d_ref = _zmap(ref_depth)

            if z_now is None or z_ref is None or z_now.shape != z_ref.shape:

                return None

            tz_now = _scene_table_z(np, d_now, intrinsics)

            tz_ref = _scene_table_z(np, d_ref, intrinsics)

            if tz_now is None or tz_ref is None:

                return None

            pop = max(float(getattr(self.config, "point_prompt_min_pop_m", 0.004) or 0.004), 0.006)

            # Strip arm/gripper pixels (well above any tabletop object) BEFORE

            # connected components: an arm silhouette touching an object would

            # merge into its component and bloat or disqualify it. The

            # per-component median-z gate below stays as a second line.

            now_above = ((z_now >= tz_now + pop) & (z_now <= tz_now + max_height)).astype(np.uint8)

            ref_above = ((z_ref >= tz_ref + pop) & (z_ref <= tz_ref + max_height)).astype(np.uint8)

            # Dilate the reference occupancy so 1-2 px settling jitter of unmoved

            # objects does not leave halo slivers in the diff.

            kernel = np.ones((7, 7), dtype=np.uint8)

            ref_grown = cv2.dilate(ref_above, kernel, iterations=1)

            appeared = ((now_above > 0) & (ref_grown == 0)).astype(np.uint8)

            appeared = cv2.erode(appeared, np.ones((3, 3), np.uint8), iterations=1)

            if int(appeared.sum()) < min_pixels:

                return None

            n_lab, labels = cv2.connectedComponents(appeared, connectivity=8)

            H, W = appeared.shape

            max_area = max_area_frac * float(H * W)

            # Selection prior: the gripper held the target when it dropped, so

            # the pre-fail EE projection marks the drop site (object-agnostic).

            # Fall back to the largest area when no end-effector prior exists.

            ee_px = None

            if positionfix_rule is not None:

                if tx is None or ty is None:

                    return None

                # Current Judge/semantic anchor: recomputed from this rollout.

                ee_px = (float(tx), float(ty))

            elif prefail_xy is not None:

                try:

                    k = intrinsics.get("K") or intrinsics.get("intrinsics")

                    wfc = intrinsics.get("world_from_camera") or intrinsics.get("extrinsics")

                    k = np.asarray(k, dtype=np.float64)

                    cfw = np.linalg.inv(np.asarray(wfc, dtype=np.float64))

                    pw = np.array([float(prefail_xy[0]), float(prefail_xy[1]),

                                   float(tz_now) + 0.03, 1.0])

                    pc = cfw @ pw

                    if pc[2] > 0:

                        ee_px = (k[0, 0] * pc[0] / pc[2] + k[0, 2],

                                 k[1, 1] * pc[1] / pc[2] + k[1, 2])

                except Exception:  # noqa: BLE001 - prior is best-effort

                    ee_px = None

            best, best_key = None, None

            for lab in range(1, n_lab):

                comp = labels == lab

                area = int(comp.sum())

                if area < min_pixels or area > max_area:

                    continue

                # Physical object gate: tabletop objects live within ~25 cm of

                # the table; higher regions are the arm/gripper silhouette.

                med_z = float(np.median(z_now[comp]))

                if not (tz_now + pop <= med_z <= tz_now + max_height):

                    continue

                if ee_px is not None:

                    cys, cxs = np.nonzero(comp)

                    nd = float(np.min(np.hypot(cxs.astype(np.float32) - ee_px[0],

                                               cys.astype(np.float32) - ee_px[1])))

                    if positionfix_rule is not None and nd > float(positionfix_rule.anchor_gate_frac) * float(diag):

                        continue

                    key = nd  # smaller is better

                    better = best_key is None or key < best_key

                else:

                    key = -float(area)  # larger is better

                    better = best_key is None or key < best_key

                if better:

                    best, best_key = comp, key

            if best is not None and self.config.verbose and ee_px is not None:

                selector_name = (

                    "live evidence anchor" if positionfix_rule is not None

                    else "pre-fail EE"

                )

                print(

                    f"[MaskDepthGraspDetector] scene-diff pick nearest {selector_name} "

                    f"px=({ee_px[0]:.0f},{ee_px[1]:.0f}) nd={best_key:.1f}px "

                    f"area={int(best.sum())}px"

                )

            return best

        except Exception:  # noqa: BLE001

            return None



    def _table_pop_component(self, np, depth, intrinsics, tx, ty, diag):

        """Connected component of above-table geometry nearest the anchor.



        Returns a bool mask or None. Bounded: the component's nearest pixel must

        lie within ``point_disambig_max_frac * diag`` of the anchor, its area

        within [40 px, point_disambig_max_area_frac * frame]. Requires intrinsics

        (world back-projection); silently unavailable otherwise."""

        if intrinsics is None:

            return None

        try:

            import cv2  # type: ignore

        except Exception:  # noqa: BLE001

            return None

        try:

            d = np.asarray(depth, dtype=np.float32)

            if d.ndim == 3 and 1 in d.shape:

                d = np.squeeze(d)

            H, W = d.shape

            tz = _scene_table_z(np, d, intrinsics)

            if tz is None:

                return None

            # Per-pixel world z map. Computed inline (not via

            # _masked_points_world, which filters/subsamples and so loses the

            # pixel correspondence a component mask needs).

            k = intrinsics.get("K") or intrinsics.get("intrinsics")

            wfc = intrinsics.get("world_from_camera") or intrinsics.get("extrinsics")

            if k is None or wfc is None:

                return None

            k = np.asarray(k, dtype=np.float32)

            wfc = np.asarray(wfc, dtype=np.float32)

            valid = np.isfinite(d) & (d > 0)

            vs, us = np.nonzero(valid)

            zs = d[vs, us]

            xs = (us.astype(np.float32) - k[0, 2]) * zs / k[0, 0]

            ys = (vs.astype(np.float32) - k[1, 2]) * zs / k[1, 1]

            pts_cam = np.stack([xs, ys, zs, np.ones_like(zs)], axis=0)

            z_world = (wfc @ pts_cam)[2]

            zmap = np.full((H, W), -np.inf, dtype=np.float32)

            zmap[vs, us] = z_world

            pop = float(getattr(self.config, "point_prompt_min_pop_m", 0.004) or 0.004)

            above = (zmap >= tz + max(pop, 0.006)).astype(np.uint8)

            if int(above.sum()) < 40:

                return None

            n_lab, labels = cv2.connectedComponents(above, connectivity=8)

            max_area = float(self.config.point_disambig_max_area_frac) * float(H * W)

            best, best_nd = None, None

            for lab in range(1, n_lab):

                comp = labels == lab

                area = int(comp.sum())

                if area < 40 or area > max_area:

                    continue

                ys, xs = np.nonzero(comp)

                nd = float(np.min(np.hypot(xs.astype(np.float32) - tx,

                                           ys.astype(np.float32) - ty)))

                if nd > float(self.config.point_disambig_max_frac) * diag:

                    continue

                if best_nd is None or nd < best_nd:

                    best, best_nd = comp, nd

            return best

        except Exception:  # noqa: BLE001

            return None



    def _mask_pops_from_table(self, np, mask, depth, intrinsics=None) -> bool:

        """True iff the masked region sits nearer the camera than its surround.



        Depth-space test (no calibration needed): dilate the mask, compare the

        masked median depth against the surrounding ring's median. An object on

        the table pops toward the camera by at least its own height along the

        ray; a mask that matches its surround segmented the tabletop. Fail-open

        (True) when the test cannot run (no cv2, empty ring, bad depth) so this

        never blocks grounding on infra hiccups."""

        min_pop = float(getattr(self.config, "point_prompt_min_pop_m", 0.0) or 0.0)

        if min_pop <= 0.0:

            return True

        # Calibrated path: with intrinsics the test is exact -- the masked cloud

        # must contain SOMETHING meaningfully above the support plane (same

        # semantic as GraspGen's pc_min_object_pop_m gate). The ring-depth

        # heuristic below can be fooled by the smooth depth gradient across a

        # table under an oblique camera; world z cannot.

        if intrinsics is not None:

            try:

                pts = _masked_points_world(np, np.asarray(mask).astype(bool), depth, intrinsics)

                tz = _scene_table_z(np, depth, intrinsics)

                if pts is not None and len(pts) >= 8 and tz is not None:

                    return float(pts[:, 2].max()) >= tz + max(min_pop, 0.006)

            except Exception:  # noqa: BLE001

                pass

        try:

            import cv2  # type: ignore

        except Exception:  # noqa: BLE001

            return True

        try:

            m = np.asarray(mask).astype(np.uint8)

            if int(m.sum()) < 8:

                return True

            kernel = np.ones((9, 9), dtype=np.uint8)

            ring = (cv2.dilate(m, kernel, iterations=2) > 0) & (m == 0)

            d = np.asarray(depth, dtype=np.float32)

            if d.ndim == 3 and 1 in d.shape:

                d = np.squeeze(d)

            obj_d = d[m > 0]

            ring_d = d[ring]

            obj_d = obj_d[np.isfinite(obj_d) & (obj_d > 0)]

            ring_d = ring_d[np.isfinite(ring_d) & (ring_d > 0)]

            if obj_d.size < 8 or ring_d.size < 8:

                return True

            return float(np.median(ring_d) - np.median(obj_d)) >= min_pop

        except Exception:  # noqa: BLE001

            return True



    def _ground_mask_ordered(

        self,

        np,

        *,

        rgb,

        depth_m,

        object_hint,

        plan,

        grounding=None,

        target_px=None,

        target_xy=None,

        intrinsics=None,

        K=None,

        world_from_camera=None,

        scene_reference=None,

        extra_prompts=None,

        exclude_masks=None,

    ):

        """Run the declared grounding branches in order and return the first valid result."""

        attempts = []

        if self.segmenter is None or rgb is None:

            return None, [{"source": "setup", "ok": False,

                           "reason": "segmenter or rgb unavailable"}]

        depth = np.asarray(depth_m, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        H, W = int(depth.shape[0]), int(depth.shape[1])

        diag = float(np.hypot(W, H))

        anchor = None

        if target_px is not None:

            try:

                anchor = (float(target_px[0]) * W, float(target_px[1]) * H)

            except Exception:  # noqa: BLE001

                anchor = None



        def excluded(mask):

            if not exclude_masks:

                return False

            mm = np.asarray(mask, dtype=bool)

            area = float(mm.sum())

            if area <= 0:

                return False

            for ex in exclude_masks:

                ee = np.asarray(ex, dtype=bool)

                if ee.shape == mm.shape:

                    overlap = float((mm & ee).sum()) / max(

                        1.0, min(area, float(ee.sum()))

                    )

                    if overlap > 0.5:

                        return True

            return False



        def accept(source, mask, provenance, reason, point=None):

            if mask is None:

                attempts.append({"source": source, "ok": False, "reason": reason})

                return None

            mm = np.asarray(mask, dtype=bool)

            if mm.shape != (H, W) or not mm.any():

                attempts.append({"source": source, "ok": False,

                                 "reason": "empty or shape-mismatched mask"})

                return None

            if excluded(mm):

                attempts.append({"source": source, "ok": False,

                                 "reason": "overlaps identity-rejected mask"})

                return None

            ys, xs = np.nonzero(mm)

            centroid = [float(xs.mean()) / W, float(ys.mean()) / H]

            attempts.append({

                "source": source, "ok": True, "reason": reason,

                "provenance": provenance,

                "mask_centroid_normalized": centroid,

            })

            self.segmenter.last_grounding_via = provenance

            if grounding is not None:

                grounding.mode = source

                grounding.provenance = provenance

                grounding.attempts = list(attempts)

                # Observable mask centroid, suitable for evaluator-only

                # grounding accuracy. Never simulator-derived.

                grounding.normalized_point = centroid

            return mm



        prompts = []

        for prompt in extra_prompts or []:

            prompt = str(prompt).strip()

            if prompt and prompt not in prompts:

                prompts.append(prompt)

        catalog = getattr(self, "object_catalog", None)

        if catalog is not None:

            for prompt in catalog.prompts_for(object_hint or ""):

                if prompt and prompt not in prompts:

                    prompts.append(prompt)

        for prompt in [

            resolve_disambig_prompt(self.config.disambig_prompt, object_hint),

            *Sam3Segmenter._prompt_fallbacks(object_hint or ""),

        ]:

            prompt = str(prompt or "").strip()

            if prompt and prompt not in prompts:

                prompts.append(prompt)



        for source in list(plan or []):

            mask = None

            provenance = source

            reason = ""

            point = None

            try:

                if source == "task_target":

                    # Text only: Molmo belongs exclusively to point_model.

                    mask = _segment_text_with_retry(

                        self.segmenter, rgb, object_hint=object_hint,

                        extra_prompts=extra_prompts, verbose=self.config.verbose,

                    )

                    provenance = getattr(

                        self.segmenter, "last_grounding_via", None

                    ) or "task-target-text"

                    reason = "task-target text mask"

                elif source == "point_model":

                    if self.pointer is None:

                        reason = "point model unavailable"

                    else:

                        for phrase in prompts[:4]:

                            point = self.pointer.point_at(rgb, phrase)

                            if point is not None:

                                break

                        if point is None:

                            reason = (

                                "point model returned no point: "

                                + str(getattr(self.pointer, "last_error", ""))

                            )

                        else:

                            mask = self.segmenter.segment_at_point(

                                rgb, point_px=(float(point[0]), float(point[1]))

                            )

                            provenance = "point-model"

                            reason = f"point-model mask prompt={phrase!r}"

                elif source == "scene_diff":

                    if self.positionfix_memory is not None and anchor is None:

                        mask = None

                        reason = "positionfix requires a current live evidence anchor"

                    else:

                        tx, ty = anchor if anchor is not None else (0.5 * W, 0.5 * H)

                        mask = self._scene_diff_component(

                            np, depth, intrinsics, scene_reference, tx, ty, diag,

                            prefail_xy=target_xy, object_category=object_hint,

                            failure_category="hold_lost",

                        )

                    provenance = "scene-diff"

                    reason = reason or "appeared component from pre-failure reference"

                elif source == "judge_point":

                    if anchor is None:

                        reason = "judge point unavailable"

                    else:

                        tx, ty = anchor

                        max_area = float(

                            self.config.point_disambig_max_area_frac

                        ) * float(H * W)

                        jit = max(4.0, 0.015 * diag)

                        for jx, jy in (

                            (0.0, 0.0), (jit, 0.0), (-jit, 0.0),

                            (0.0, jit), (0.0, -jit),

                        ):

                            px = (

                                min(W - 1.0, max(0.0, tx + jx)),

                                min(H - 1.0, max(0.0, ty + jy)),

                            )

                            candidate = self.segmenter.segment_at_point(

                                rgb, point_px=px

                            )

                            if candidate is None:

                                continue

                            if float(np.asarray(candidate).sum()) > max_area:

                                continue

                            if not self._mask_pops_from_table(

                                np, candidate, depth, intrinsics=intrinsics

                            ):

                                continue

                            mask, point = candidate, px

                            break

                        provenance = "judge-point-prompt"

                        reason = "bounded point-prompt with depth sanity"

                elif source == "recent_contact":

                    if target_xy is None or K is None or world_from_camera is None:

                        reason = "recent-contact world prior unavailable"

                    else:

                        mask = self._disambiguate_mask(

                            np, rgb=rgb, depth_m=depth, K=K,

                            world_from_camera=world_from_camera,

                            object_hint=object_hint, target_xy=target_xy,

                        )

                        provenance = "recent-contact-world"

                        reason = "nearest text instance in world XY"

                elif source == "multiview":

                    reason = "multiview grounding backend unavailable"

                else:

                    reason = f"unknown grounding source {source!r}"

            except Exception as exc:  # noqa: BLE001

                reason = f"{type(exc).__name__}: {exc}"

                mask = None

            chosen = accept(source, mask, provenance, reason, point=point)

            if chosen is not None:

                return chosen, attempts



        self.segmenter.last_grounding_via = "failed"

        if grounding is not None:

            grounding.mode = "failed"

            grounding.provenance = "ordered-plan-exhausted"

            grounding.attempts = list(attempts)

        return None, attempts



    def _disambiguate_mask_px(self, np, *, rgb, depth_m, object_hint, target_px, intrinsics=None, scene_reference=None, extra_prompts=None, target_xy=None, exclude_masks=None):

        """Point-anchored grounding cascade (cap-x aligned).



        ① SAM3 text on ``object_hint``: pick the candidate that CONTAINS the

        anchor (point-in-mask); else, if the nearest ``object_hint`` mask has its

        boundary within ``point_disambig_max_frac`` of the anchor, use that FULL

        mask (``text-near-surface``). ② SAM3 point-prompt at ``target_px``.

        ③ fail closed.



        Rationale: the sim's ground-truth anchor is the object's 3D *centre*

        projected to pixels, which under an oblique camera lands just *below* the

        visible body mask (bottle base / table seam), so it is rarely strictly

        inside the mask. A tiny point-prompt box there segments a table-height

        sliver -> near-zero-Z point cloud -> table-hugging grasps. Preferring the

        nearby FULL object-hint mask keeps the whole body (correct Z) while still

        never falling back to a generic ``'object'`` enumeration or a nearest-

        centroid pick among distractors (the wrong-ketchup-neighbour bug).

        """

        if self.segmenter is None or rgb is None or target_px is None:

            return None

        try:

            depth = np.asarray(depth_m, dtype=np.float32)

            if depth.ndim == 3 and 1 in depth.shape:

                depth = np.squeeze(depth)

            H, W = int(depth.shape[0]), int(depth.shape[1])

        except Exception:  # noqa: BLE001

            return None

        try:

            tx = float(target_px[0]) * W

            ty = float(target_px[1]) * H

        except (TypeError, ValueError, IndexError):

            return None

        itx, ity = int(round(tx)), int(round(ty))

        itx = min(W - 1, max(0, itx))

        ity = min(H - 1, max(0, ity))



        def _excluded(mk):

            """True when ``mk`` mostly overlaps an identity-rejected mask, so

            the retry after a verify_object_identity mismatch tries something

            ELSE instead of deterministically re-picking the same object."""

            if not exclude_masks:

                return False

            mkb = np.asarray(mk, dtype=bool)

            a = float(mkb.sum())

            if a <= 0:

                return False

            for ex in exclude_masks:

                exb = np.asarray(ex, dtype=bool)

                if exb.shape != mkb.shape:

                    continue

                inter = float((mkb & exb).sum())

                if inter / max(1.0, min(a, float(exb.sum()))) > 0.5:

                    return True

            return False



        # ① Text grounding (adaptive cascade): object-hint prompts only (no

        # generic 'object'). Run a STRICT primary pass first, then a RELAXED

        # retry pass only if the primary found nothing usable -- keeps everyday

        # grounding high-precision while recovering weak-but-correct detections

        # (small LIBERO objects score ~0.3 and dip under the primary gate).

        prompts = []

        # Priority: reflector-provided prompts > catalog packaging phrases >

        # derived hint prompt > generic fallbacks (first usable wins).

        for p in extra_prompts or []:

            p = str(p).strip()

            if p and p not in prompts:

                prompts.append(p)

        catalog = getattr(self, "object_catalog", None)

        if catalog is not None:

            for p in catalog.prompts_for(object_hint or ""):

                if p and p not in prompts:

                    prompts.append(p)

        derived = resolve_disambig_prompt(self.config.disambig_prompt, object_hint)

        if derived and derived not in prompts:

            prompts.append(derived)

        for p in Sam3Segmenter._prompt_fallbacks(object_hint or ""):

            if p and p not in prompts:

                prompts.append(p)



        diag = float(np.hypot(W, H))

        max_area = float(self.config.point_disambig_max_area_frac) * float(H * W)

        used_prompt = None

        near_stash = None  # demoted near-surface candidate: (mask, tag, nd, area)





        def _text_pick(thr):

            """Enumerate object-hint candidates at score threshold ``thr`` (None =

            config primary) and apply the point-anchored disambiguation. Returns a

            bool mask or None; records the winning ``used_prompt``."""

            nonlocal used_prompt, near_stash

            cands = None

            for p in prompts:

                cands = self.segmenter.segment_candidates(rgb, prompt=p, threshold=thr)

                if cands:

                    used_prompt = p

                    break

            if not cands:

                return None

            tag = "text" if thr is None else f"text@{float(thr):.2f}"

            entries = []

            for mk in cands:

                mk = np.asarray(mk).astype(bool)

                if mk.shape != (H, W):

                    continue

                ys, xs = np.nonzero(mk)

                if ys.size == 0:

                    continue

                nd = float(np.min(np.hypot(xs.astype(np.float32) - tx,

                                           ys.astype(np.float32) - ty)))

                entries.append({"mask": mk, "area": int(ys.size),

                                "in": bool(mk[ity, itx]), "nd": nd})

            gated = [e for e in entries if e["area"] <= max_area] or entries

            # Reject table-flat text masks using the same geometric check as point-prompt masks.

            gated = [

                e for e in gated

                if self._mask_pops_from_table(np, e["mask"], depth, intrinsics=intrinsics)

            ]

            gated = [e for e in gated if not _excluded(e["mask"])]

            containing = [e for e in gated if e["in"]]

            if containing:

                pick = min(containing, key=lambda e: e["area"])

                self.segmenter.last_grounding_via = tag

                if self.config.verbose:

                    print(

                        f"[MaskDepthGraspDetector] grounding_via={tag} "

                        f"prompt={used_prompt!r} via=point-in-mask "

                        f"pick area={pick['area']}px anchor=({tx:.0f},{ty:.0f})"

                    )

                return pick["mask"]

            # No mask contains the anchor: STASH the nearest full object-hint

            # mask instead of accepting it here. A stale anchor (object moved

            # since the judge fired) makes nearest-mask pick a look-alike



            # masked the milk carton for 'cream cheese'), so Molmo pointing

            # gets priority and this candidate is only used if Molmo fails.

            if gated:

                near = min(gated, key=lambda e: e["nd"])

                if near["nd"] <= float(self.config.point_disambig_max_frac) * diag:

                    if near_stash is None or near["nd"] < near_stash[2]:

                        near_stash = (near["mask"], tag, near["nd"], near["area"])

                    if self.config.verbose:

                        print(

                            f"[MaskDepthGraspDetector] {tag} near-surface candidate "

                            f"stashed (demoted below Molmo): prompt={used_prompt!r} "

                            f"nd={near['nd']:.1f}px area={near['area']}px"

                        )

                elif self.config.verbose:

                    print(

                        f"[MaskDepthGraspDetector] {tag} candidates too far "

                        f"(nearest nd={near['nd']:.1f}px > "

                        f"{self.config.point_disambig_max_frac:.2f}*diag)"

                    )

            return None



        # Strict primary pass, then relaxed retry (only if a lower retry threshold

        # is configured on the segmenter). None => the segmenter's config primary.

        thresholds = [None]

        seg_cfg = getattr(self.segmenter, "config", None)

        rt = float(getattr(seg_cfg, "retry_threshold", -1.0))

        primary = float(getattr(seg_cfg, "threshold", 0.5))

        if rt >= 0.0 and rt < primary:

            thresholds.append(rt)

        for ti, thr in enumerate(thresholds):

            mask = _text_pick(thr)

            if mask is not None:

                return mask

            if self.config.verbose and ti + 1 < len(thresholds):

                print(

                    "[MaskDepthGraspDetector] text grounding empty at "

                    f"{'primary' if thr is None else f'{thr:.2f}'}; "

                    f"retrying at {thresholds[ti + 1]:.2f}"

                )



        # ② Molmo point source: when text grounding gave no usable mask, ask

        # Molmo2 to point at the object and segment there. Molmo points at the

        # visible surface, so unlike the sim's centre-projection anchor the

        # point-prompt box lands on the body (not the table seam).

        if self.pointer is not None:

            # Molmo yields no point for raw brand names ('cream cheese') but

            # works with packaging phrases -- enumerate the same merged prompt

            # list (reflector > catalog > hint), first point wins (cap-x-style

            # synonym enumeration; bounded to keep takeover latency low).

            molmo_phrases = []

            for cand in [*prompts, (object_hint or "").strip()]:

                cand = str(cand).strip()

                if cand and cand not in molmo_phrases:

                    molmo_phrases.append(cand)

            mpt = None

            obj_phrase = ""

            for obj_phrase in molmo_phrases[:4]:

                mpt = self.pointer.point_at(rgb, obj_phrase)

                if mpt is not None:

                    break

            if mpt is not None:

                mx, my = float(mpt[0]), float(mpt[1])

                # Consistency gate: the reflector/Judge anchor is authoritative for

                # WHICH instance to grasp. If Molmo points farther than the gate

                # from the anchor it is pointing at a look-alike neighbour -- reject

                # it and fall through to the anchor point-prompt (③) rather than

                # segmenting the wrong object.

                gate_frac = float(getattr(self.config, "molmo_anchor_gate_frac", 0.0) or 0.0)

                md = float(np.hypot(mx - tx, my - ty))

                diag = float(np.hypot(W, H))

                if gate_frac > 0.0 and md > gate_frac * diag:

                    if self.config.verbose:

                        print(

                            f"[MaskDepthGraspDetector] molmo-point REJECTED "

                            f"obj={obj_phrase!r} molmo_px=({mx:.0f},{my:.0f}) "

                            f"nd={md:.1f}px > {gate_frac:.2f}*diag={gate_frac * diag:.0f}px "

                            f"from anchor=({tx:.0f},{ty:.0f}); falling back to anchor point-prompt"

                        )

                    mpt = None

            if mpt is not None:

                mx, my = float(mpt[0]), float(mpt[1])

                mask = self.segmenter.segment_at_point(rgb, point_px=(mx, my))

                if mask is not None and _excluded(mask):

                    if self.config.verbose:

                        print(

                            "[MaskDepthGraspDetector] molmo mask overlaps an "

                            "identity-rejected mask; skipping"

                        )

                    mask = None

                if mask is not None:

                    self.segmenter.last_grounding_via = "molmo-point"

                    if self.config.verbose:

                        print(

                            f"[MaskDepthGraspDetector] grounding_via=molmo-point "

                            f"obj={obj_phrase!r} molmo_px=({mx:.0f},{my:.0f}) "

                            f"px={int(mask.sum())} (gt_anchor=({tx:.0f},{ty:.0f}))"

                        )

                    return mask

            elif self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] molmo point failed "

                    f"({getattr(self.pointer, 'last_error', None)}); "

                    f"falling back to gt-anchor point-prompt"

                )



        # Use scene difference as the displacement fallback after semantic grounding.

        diag0 = float(np.hypot(W, H))

        mask = self._scene_diff_component(

            np, depth, intrinsics, scene_reference, tx, ty, diag0,

            prefail_xy=target_xy,

            object_category=object_hint,

            failure_category="hold_lost",

        )

        if mask is not None and _excluded(mask):

            if self.config.verbose:

                print(

                    "[MaskDepthGraspDetector] scene-diff pick overlaps an "

                    "identity-rejected mask; skipping"

                )

            mask = None

        if mask is not None:

            self.segmenter.last_grounding_via = "scene-diff"

            if self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] grounding_via=scene-diff "

                    f"px={int(mask.sum())} (anchor=({tx:.0f},{ty:.0f}) unused)"

                )

            return mask



        # ②.5 Demoted near-surface acceptance: Molmo had no answer, fall back

        # to the nearest object-hint text mask (kept for the fresh-anchor case

        # where the centre projection lands just below the body mask).

        if near_stash is not None:

            n_mask, n_tag, n_nd, n_area = near_stash

            self.segmenter.last_grounding_via = f"{n_tag}-near-surface"

            if self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] grounding_via={n_tag}-near-surface "

                    f"(post-molmo fallback) nd={n_nd:.1f}px area={n_area}px "

                    f"anchor=({tx:.0f},{ty:.0f})"

                )

            return n_mask



        # ③ Point-prompt at the (ground-truth) anchor pixel.

        # Validated by a depth pop-out test: an object sticks out of the table,

        # so its masked depth must be nearer the camera than the surrounding

        # ring. A mask that fails the test segmented the tabletop (the anchor

        # landed beside a thin object) -- retry a small bounded jitter around

        # the anchor before failing closed, instead of feeding a table-height



        jit = max(4.0, 0.015 * diag)

        for jx, jy in ((0.0, 0.0), (jit, 0.0), (-jit, 0.0), (0.0, jit), (0.0, -jit)):

            px_x = min(W - 1.0, max(0.0, tx + jx))

            px_y = min(H - 1.0, max(0.0, ty + jy))

            mask = self.segmenter.segment_at_point(rgb, point_px=(px_x, px_y))

            if mask is None:

                continue

            # Same area sanity the text path applies: a mask covering a large

            # fraction of the frame is the tabletop, and it defeats the pop

            # test below because a huge region always CONTAINS some standing



            # passed and GraspGen grasped a random 71-deg side pose in it).

            if float(np.asarray(mask).sum()) > max_area:

                if self.config.verbose:

                    print(

                        f"[MaskDepthGraspDetector] point-prompt at "

                        f"({px_x:.0f},{px_y:.0f}) mask too large "

                        f"({int(np.asarray(mask).sum())}px > {max_area:.0f}px); jittering"

                    )

                continue

            if _excluded(mask):

                continue

            if not self._mask_pops_from_table(np, mask, depth, intrinsics=intrinsics):

                if self.config.verbose:

                    print(

                        f"[MaskDepthGraspDetector] point-prompt at "

                        f"({px_x:.0f},{px_y:.0f}) segmented a table-flat region "

                        f"(px={int(mask.sum())}); jittering"

                    )

                continue

            if self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] grounding_via=point-prompt "

                    f"anchor=({px_x:.0f},{px_y:.0f}) px={int(mask.sum())}"

                )

            return mask



        # 3.5 Geometric table-pop fallback (classic plane-removal clustering):

        # when text grounding has no mask and the anchor point-prompt only finds

        # the tabletop, segment DIRECTLY from geometry -- back-project the full

        # frame, keep pixels meaningfully above the support plane, and take the

        # connected component nearest the anchor. Object-agnostic: no SAM, no

        # names, just 'something sticking out of the table near the point'.

        mask = self._table_pop_component(np, depth, intrinsics, tx, ty, diag)

        if mask is not None and _excluded(mask):

            mask = None

        if mask is not None:

            self.segmenter.last_grounding_via = "table-pop"

            if self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] grounding_via=table-pop "

                    f"anchor=({tx:.0f},{ty:.0f}) px={int(mask.sum())}"

                )

            return mask



        # ④ Fail closed.

        self.segmenter.last_grounding_via = "failed"

        if self.config.verbose:

            print(

                f"[MaskDepthGraspDetector] grounding_via=failed "

                f"text_prompts={prompts!r} anchor=({tx:.0f},{ty:.0f})"

            )

        return None



    def _disambiguate_mask(self, np, *, rgb, depth_m, K, world_from_camera, object_hint, target_xy):

        """Enumerate every instance matching the prompt and return the bool mask

        whose world-XY centroid is nearest ``target_xy`` (the VLA's pre-failure

        intent). Returns ``None`` to signal the caller to use the plain single-

        mask path (no segmenter, no candidates, or no pick within tolerance)."""

        if self.segmenter is None or rgb is None:

            return None

        prompt = resolve_disambig_prompt(self.config.disambig_prompt, object_hint)

        cands = self.segmenter.segment_candidates(rgb, prompt=prompt)

        if not cands:

            return None

        try:

            depth = np.asarray(depth_m, dtype=np.float32)

            if depth.ndim == 3 and 1 in depth.shape:

                depth = np.squeeze(depth)

        except Exception:  # noqa: BLE001

            return None

        prior = np.asarray(target_xy, dtype=float).reshape(2)

        best = None  # (dist, mask, centroid)

        for mk in cands:

            mk = np.asarray(mk).astype(bool)

            if mk.shape != depth.shape:

                continue

            c = self._mask_world_centroid_xy(

                np, mk, depth, K, world_from_camera,

                int(self.config.min_valid_depth_pixels),

            )

            if c is None:

                continue

            d = float(np.hypot(c[0] - prior[0], c[1] - prior[1]))

            if best is None or d < best[0]:

                best = (d, mk, c)

        if best is None:

            return None

        if best[0] > float(self.config.disambig_max_dist_m):

            if self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] disambig: nearest instance d={best[0]:.3f}m "

                    f"> max {self.config.disambig_max_dist_m:.3f}m; using plain mask"

                )

            return None

        if self.config.verbose:

            print(

                f"[MaskDepthGraspDetector] disambig prompt={prompt!r} "

                f"n={len(cands)} pick d={best[0]:.3f}m "

                f"centroid=({best[2][0]:+.3f},{best[2][1]:+.3f}) prior=({prior[0]:+.3f},{prior[1]:+.3f})"

            )

        return best[1]



    def _segment_text_with_retry(self, rgb, *, object_hint: str, extra_prompts=None):

        """Plain task-target text grounding with adaptive threshold + Molmo.



        Generic head-noun fallbacks live in ``Sam3Segmenter``; object-specific

        prompt lists come from reflector ``StrategySpec.params['sam_text_prompts']``.

        """

        if self.segmenter is None:

            return None

        mask = _segment_text_with_retry(

            self.segmenter,

            rgb,

            object_hint=object_hint,

            extra_prompts=extra_prompts,

            verbose=self.config.verbose,

        )

        if mask is not None:

            return mask

        if self.pointer is not None:

            obj_phrase = (object_hint or "").strip()

            mpt = self.pointer.point_at(rgb, obj_phrase) if obj_phrase else None

            if mpt is not None:

                mx, my = float(mpt[0]), float(mpt[1])

                mask = self.segmenter.segment_at_point(rgb, point_px=(mx, my))

                if mask is not None:

                    self.segmenter.last_grounding_via = "molmo-point"

                    if self.config.verbose:

                        print(

                            f"[MaskDepthGraspDetector] grounding_via=molmo-point "

                            f"obj={obj_phrase!r} molmo_px=({mx:.0f},{my:.0f}) "

                            f"px={int(mask.sum())}"

                        )

                    return mask

            elif self.config.verbose:

                print(

                    f"[MaskDepthGraspDetector] molmo point failed "

                    f"({getattr(self.pointer, 'last_error', None)}); task_target fail-closed"

                )

        return None



    def detect(self, *, rgb, depth_m, intrinsics, object_hint, target_xy=None, target_px=None, sam_text_prompts=None, **kwargs):

        detect_started = time.perf_counter()

        self.last_error = None

        self.last_perceive_latency = None

        self.last_visual_selection_evidence = {}

        self.last_debug = {}

        try:

            import numpy as np  # type: ignore

        except Exception:

            return self._fail("numpy unavailable")



        # --- intrinsics + extrinsics ------------------------------------- #

        try:

            if "K" in intrinsics:

                k = intrinsics["K"]

            elif "intrinsics" in intrinsics:

                k = intrinsics["intrinsics"]

            elif {"fx", "fy", "ppx", "ppy"}.issubset(intrinsics):

                k = [

                    [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],

                    [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],

                    [0.0, 0.0, 1.0],

                ]

            else:

                raise ValueError("intrinsics must include K/intrinsics or fx/fy/ppx/ppy")

            K = np.asarray(k, dtype=float).reshape(3, 3)

            # Explicit None checks (not ``a or b``): values may be numpy arrays,

            # whose truthiness is ambiguous.

            wfc = intrinsics.get("world_from_camera")

            if wfc is None:

                wfc = intrinsics.get("extrinsics")

            if wfc is None:

                wfc = self.config.world_from_camera

            if wfc is None:

                return self._fail("missing world_from_camera/extrinsics")

            world_from_camera = np.asarray(wfc, dtype=float).reshape(4, 4)

        except Exception as e:  # noqa: BLE001

            return self._fail(f"bad grasp detector inputs: {type(e).__name__}: {e}")



        # --- segmap / mask (+ SAM3 fallback for free text) --------------- #

        segmap = intrinsics.get("segmap")

        if segmap is None and "mask" in intrinsics:

            segmap = intrinsics["mask"]

        match = re.search(r"(?:seg(?:ment)?|id)?\s*:?\s*(\d+)", object_hint or "")

        segmap_id = int(match.group(1)) if match else 1

        if segmap is None:

            if self.segmenter is None:

                return self._fail("no segmap/mask provided and no SAM3 segmenter wired")

            if rgb is None:

                return self._fail("no segmap/mask provided and no rgb for SAM3 grounding")

            if re.fullmatch(r"\s*(?:seg(?:ment)?|id)\s*:?\s*\d+\s*", object_hint or "", re.IGNORECASE):

                return self._fail(f"object_hint {object_hint!r} expects a provided mask/segmap")

            chosen = None

            # Resolve target identity from reference visual memory and current image evidence.

            visual_selector = getattr(self, "visual_object_selector", None)

            if visual_selector is not None:

                try:

                    chosen = visual_selector.select_mask(

                        object_hint, rgb, self.segmenter, prompt="object",

                        scene_bundle_identity=kwargs.get("scene_bundle_identity"),

                    )

                except TypeError:

                    # Support selectors that do not accept the optional grounding-plan arguments.

                    chosen = visual_selector.select_mask(

                        object_hint, rgb, self.segmenter, prompt="object"

                    )

                if chosen is None:

                    return self._fail(

                        getattr(visual_selector, "last_error", None)

                        or "MEMORY_SOM_SELECTION_FAILURE"

                    )

                self.segmenter.last_grounding_via = "registry-memory-som"

                self.last_visual_selection_evidence = dict(

                    getattr(visual_selector, "last_evidence", {}) or {}

                )

                self.last_debug["semantic_memory"] = dict(

                    self.last_visual_selection_evidence

                )

            grounding = kwargs.get("grounding")

            ordered_plan = (

                [] if chosen is not None

                else list(getattr(grounding, "plan", None) or [])

            )

            if ordered_plan:

                chosen, attempts = self._ground_mask_ordered(

                    np, rgb=rgb, depth_m=depth_m, object_hint=object_hint,

                    plan=ordered_plan, grounding=grounding,

                    target_px=target_px, target_xy=target_xy,

                    intrinsics=intrinsics, K=K,

                    world_from_camera=world_from_camera,

                    scene_reference=kwargs.get("scene_reference"),

                    extra_prompts=sam_text_prompts,

                    exclude_masks=kwargs.get("exclude_masks"),

                )

                self.last_grounding_attempts = attempts

            elif self.config.spatial_disambig and target_px is not None:

                # Legacy adaptive cascade only for callers without a declared plan.

                chosen = self._disambiguate_mask_px(

                    np, rgb=rgb, depth_m=depth_m,

                    object_hint=object_hint, target_px=target_px,

                    intrinsics=intrinsics,

                    scene_reference=kwargs.get("scene_reference"),

                    extra_prompts=sam_text_prompts,

                    target_xy=target_xy,

                    exclude_masks=kwargs.get("exclude_masks"),

                )

            if (

                not ordered_plan and chosen is None

                and self.config.spatial_disambig and target_xy is not None

            ):

                chosen = self._disambiguate_mask(

                    np, rgb=rgb, depth_m=depth_m, K=K,

                    world_from_camera=world_from_camera,

                    object_hint=object_hint, target_xy=target_xy,

                )

            if chosen is not None:

                segmap = chosen.astype(np.int32)

            elif ordered_plan:

                via = getattr(self.segmenter, "last_grounding_via", None) or "failed"

                return self._fail(

                    f"ordered grounding failed (via={via}, attempts={self.last_grounding_attempts})"

                )

            elif target_px is not None:

                via = getattr(self.segmenter, "last_grounding_via", None) or "failed"

                return self._fail(f"grounding failed (via={via}): {self.segmenter.last_error}")

            else:

                segmap = self._segment_text_with_retry(

                    rgb, object_hint=object_hint, extra_prompts=sam_text_prompts,

                )

            if segmap is None:

                return self._fail(f"SAM3 grounding failed: {self.segmenter.last_error}")

            segmap_id = 1

            try:  # persisted for the pre-grasp identity check (fails open)

                self.segmenter.last_grounding_mask = np.asarray(segmap) > 0

                self.segmenter.last_grounding_rgb = rgb

            except Exception:  # noqa: BLE001

                pass



        depth_pca_started = time.perf_counter()

        # --- masked point cloud in world frame --------------------------- #

        try:

            depth = np.asarray(depth_m, dtype=np.float32)

            if depth.ndim == 3 and 1 in depth.shape:

                depth = np.squeeze(depth)

            seg = np.asarray(segmap)

            if seg.ndim == 3 and 1 in seg.shape:

                seg = np.squeeze(seg)

            if depth.ndim != 2 or seg.ndim != 2 or seg.shape != depth.shape:

                raise ValueError(f"depth {depth.shape} / segmap {seg.shape} mismatch")

            mask = ContactGraspNetDetector._dilate_mask(

                seg.astype(np.int32) == int(segmap_id), int(self.config.mask_dilation_px)

            )

            mask = ContactGraspNetDetector._erode_mask(mask, int(self.config.mask_erode_px))

        except Exception as e:  # noqa: BLE001

            return self._fail(f"invalid RGB-D segmentation packet: {type(e).__name__}: {e}")



        ys, xs = np.nonzero(mask)

        if xs.size == 0:

            return self._fail("target mask is empty")

        z = depth[ys, xs]

        valid = np.isfinite(z) & (z > 0.0)

        if int(valid.sum()) < int(self.config.min_valid_depth_pixels):

            return self._fail(f"target has too few valid depth pixels: {int(valid.sum())}")

        xs = xs[valid].astype(float)

        ys = ys[valid].astype(float)

        z = z[valid].astype(float)



        X = (xs - K[0, 2]) * z / K[0, 0]

        Y = (ys - K[1, 2]) * z / K[1, 1]

        pts_cam = np.stack([X, Y, z, np.ones_like(z)], axis=0)

        pts_w = (world_from_camera @ pts_cam)[:3].T



        # Remove points within the support-plane noise band before computing mask
        # centroid and height statistics. Retain points above that band.

        noise = float(getattr(self.config, "mask_table_noise_m", 0.0) or 0.0)

        if noise > 0.0:

            tz = _scene_table_z(

                np, depth,

                {"K": np.asarray(K).tolist(),

                 "world_from_camera": np.asarray(world_from_camera).tolist()},

            )

            if tz is not None:

                keep = pts_w[:, 2] >= (tz + noise)

                kept = int(keep.sum())

                if kept >= int(self.config.min_valid_depth_pixels) and kept < len(pts_w):

                    if self.config.verbose:

                        print(

                            f"[MaskDepthGraspDetector] table-noise floor: kept "

                            f"{kept}/{len(pts_w)} pts above z={tz + noise:.3f}"

                        )

                    pts_w = pts_w[keep]



        xy = pts_w[:, :2]

        # Centroid points: optionally restrict to the object TOP band so an angled

        # overhead camera's front-face pixels don't drag the grasp XY off-footprint

        # (see center_top_band_m docs). Falls back to all points if the band is too

        # sparse.

        xy_c = xy

        if float(self.config.center_top_band_m) > 0.0:

            z_all = pts_w[:, 2]

            z_hi = float(np.percentile(z_all, float(self.config.center_top_band_percentile)))

            top_sel = z_all >= (z_hi - float(self.config.center_top_band_m))

            if int(np.count_nonzero(top_sel)) >= int(self.config.center_top_band_min_pixels):

                xy_c = xy[top_sel]

                if self.config.verbose:

                    print(

                        "[MaskDepthGraspDetector] top-band centroid: "

                        f"z_hi={z_hi:.4f} band={self.config.center_top_band_m:.3f} "

                        f"pts={int(np.count_nonzero(top_sel))}/{xy.shape[0]} "

                        f"shift={float(np.hypot(*(np.median(xy_c,axis=0)-np.median(xy,axis=0)))):.4f}m"

                    )

        center_xy = np.median(xy_c, axis=0) if self.config.xy_stat == "median" else np.mean(xy_c, axis=0)

        ox, oy, oz = self.config.offset_world_m

        if self.config.grasp_below_top_enabled:

            # Property-driven descent (replaces a fixed world-z magic constant):

            # anchor on the empirically-good mask-median height, then descend a

            # FRACTION of the object's own perceived height. Tall carton -> larger

            # descent onto the wide body (avoids the tapered top that slips); short

            # can -> tiny descent (won't drive into the table). Bounded and floored.

            z_col = pts_w[:, 2]

            z_med = float(np.percentile(z_col, float(self.config.z_percentile)))

            z_top = float(np.percentile(z_col, float(self.config.grasp_top_percentile)))

            z_base = float(np.percentile(z_col, float(self.config.grasp_base_percentile)))

            height = max(0.0, z_top - z_base)

            if (self.config.grasp_anchor or "table") == "table":

                # Compute table-anchored grasp height from the scene support plane and object height.

                intr_local = {

                    "K": np.asarray(K).tolist(),

                    "world_from_camera": np.asarray(world_from_camera).tolist(),

                }

                table_ref = _scene_table_z(np, depth, intr_local)

                if table_ref is None or table_ref > z_base:

                    table_ref = z_base

                target_z = float(table_ref) + max(

                    float(self.config.grasp_below_top_min_m),

                    float(self.config.grasp_height_frac) * max(0.0, z_top - float(table_ref)),

                )

            else:

                descend = min(

                    float(self.config.grasp_below_top_max_m),

                    max(float(self.config.grasp_below_top_min_m),

                        float(self.config.grasp_below_top_frac) * height),

                )

                target_z = z_med - descend

            if self.config.verbose:

                anchor_tag = self.config.grasp_anchor or "table"

                print(

                    f"[MaskDepthGraspDetector] grasp height (anchor={anchor_tag}): "

                    f"z_med={z_med:.4f} z_top={z_top:.4f} z_base={z_base:.4f} "

                    f"h={height:.4f} target_z={target_z:.4f}"

                )

        else:

            target_z = float(np.percentile(pts_w[:, 2], float(self.config.z_percentile)))

        target_z = target_z + oz



        table_z = None

        if float(self.config.local_table_clearance_m) > 0.0:

            try:

                all_ys, all_xs = np.nonzero(np.isfinite(depth) & (depth > 0.0))

                all_z = depth[all_ys, all_xs].astype(float)

                all_xs_f = all_xs.astype(float)

                all_ys_f = all_ys.astype(float)

                all_X = (all_xs_f - K[0, 2]) * all_z / K[0, 0]

                all_Y = (all_ys_f - K[1, 2]) * all_z / K[1, 1]

                all_cam = np.stack([all_X, all_Y, all_z, np.ones_like(all_z)], axis=0)

                all_w = (world_from_camera @ all_cam)[:3].T



                mask_valid = mask[all_ys, all_xs]

                dxy = np.linalg.norm(all_w[:, :2] - center_xy[None, :], axis=1)

                object_z_ref = float(np.percentile(pts_w[:, 2], 50.0))

                z_near_object = (

                    (all_w[:, 2] <= object_z_ref + float(self.config.local_table_max_above_object_m))

                    & (all_w[:, 2] >= object_z_ref - float(self.config.local_table_max_below_object_m))

                )

                ring = (

                    (~mask_valid)

                    & (dxy >= float(self.config.local_table_ring_inner_m))

                    & (dxy <= float(self.config.local_table_ring_outer_m))

                    & z_near_object

                )

                table_pts_z = all_w[ring, 2]

                if int(table_pts_z.size) >= int(self.config.local_table_min_points):

                    table_z = float(np.percentile(table_pts_z, float(self.config.local_table_z_percentile)))

                    floor_z = table_z + float(self.config.local_table_clearance_m)

                    target_z = max(target_z, floor_z)

                    if self.config.verbose:

                        print(

                            "[MaskDepthGraspDetector] local_table "

                            f"z_p{self.config.local_table_z_percentile:g}={table_z:.4f} "

                            f"clearance={self.config.local_table_clearance_m:.4f} "

                            f"target_z={target_z:.4f}"

                        )

                elif self.config.verbose:

                    print(

                        "[MaskDepthGraspDetector] local_table unavailable: "

                        f"{int(table_pts_z.size)} pts < {int(self.config.local_table_min_points)}"

                    )

            except Exception as e:  # noqa: BLE001

                if self.config.verbose:

                    print(f"[MaskDepthGraspDetector] local_table failed: {type(e).__name__}: {e}")



        target = np.array([center_xy[0] + ox, center_xy[1] + oy, target_z], dtype=float)



        # --- top-down wrist yaw from mask PCA ---------------------------- #

        try:

            cov = np.cov(xy.T)

            eigvals, eigvecs = np.linalg.eigh(cov)

            long_axis = eigvecs[:, int(np.argmax(eigvals))]

            long_axis = long_axis / (np.linalg.norm(long_axis) or 1.0)

            opening = np.array([-long_axis[1], long_axis[0]], dtype=float)

            opening = opening / (np.linalg.norm(opening) or 1.0)

            if self.config.tool_yaw_offset_deg:

                a = math.radians(float(self.config.tool_yaw_offset_deg))

                ca, sa = math.cos(a), math.sin(a)

                opening = np.array([ca * opening[0] - sa * opening[1], sa * opening[0] + ca * opening[1]])

                opening = opening / (np.linalg.norm(opening) or 1.0)

            if self.config.flip_opening:

                opening = -opening



            z_axis = np.array([0.0, 0.0, -1.0])

            opening_axis = np.array([opening[0], opening[1], 0.0])

            if self.config.physical_open_axis_angle_deg is not None:

                a = math.radians(float(self.config.physical_open_axis_angle_deg))

                ca, sa = math.cos(a), math.sin(a)

                b_axis = np.array([opening_axis[1], -opening_axis[0], 0.0])

                x_axis = ca * opening_axis - sa * b_axis

                y_axis = sa * opening_axis + ca * b_axis

                x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)

                y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)

            elif self.config.physical_open_axis == "y":

                y_axis = opening_axis

                x_axis = np.cross(y_axis, z_axis)

                x_axis = x_axis / (np.linalg.norm(x_axis) or 1.0)

            else:

                x_axis = opening_axis

                y_axis = np.cross(z_axis, x_axis)

                y_axis = y_axis / (np.linalg.norm(y_axis) or 1.0)

            R = np.column_stack([x_axis, y_axis, z_axis])

            quat = ContactGraspNetDetector._quat_wxyz_from_rot([[float(R[i][j]) for j in range(3)] for i in range(3)])

            norm = math.sqrt(sum(v * v for v in quat)) or 1.0

            quat = [v / norm for v in quat]

        except Exception as e:  # noqa: BLE001

            return self._fail(f"yaw/orientation failed: {type(e).__name__}: {e}")



        proposal = GraspProposal(

            position=[float(v) for v in target],

            quaternion=[float(v) for v in quat],

            score=1.0,

            source=f"mask_depth_topdown_pca:{segmap_id}",

        )

        detect_ended = time.perf_counter()

        visual_latency = dict(

            (getattr(self, "last_visual_selection_evidence", {}) or {}).get(

                "latency_ms", {}

            )

        )

        selector_total = float(visual_latency.get("selector_total", 0.0))

        depth_pca_ms = round((detect_ended - depth_pca_started) * 1000, 3)

        total_ms = round((detect_ended - detect_started) * 1000, 3)

        latency = {

            "schema": "perceive-grasps-latency-v1",

            "recorded_at_unix_ns": time.time_ns(),

            "object_hint": str(object_hint or ""),

            "source": proposal.source,

            "candidate_sam3_ms": float(

                visual_latency.get("candidate_sam3", 0.0)

            ),

            "molmo_ms": float(visual_latency.get("molmo", 0.0)),

            "point_sam3_ms": float(

                visual_latency.get("point_sam3", 0.0)

            ),

            "selector_total_ms": selector_total,

            "depth_pca_ms": depth_pca_ms,

            "detector_other_ms": round(

                max(0.0, total_ms - selector_total - depth_pca_ms), 3

            ),

            "perceive_grasps_total_ms": total_ms,

        }

        self.last_perceive_latency = latency

        visual_selector = getattr(self, "visual_object_selector", None)

        debug_dir = getattr(visual_selector, "debug_dir", None)

        if debug_dir:

            debug_dir = Path(debug_dir)

            debug_dir.mkdir(parents=True, exist_ok=True)

            sidecar = debug_dir / "PERCEIVE_GRASPS_LATENCY.jsonl"

            old = sidecar.read_text(encoding="utf-8") if sidecar.exists() else ""

            line = json.dumps(latency, sort_keys=True, separators=(",", ":")) + "\n"

            tmp = sidecar.with_name(

                sidecar.name + f".tmp.{os.getpid()}.{time.time_ns()}"

            )

            tmp.write_text(old + line, encoding="utf-8")

            os.replace(tmp, sidecar)

        return proposal





def _estimate_local_table_z(

    np,

    *,

    depth,

    K,

    world_from_camera,

    center_xy,

    object_z_ref,

    mask_exclude=None,

    ring_inner_m=0.02,

    ring_outer_m=0.10,

    z_percentile=90.0,

    min_points=200,

    max_above_object_m=0.03,

    max_below_object_m=0.08,

):

    """Estimate the local table height (world z) in a ring around ``center_xy``.



    Returns a float (world z) or ``None`` when there are too few clean points.

    Points belonging to ``mask_exclude`` (the target object) are dropped, and a

    z-band around ``object_z_ref`` rejects background / robot outliers so they

    cannot poison the high percentile. Shared by the mask top-down detector and

    the hybrid trust gate so both reason about the same table plane.

    """

    try:

        depth = np.asarray(depth, dtype=np.float32)

        if depth.ndim == 3 and 1 in depth.shape:

            depth = np.squeeze(depth)

        if depth.ndim != 2:

            return None

        ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0.0))

        if xs.size == 0:

            return None

        z = depth[ys, xs].astype(float)

        X = (xs.astype(float) - K[0, 2]) * z / K[0, 0]

        Y = (ys.astype(float) - K[1, 2]) * z / K[1, 1]

        cam = np.stack([X, Y, z, np.ones_like(z)], axis=0)

        w = (world_from_camera @ cam)[:3].T

        dxy = np.linalg.norm(w[:, :2] - np.asarray(center_xy, float)[None, :], axis=1)

        keep = (dxy >= float(ring_inner_m)) & (dxy <= float(ring_outer_m))

        keep &= (w[:, 2] <= float(object_z_ref) + float(max_above_object_m))

        keep &= (w[:, 2] >= float(object_z_ref) - float(max_below_object_m))

        if mask_exclude is not None:

            me = np.asarray(mask_exclude)

            if me.shape == depth.shape:

                keep &= ~me[ys, xs].astype(bool)

        ring_z = w[keep, 2]

        if int(ring_z.size) < int(min_points):

            return None

        return float(np.percentile(ring_z, float(z_percentile)))

    except Exception:  # noqa: BLE001  (estimation is best-effort; fail open)

        return None





# --------------------------------------------------------------------------- #

# Hybrid grasp source (heuristic shell: CGN primary -> top-down fallback)      #

# --------------------------------------------------------------------------- #

class HybridGraspDetector:

    """The grasp-source heuristic shell (implements ``GraspDetector``).



    Mirrors cap-x's split between a learned grasp pose (``sample_grasp_pose`` =

    Contact-GraspNet) and a geometric pose (``get_object_pose`` = OBB/PCA), but

    composes them deterministically: try the learned ``primary`` first; if it

    declines (``None``) or returns a sub-threshold score, fall back to the

    deterministic ``fallback`` (mask/depth top-down). The returned

    :class:`GraspProposal`'s ``source`` records which path fired so policies and

    Phase-3 reflection can reason about it.



    Per DESIGN_OVERVIEW_CN §4.1 the *shell* (this class + :class:`HybridGraspConfig`)

    is what reflection tunes; the learned weights inside ``primary`` are off-limits.

    """



    def __init__(

        self,

        primary: Optional[object] = None,

        fallback: Optional[object] = None,

        *,

        config: Optional[HybridGraspConfig] = None,

    ) -> None:

        self.primary = primary

        self.fallback = fallback

        self.config = config or HybridGraspConfig()

        self.last_error: Optional[str] = None

        self.last_source: Optional[str] = None

        self.last_debug: dict = {}



    def _copy_debug(self, det) -> None:

        self.last_debug = dict(getattr(det, "last_debug", {}) or {})



    def _try(self, det, *, rgb, depth_m, intrinsics, object_hint, target_xy=None, target_px=None, sam_text_prompts=None, scene_reference=None, exclude_masks=None, scene_bundle_identity=None):

        if det is None:

            return None

        try:

            kwargs = dict(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint)

            if sam_text_prompts:

                kwargs["sam_text_prompts"] = list(sam_text_prompts)

            # ``target_xy`` (world prior) and ``target_px`` (Judge pixel anchor)

            # are opt-in disambiguation seams: only the mask/depth detector

            # consumes them, so degrade gracefully (drop kwargs progressively) for

            # detectors (e.g. CGN primary) whose detect() doesn't accept them.

            try:

                g = det.detect(target_xy=target_xy, target_px=target_px,

                               scene_reference=scene_reference,

                               exclude_masks=exclude_masks,

                               scene_bundle_identity=scene_bundle_identity,

                               **kwargs)

            except TypeError:

                try:

                    g = det.detect(target_xy=target_xy, **kwargs)

                except TypeError:

                    kwargs.pop("sam_text_prompts", None)

                    try:

                        g = det.detect(**kwargs)

                    except TypeError:

                        g = det.detect(

                            rgb=rgb, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint

                        )

            if g is None:

                err = getattr(det, "last_error", None)

                if err:

                    self.last_error = f"{type(det).__name__}: {err}"

            return g

        except Exception as e:  # noqa: BLE001  (a detector must never crash the shell)

            self.last_error = f"{type(det).__name__}.detect raised {type(e).__name__}: {e}"

            return None



    def _primary_below_table(self, g, *, depth_m, intrinsics, object_hint):

        """Return the local table z if the primary grasp sits at/below it, else None.



        Best-effort geometric trust gate: estimate the local table plane the SAME

        way the mask/depth fallback does -- a ring centred on the OBJECT CENTROID

        (not the grasp point, which for a cup can land on the rim and pull the

        estimate up the wall), excluding the object's own (dilated) mask. When the

        grasp fingertip z is below ``table_z + min_primary_z_above_table_m`` we

        treat the primary as untrustworthy (thin/flat object signature). Fails open

        (returns None) when the gate is disabled or the table cannot be estimated.

        """

        if float(self.config.min_primary_z_above_table_m) <= 0.0:

            return None

        if g is None or depth_m is None or not isinstance(intrinsics, dict):

            return None

        try:

            import numpy as np  # local import keeps module import side-effect free



            pos = [float(v) for v in g.position]

            if "K" in intrinsics:

                k = intrinsics["K"]

            elif "intrinsics" in intrinsics:

                k = intrinsics["intrinsics"]

            elif {"fx", "fy", "ppx", "ppy"}.issubset(intrinsics):

                k = [

                    [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],

                    [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],

                    [0.0, 0.0, 1.0],

                ]

            else:

                return None

            K = np.asarray(k, dtype=float).reshape(3, 3)

            wfc = intrinsics.get("world_from_camera")

            if wfc is None:

                wfc = intrinsics.get("extrinsics")

            if wfc is None:

                return None

            world_from_camera = np.asarray(wfc, dtype=float).reshape(4, 4)



            segmap = intrinsics.get("segmap")

            if segmap is None:

                segmap = intrinsics.get("mask")

            if segmap is None:

                return None  # no object footprint -> cannot anchor a clean ring

            seg = np.asarray(segmap)

            if seg.ndim == 3 and 1 in seg.shape:

                seg = np.squeeze(seg)

            match = re.search(r"(?:seg(?:ment)?|id)?\s*:?\s*(\d+)", object_hint or "")

            segmap_id = int(match.group(1)) if match else 1

            mask_bool = seg.astype(np.int32) == int(segmap_id)

            mask_exclude = ContactGraspNetDetector._dilate_mask(mask_bool, 5)



            # Object centroid xy + median z in world (same anchor the fallback uses).

            depth = np.asarray(depth_m, dtype=np.float32)

            if depth.ndim == 3 and 1 in depth.shape:

                depth = np.squeeze(depth)

            if depth.shape != mask_bool.shape:

                return None

            mys, mxs = np.nonzero(mask_bool)

            if mxs.size == 0:

                return None

            mz = depth[mys, mxs]

            mvalid = np.isfinite(mz) & (mz > 0.0)

            if int(mvalid.sum()) < 10:

                return None

            mxs = mxs[mvalid].astype(float)

            mys = mys[mvalid].astype(float)

            mz = mz[mvalid].astype(float)

            mX = (mxs - K[0, 2]) * mz / K[0, 0]

            mY = (mys - K[1, 2]) * mz / K[1, 1]

            obj_cam = np.stack([mX, mY, mz, np.ones_like(mz)], axis=0)

            obj_w = (world_from_camera @ obj_cam)[:3].T

            center_xy = np.median(obj_w[:, :2], axis=0)

            object_z_ref = float(np.percentile(obj_w[:, 2], 50.0))



            table_z = _estimate_local_table_z(

                np,

                depth=depth,

                K=K,

                world_from_camera=world_from_camera,

                center_xy=center_xy,

                object_z_ref=object_z_ref,

                mask_exclude=mask_exclude,

                ring_inner_m=float(self.config.table_ring_inner_m),

                ring_outer_m=float(self.config.table_ring_outer_m),

                z_percentile=float(self.config.table_z_percentile),

                min_points=int(self.config.table_min_points),

                max_above_object_m=float(self.config.table_max_above_object_m),

                max_below_object_m=float(self.config.table_max_below_object_m),

            )

            if table_z is None:

                return None

            if pos[2] < table_z + float(self.config.min_primary_z_above_table_m):

                return table_z

            return None

        except Exception:  # noqa: BLE001  (gate is best-effort; never crash the shell)

            return None



    def detect(self, *, rgb, depth_m, intrinsics, object_hint, prefer: Optional[str] = None, target_xy=None, target_px=None, sam_text_prompts=None, **kwargs):

        self.last_error = None

        self.last_source = None

        self.last_debug = {}

        mode = (prefer or self.config.prefer or "auto").lower()



        if mode in ("fallback", "topdown", "mask"):

            g = self._try(

                self.fallback,

                rgb=rgb,

                depth_m=depth_m,

                intrinsics=intrinsics,

                object_hint=object_hint,

                target_xy=target_xy,

                target_px=target_px,

                sam_text_prompts=sam_text_prompts,

                scene_bundle_identity=kwargs.get("scene_bundle_identity"),

            )

            self._copy_debug(self.fallback)

            self.last_source = g.source if g else None

            if g is None and self.last_error is None:

                self.last_error = "fallback declined"

            return g



        # primary first (auto / primary)

        g = self._try(

            self.primary,

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            target_xy=target_xy,

            target_px=target_px,

            sam_text_prompts=sam_text_prompts,

        )

        self._copy_debug(self.primary)

        passes_score = g is not None and (g.score is None or float(g.score) >= float(self.config.min_primary_score))



        if mode == "primary":

            # Explicit LLM request: honour it, gate or not (only score-filter).

            if passes_score:

                self.last_source = g.source

                return g

            self.last_error = self.last_error or "primary declined and prefer=primary (no fallback)"

            return g  # may be None or sub-threshold; honour the explicit request



        # auto: trust primary only if it clears BOTH the score gate and the

        # geometric table gate (a sub-table grasp = thin/flat object CGN can't grasp).

        if passes_score:

            table_z = self._primary_below_table(

                g, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint

            )

            if table_z is None:

                self.last_source = g.source

                return g

            self.last_error = (

                f"primary grasp z={float(g.position[2]):.4f} below local table "

                f"z={table_z:.4f}+{float(self.config.min_primary_z_above_table_m):.4f}; distrust -> fallback"

            )

            if self.config.verbose:

                print(f"[hybrid] {self.last_error}")



        # auto: deterministic fallback

        fb = self._try(

            self.fallback,

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            target_xy=target_xy,

            target_px=target_px,

            sam_text_prompts=sam_text_prompts,

        )

        self._copy_debug(self.fallback)

        if fb is not None:

            self.last_source = fb.source

            return fb

        self.last_source = g.source if g is not None else None

        self.last_error = self.last_error or "both primary and fallback declined"

        return g



    def detect_many(

        self,

        *,

        rgb,

        depth_m,

        intrinsics,

        object_hint,

        top_k: int = 15,

        prefer: Optional[str] = None,

        target_xy=None,

        target_px=None,

        sam_text_prompts=None,

        **kwargs,

    ) -> list[GraspProposal]:

        """Return up to ``top_k`` grasp candidates sorted by score (descending).



        The output feeds a cuRobo goalset planner (cap-x style): the motion layer

        iterates IK + collision-free planning over candidates and picks the first

        that's reachable, instead of betting everything on argmax-1.



        Routing follows the same ``prefer`` semantics as :meth:`detect`:



        - ``"primary"`` -> ``primary.detect_many`` only (no fallback)

        - ``"fallback"`` (or ``"topdown"`` / ``"mask"``) -> ``[fallback.detect()]``

          as a length-1 list (the deterministic top-down doesn't expose a top-K

          surface; iterating wouldn't help -- the geometry is fixed)

        - ``"auto"`` -> primary top-K filtered by score floor; if empty, the

          fallback's single grasp; if that also fails, ``[]``



        The hybrid table gate is *not* applied per-candidate here: the planner

        will reject unreachable / colliding ones anyway, and gating before

        planning would re-introduce the score-vs-table-z brittleness we just

        moved away from. ``prefer="primary"`` ignores the score floor too --

        explicit LLM opt-in.

        """

        self.last_error = None

        self.last_source = None

        self.last_debug = {}

        mode = (prefer or self.config.prefer or "auto").lower()

        k = max(1, int(top_k))



        if mode in ("fallback", "topdown", "mask"):

            g = self._try(

                self.fallback,

                rgb=rgb,

                depth_m=depth_m,

                intrinsics=intrinsics,

                object_hint=object_hint,

                target_xy=target_xy,

                target_px=target_px,

                sam_text_prompts=sam_text_prompts,

                scene_bundle_identity=kwargs.get("scene_bundle_identity"),

            )

            if g is None:

                self.last_error = self.last_error or "fallback declined"

                return []

            self._copy_debug(self.fallback)

            self.last_source = g.source

            return [g]



        # primary first (auto / primary)

        primary_many = []

        if self.primary is not None:

            try:

                fn = getattr(self.primary, "detect_many", None)

                if callable(fn):

                    try:

                        primary_many = list(fn(

                            rgb=rgb, depth_m=depth_m, intrinsics=intrinsics,

                            object_hint=object_hint, top_k=k, target_xy=target_xy, target_px=target_px,

                            sam_text_prompts=sam_text_prompts,

                            scene_reference=kwargs.get("scene_reference"),

                            exclude_masks=kwargs.get("exclude_masks"),

                        ))

                    except TypeError:

                        try:

                            primary_many = list(fn(

                                rgb=rgb, depth_m=depth_m, intrinsics=intrinsics,

                                object_hint=object_hint, top_k=k, target_xy=target_xy,

                                sam_text_prompts=sam_text_prompts,

                            ))

                        except TypeError:

                            try:

                                primary_many = list(fn(

                                    rgb=rgb, depth_m=depth_m, intrinsics=intrinsics,

                                    object_hint=object_hint, top_k=k,

                                    sam_text_prompts=sam_text_prompts,

                                ))

                            except TypeError:

                                primary_many = list(fn(

                                    rgb=rgb, depth_m=depth_m, intrinsics=intrinsics,

                                    object_hint=object_hint, top_k=k,

                                ))

                else:

                    try:

                        g = self.primary.detect(

                            rgb=rgb,

                            depth_m=depth_m,

                            intrinsics=intrinsics,

                            object_hint=object_hint,

                            target_xy=target_xy,

                            target_px=target_px,

                            sam_text_prompts=sam_text_prompts,

                        )

                    except TypeError:

                        g = self.primary.detect(

                            rgb=rgb, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint,

                        )

                    primary_many = [g] if g is not None else []

                self._copy_debug(self.primary)

            except Exception as e:  # noqa: BLE001

                self.last_error = f"{type(self.primary).__name__}.detect_many raised {type(e).__name__}: {e}"

                primary_many = []



        if mode == "primary":

            # explicit opt-in: honour score-filter only when ``min_primary_score``

            # > 0, else return raw top-K so the LLM/planner can decide.

            floor = float(self.config.min_primary_score)

            if floor > 0.0:

                primary_many = [

                    g for g in primary_many if g.score is None or float(g.score) >= floor

                ]

            if primary_many:

                self.last_source = primary_many[0].source

            return primary_many[:k]



        # auto: filter by score floor; keep order (already descending).

        floor = float(self.config.min_primary_score)

        kept = [g for g in primary_many if g.score is None or float(g.score) >= floor]

        if kept:

            self.last_source = kept[0].source

            return kept[:k]



        fb = self._try(

            self.fallback,

            rgb=rgb,

            depth_m=depth_m,

            intrinsics=intrinsics,

            object_hint=object_hint,

            target_xy=target_xy,

            target_px=target_px,

            sam_text_prompts=sam_text_prompts,

            scene_reference=kwargs.get("scene_reference"),

            exclude_masks=kwargs.get("exclude_masks"),

        )

        if fb is not None:

            self._copy_debug(self.fallback)

            self.last_source = fb.source

            return [fb]

        self.last_error = self.last_error or "both primary and fallback declined"

        return []





# Perception determinism checks.

@dataclass

class PerceptionReplayConfig:

    """Which camera/frame to replay and the agreement tolerances.



    ``camera`` empty -> auto-pick a fixed camera (``external_fixed`` /

    ``eye_to_hand``). Tolerances default to :class:`GraspNetConfig`'s."""



    camera: str = ""

    pos_tol_m: float = 0.01

    quat_tol_rad: float = math.radians(5)





class GraspPerceptionValidator:

    """Offline :class:`GeometryValidator` for ``perception_determinism``.



    Re-runs :class:`ContactGraspNetDetector` twice on the recorded RGB-D frame and

    checks the two grasps agree within ``pos_tol_m`` / ``quat_tol_rad``. The mask

    is grounded once (recorded ``segmap_path`` or a single SAM3 call) so the test

    isolates the grasp network's own determinism (its point-cloud downsampling uses

    ``np.random.choice``). ``grasp_reachable`` / ``rewind_path_collision_free``

    delegate to :class:`KinematicBoundsValidator` so the full checklist is intact.



    Replay inputs travel in ``rec.metadata['perception']``::



        {"camera": "overhead_rgb", "object_hint": "carrot",

         "segmap_path": "/abs/segmap.npy"}   # segmap_path optional

    """



    def __init__(

        self,

        detector: Optional["ContactGraspNetDetector"] = None,

        *,

        config: Optional[PerceptionReplayConfig] = None,

        kinematics: Optional["KinematicBoundsValidator"] = None,

    ) -> None:

        self.detector = detector

        self.config = config or PerceptionReplayConfig()

        self._bounds = kinematics or KinematicBoundsValidator()

        self.last_detail: Optional[dict] = None



    def grasp_reachable(self, rec: EpisodeRecord) -> GeoCheck:

        return self._bounds.grasp_reachable(rec)



    def rewind_path_collision_free(self, rec: EpisodeRecord) -> GeoCheck:

        return self._bounds.rewind_path_collision_free(rec)



    # -- frame/camera resolution -- #

    def _pick_camera(self, rec: EpisodeRecord):

        if not rec.cameras:

            return None

        if self.config.camera:

            for c in rec.cameras:

                if c.name == self.config.camera:

                    return c

            return None

        for role in ("external_fixed", "eye_to_hand"):

            for c in rec.cameras:

                if c.role == role:

                    return c

        return rec.cameras[0]



    def _pick_frame(self, rec: EpisodeRecord, camera: str):

        frames = [f for f in rec.frames if f.camera == camera and f.depth_path]

        if not frames:

            return None

        anchor = None

        if rec.recovery is not None:

            anchor = rec.recovery.triggered_ts

        if anchor is None:

            anchor = rec.first_fail_ts

        if anchor is None:

            return frames[-1]

        return min(frames, key=lambda f: abs(f.ts - anchor))



    @staticmethod

    def _load_depth_m(np, path: str, depth_scale: float):

        p = Path(path)

        if p.suffix == ".npy":

            raw = np.load(p)

        else:

            import cv2  # type: ignore



            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

            if raw is None:

                raise FileNotFoundError(f"could not read depth image {p}")

        return np.asarray(raw, dtype=np.float32) * float(depth_scale)



    @staticmethod

    def _load_rgb_bytes(path: str) -> bytes:

        return Path(path).read_bytes()



    def _resolve_mask(self, np, rec: EpisodeRecord, frame):

        meta = rec.metadata.get("perception", {}) if isinstance(rec.metadata, dict) else {}

        segmap_path = meta.get("segmap_path")

        if segmap_path:

            return np.load(segmap_path).astype(np.int32), None

        object_hint = meta.get("object_hint", "")

        if not object_hint:

            return None, "no perception.object_hint or perception.segmap_path in metadata"

        if self.detector is None or self.detector.segmenter is None:

            return None, "object_hint set but no SAM3 segmenter wired to ground it"

        try:

            rgb = self._load_rgb_bytes(frame.rgb_path)

        except Exception as e:  # noqa: BLE001

            return None, f"cannot read rgb {frame.rgb_path}: {type(e).__name__}: {e}"

        mask = self.detector.segmenter.segment(rgb, prompt=object_hint)

        if mask is None:

            return None, f"SAM3 grounding failed: {self.detector.segmenter.last_error}"

        return mask.astype(np.int32), None



    def perception_repeatable(self, rec: EpisodeRecord) -> GeoCheck:

        self.last_detail = None

        if rec.recovery is None or rec.recovery.perceived_grasp is None:

            return GeoCheck.pending("no perceived grasp to reproduce")

        if self.detector is None:

            return GeoCheck.pending("no grasp detector wired")

        try:

            import numpy as np  # type: ignore

        except Exception:

            return GeoCheck.pending("numpy unavailable")



        cam = self._pick_camera(rec)

        if cam is None:

            return GeoCheck.pending("no usable camera spec in record")

        frame = self._pick_frame(rec, cam.name)

        if frame is None:

            return GeoCheck.pending(f"no depth frame for camera {cam.name!r}")



        try:

            depth_m = self._load_depth_m(np, frame.depth_path, cam.depth_scale)

        except Exception as e:  # noqa: BLE001

            return GeoCheck.pending(f"cannot load depth: {type(e).__name__}: {e}")



        mask, err = self._resolve_mask(np, rec, frame)

        if mask is None:

            return GeoCheck.pending(err or "could not resolve target mask")



        intr = {"K": cam.intrinsics, "world_from_camera": cam.extrinsics, "mask": mask}

        g1 = self.detector.detect(rgb=None, depth_m=depth_m, intrinsics=intr, object_hint="seg:1")

        if g1 is None:

            return GeoCheck(False, f"first perception run produced no grasp: {self.detector.last_error}")

        g2 = self.detector.detect(rgb=None, depth_m=depth_m, intrinsics=intr, object_hint="seg:1")

        if g2 is None:

            return GeoCheck(False, f"second perception run produced no grasp: {self.detector.last_error}")



        dpos = math.sqrt(sum((a - b) ** 2 for a, b in zip(g1.position, g2.position)))

        dang = _quat_angle_rad(g1.quaternion, g2.quaternion)

        self.last_detail = {

            "camera": cam.name,

            "frame_ts": frame.ts,

            "pos_delta_m": dpos,

            "rot_delta_deg": math.degrees(dang),

            "grasp": list(g1.position),

            "score": g1.score,

        }

        if dpos <= self.config.pos_tol_m and dang <= self.config.quat_tol_rad:

            return GeoCheck(True, f"perception reproducible: Δpos={dpos*1000:.2f} mm, Δrot={math.degrees(dang):.3f}°")

        return GeoCheck(

            False,

            f"perception NOT reproducible: Δpos={dpos*1000:.2f} mm (tol {self.config.pos_tol_m*1000:.1f}), "

            f"Δrot={math.degrees(dang):.3f}° (tol {math.degrees(self.config.quat_tol_rad):.1f})",

        )





def _quat_angle_rad(q1, q2) -> float:

    dot = abs(sum(a * b for a, b in zip(q1, q2)))

    dot = max(-1.0, min(1.0, dot))

    return 2.0 * math.acos(dot)





# --------------------------------------------------------------------------- #

# factories (fail-closed: return None when deps/config are unavailable)        #

# --------------------------------------------------------------------------- #

def build_motion_planner(config: Optional[CuroboConfig] = None) -> Optional[CuroboMotionPlanner]:

    try:

        return CuroboMotionPlanner(config)

    except Exception as e:  # noqa: BLE001  (ImportError / CUDA / config)

        print(f"[recovery.geometry] curobo planner unavailable: {type(e).__name__}: {e}")

        return None





def build_grasp_detector(

    config: Optional[GraspNetConfig] = None,

    *,

    sam3: Optional[Sam3Config] = None,

    enable_sam3: bool = True,

) -> Optional[ContactGraspNetDetector]:

    """Build the grasp detector, optionally wiring a SAM3 text segmenter.



    ``enable_sam3`` (default) attaches a :class:`Sam3Segmenter` so free-text

    ``object_hint``s are grounded to a mask. If SAM3 resources are unavailable the

    detector is still returned (callers can supply an explicit mask/segmap)."""

    try:

        segmenter = build_sam3_segmenter(sam3) if enable_sam3 else None

        return ContactGraspNetDetector(config, segmenter=segmenter)

    except Exception as e:  # noqa: BLE001

        print(f"[recovery.geometry] contact-graspnet detector unavailable: {type(e).__name__}: {e}")

        return None





def build_mask_grasp_detector(

    config: Optional[MaskDepthGraspConfig] = None,

    *,

    segmenter: Optional["Sam3Segmenter"] = None,

    pointer: Optional["MolmoPointer"] = None,

    sam3: Optional[Sam3Config] = None,

    molmo: Optional[MolmoConfig] = None,

    enable_sam3: bool = True,

    enable_molmo: bool = False,

) -> MaskDepthGraspDetector:

    """Build the deterministic mask/depth top-down detector (always available).



    Pass an already-built ``segmenter`` to share one SAM3 process with the CGN

    detector (avoids launching two). Otherwise it is built from ``sam3`` when

    ``enable_sam3``. Never returns ``None``: it has no heavy deps and is the

    fallback of last resort."""

    if segmenter is None and enable_sam3:

        segmenter = build_sam3_segmenter(sam3)

    if pointer is None and enable_molmo:

        pointer = build_molmo_pointer(molmo)

    return MaskDepthGraspDetector(config, segmenter=segmenter, pointer=pointer)





def build_graspgen_detector(

    config: Optional[GraspGenConfig] = None,

    *,

    segmenter: Optional["Sam3Segmenter"] = None,

    pointer: Optional["MolmoPointer"] = None,

    sam3: Optional[Sam3Config] = None,

    molmo: Optional[MolmoConfig] = None,

    enable_sam3: bool = True,

    enable_molmo: bool = False,

) -> GraspGenDetector:

    """Build the lightweight GraspGen ZMQ client detector.



    This never loads the GraspGen model in-process; callers must start the

    GraspGen server separately. It always returns a detector, and failures are

    reported at ``detect[_many]`` time via ``last_error``.

    """

    if segmenter is None and enable_sam3:

        segmenter = build_sam3_segmenter(sam3)

    if pointer is None and enable_molmo:

        pointer = build_molmo_pointer(molmo)

    return GraspGenDetector(config, segmenter=segmenter, pointer=pointer)





def build_hybrid_grasp_detector(

    config: Optional[GraspNetConfig] = None,

    *,

    mask_config: Optional[MaskDepthGraspConfig] = None,

    hybrid: Optional[HybridGraspConfig] = None,

    sam3: Optional[Sam3Config] = None,

    molmo: Optional[MolmoConfig] = None,

    shared_segmenter: Optional[Sam3Segmenter] = None,

    shared_pointer: Optional[MolmoPointer] = None,

    enable_sam3: bool = True,

    enable_molmo: bool = False,

    primary_score_floor: float = 0.0,

) -> HybridGraspDetector:

    """Build the hybrid grasp-source shell: Contact-GraspNet primary + mask/depth

    top-down fallback, sharing a single SAM3 segmenter.



    The CGN primary is wired only if its resources exist (else ``primary=None``

    and the shell runs fallback-only, fail-closed). The deterministic fallback is

    always present, so on hardware a flat-lying object still yields a grasp.



    The accept/reject decision lives in the *shell*, not buried in the runner:

    the CGN ``score_threshold`` is lowered to ``primary_score_floor`` (default 0)

    so CGN returns its best on-object candidate, and :class:`HybridGraspConfig`'s

    ``min_primary_score`` (reflection-tunable) decides whether to trust it. The

    real flat-object trigger is CGN returning *no grasps at all* -> fall back."""

    base = config or GraspNetConfig()

    primary_cfg = replace(base, score_threshold=float(primary_score_floor))

    segmenter = shared_segmenter

    if segmenter is None and enable_sam3:

        segmenter = build_sam3_segmenter(sam3)

    pointer = shared_pointer

    if pointer is None and enable_molmo:

        pointer = build_molmo_pointer(molmo)

    try:

        primary: Optional[ContactGraspNetDetector] = ContactGraspNetDetector(primary_cfg, segmenter=segmenter)

    except Exception as e:  # noqa: BLE001

        print(f"[recovery.geometry] contact-graspnet primary unavailable: {type(e).__name__}: {e}")

        primary = None

    fallback = MaskDepthGraspDetector(mask_config, segmenter=segmenter, pointer=pointer)

    return HybridGraspDetector(primary, fallback, config=hybrid)

