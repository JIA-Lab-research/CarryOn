#!/usr/bin/env python3
"""Evaluate motion using scene roles, swept geometry and the contact graph."""
import hashlib

import numpy as np

import contact_graph as CG
import phase_latch as PL
import relational_scene as RSC   # Held and remaining target geometry helpers.

LABELS = ("TARGET", "SOURCE_SUPPORT", "DESTINATION", "OTHER")

# Dispatch branch labels written by the shield at the decision point.
BRANCH_N_GE_2 = "N_GE_2_UNION"            # >=2 candidates -> exempt the union
BRANCH_N_EQ_1 = "N_EQ_1"                  # unchanged behaviour
BRANCH_N_EQ_0 = "N_EQ_0_FAIL_CLOSED"       # no candidate at all
BRANCH_M_EQ_0 = "M_EQ_0_FAIL_CLOSED"       # lab OK but destination mask is empty
BRANCH_ROLES_FAIL = "ROLES_FAIL_FAIL_CLOSED"
BRANCH_OTHER_FAIL = "OTHER_FAIL_STOP"       # undeclared status: raise, never fold





BRANCH_IDENTITY_COLLISION = "IDENTITY_COLLISION_FAIL_CLOSED"
# Semantic grounding failures use the fail-closed branch; configuration defects raise.


def _xyz(p):
    """Serialise a trigger point for telemetry (None-safe)."""
    return None if p is None else [round(float(v), 4) for v in p]


def _concat(groups, labels):
    pts = [groups[l] for l in labels if l in groups and groups[l] is not None and len(groups[l])]
    return np.vstack(pts) if pts else np.empty((0, 3))


def _hazard_labels(actor, regime, relational_on):
    """Scene labels that are FORBIDDEN contacts for this actor in this regime.
    relational_on=False -> every label is a hazard (no intended-contact exempt)."""
    if not relational_on:
        return list(LABELS)
    haz = []
    for lb in LABELS:
        if not CG.contact_allowed(actor, lb, regime):
            haz.append(lb)
    return haz


def _clear_static(pts, spheres):
    if len(pts) == 0 or not spheres:
        return float("inf")
    return min(float(np.min(np.linalg.norm(pts - np.asarray(c), axis=1)) - r) for c, r in spheres)


def _clear_swept(pts, spheres, offset):
    if len(pts) == 0 or not spheres:
        return float("inf")
    return min(float(np.min(np.linalg.norm(pts - (np.asarray(c) + offset), axis=1)) - r) for c, r in spheres)


def _unswept_self_clearance(point, spheres):
    """Surface-referenced distance from one point to the nearest FK sphere
    with NO swept offset -- the exact quantity the self-removal step tests
    against self_band. Telemetry only. Reported BESIDE clearance_m, which
    for the gripper actor is the SWEPT value: the two are different calibers
    and are never subtracted."""
    if point is None or not spheres:
        return None
    p = np.asarray(point, float)
    return float(min(float(np.linalg.norm(p - np.asarray(c))) - float(r)
                     for c, r in spheres))


def _clear_static_pt(pts, spheres):
    """(clearance, point) at the minimum-clearance point. The scalar minimum is
    identical to _clear_static; only the argmin index is additionally tracked."""
    if len(pts) == 0 or not spheres:
        return float("inf"), None
    best = np.full(len(pts), np.inf)
    for c, r in spheres:
        best = np.minimum(best, np.linalg.norm(pts - np.asarray(c), axis=1) - float(r))
    i = int(np.argmin(best))
    return float(best[i]), pts[i]


def _clear_swept_pt(pts, spheres, offset):
    """(clearance, point) at the minimum-clearance point under a swept offset."""
    if len(pts) == 0 or not spheres:
        return float("inf"), None
    best = np.full(len(pts), np.inf)
    for c, r in spheres:
        best = np.minimum(best, np.linalg.norm(pts - (np.asarray(c) + offset), axis=1) - float(r))
    i = int(np.argmin(best))
    return float(best[i]), pts[i]


def _quat_xyzw_to_mat(quat):
    """Unit-quaternion (xyzw) to a 3x3 active rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    n = float(x * x + y * y + z * z + w * w)
    if n <= np.finfo(np.float64).eps:
        raise ValueError("zero-norm EE quaternion")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ], dtype=np.float64)


def _rigid_attached(cloud_latch, ee_latch, rot_latch, ee_now, quat_now):
    """Apply R_now R_latch^T (cloud_latch - ee_latch) + ee_now."""
    points_local = ((np.asarray(cloud_latch, dtype=np.float64) -
                     np.asarray(ee_latch, dtype=np.float64)) @
                    np.asarray(rot_latch, dtype=np.float64))
    rot_now = _quat_xyzw_to_mat(quat_now)
    return points_local @ rot_now.T + np.asarray(ee_now, dtype=np.float64)


def motion_unsafe(groups, regime, static_spheres, grip_spheres, offsets, floor,
                  *, relational_on=True, held_point_radius=0.0,
                  floor_gripper=None, floor_held=None):
    """Return global earliest unsafe k across actors; arm then gripper win ties.

    Driver records (actor, label, clearance_m, point). Labels and geometry are
    unchanged; a later gripper hit cannot hide an earlier held-object hit.
    """
    offsets = list(offsets) if offsets is not None and len(offsets) else [np.zeros(3)]
    # Actor-specific standoffs default to the arm/wrist value in `floor`.
    _fg = float(floor if floor_gripper is None else floor_gripper)
    _fh = float(floor if floor_held is None else floor_held)

    def _worst(labels, fn):
        """(clearance, label, point) for the label achieving the minimum
        clearance. Identical selection to the pre-increment version; the
        trigger point is carried out for offline class-B attribution."""
        best_c, best_l, best_p = float("inf"), None, None
        for lb in labels:
            pts = groups.get(lb)
            if pts is None or not len(pts):
                continue
            c, p = fn(np.asarray(pts))
            if c < best_c:
                best_c, best_l, best_p = c, lb, p
        return best_c, best_l, best_p

    # --- arm / wrist: offset-invariant static clearance to its hazards -------
    c, lb, pt = _worst(_hazard_labels(CG.ROBOT_ARM_WRIST, regime, relational_on),
                       lambda p: _clear_static_pt(p, static_spheres))
    if c <= floor:
        return True, 0, (CG.ROBOT_ARM_WRIST, lb, round(float(c), 5), _xyz(pt))
    # --- terminal gripper: swept clearance to its hazards --------------------
    grip_labels = _hazard_labels(CG.ROBOT_TERMINAL_GRIPPER, regime, relational_on)
    first_gripper = None
    for k, off in enumerate(offsets):
        c, lb, pt = _worst(grip_labels, lambda p, o=off: _clear_swept_pt(p, grip_spheres, o))
        if c <= _fg:
            first_gripper = (True, k, (CG.ROBOT_TERMINAL_GRIPPER, lb, round(float(c), 5), _xyz(pt)))
            if k == 0:
                return first_gripper
            break
    # --- held target (CARRY): its points swept with the gripper, DEST exempt -
    if relational_on and regime == PL.CARRY:
        held_pts = groups.get("HELD_TARGET")
        if held_pts is not None and len(held_pts):
            held_labels = _hazard_labels(CG.HELD_TARGET, regime, relational_on)
            for k, off in enumerate(offsets):
                # Equal-k ties retain gripper priority; only earlier held hazards shorten the grant.
                if first_gripper is not None and k >= first_gripper[1]:
                    break
                moved = held_pts + off

                def _held_clear(p, m=moved):
                    d = np.linalg.norm(p[:, None, :] - m[None, :, :], axis=2)
                    per_pt = d.min(axis=1)
                    i = int(np.argmin(per_pt))
                    return float(per_pt[i]) - held_point_radius, p[i]

                c, lb, pt = _worst(held_labels, _held_clear)
                if c <= _fh:
                    return True, k, (CG.HELD_TARGET, lb, round(float(c), 5), _xyz(pt))
    return first_gripper if first_gripper is not None else (False, None, None)


def motion_unsafe_rollout(groups, regime, static_spheres, rollout_steps, floor,
                          *, relational_on=True, held_point_radius=0.0,
                          floor_gripper=None, floor_held=None,
                          held_cloud_latch=None, held_ee_latch=None,
                          held_rot_latch=None):
    """Evaluate per-step rollout gripper geometry and the rigid held-object cloud."""
    rollout_steps = list(rollout_steps)
    if not rollout_steps:
        raise RuntimeError("ROLLOUT_EMPTY_EXECUTION_WINDOW")
    _fg = float(floor if floor_gripper is None else floor_gripper)
    _fh = float(floor if floor_held is None else floor_held)

    def _worst(labels, fn):
        best_c, best_l, best_p = float("inf"), None, None
        for lb in labels:
            pts = groups.get(lb)
            if pts is None or not len(pts):
                continue
            c, p = fn(np.asarray(pts))
            if c < best_c:
                best_c, best_l, best_p = c, lb, p
        return best_c, best_l, best_p

    c, lb, pt = _worst(
        _hazard_labels(CG.ROBOT_ARM_WRIST, regime, relational_on),
        lambda p: _clear_static_pt(p, static_spheres))
    if c <= floor:
        return True, 0, (CG.ROBOT_ARM_WRIST, lb, round(float(c), 5), _xyz(pt))

    grip_labels = _hazard_labels(CG.ROBOT_TERMINAL_GRIPPER, regime, relational_on)
    first_gripper = None
    for k, step in enumerate(rollout_steps):
        if int(step.get("k", -1)) != k:
            raise RuntimeError("ROLLOUT_K_NONCONTIGUOUS:%r!=%d" % (step.get("k"), k))
        spheres = step.get("spheres_k")
        if not spheres:
            raise RuntimeError("ROLLOUT_EMPTY_GRIPPER_SPHERES:k=%d" % k)
        c, lb, pt = _worst(grip_labels, lambda p, s=spheres: _clear_static_pt(p, s))
        if c <= _fg:
            first_gripper = (True, k, (CG.ROBOT_TERMINAL_GRIPPER, lb, round(float(c), 5), _xyz(pt)))
            if k == 0:
                return first_gripper
            break

    if relational_on and regime == PL.CARRY:
        held_pts = groups.get("HELD_TARGET")
        if held_pts is not None and len(held_pts):
            if (held_cloud_latch is None or held_ee_latch is None or
                    held_rot_latch is None):
                raise RuntimeError("ROLLOUT_HELD_LATCH_MISSING")
            held_labels = _hazard_labels(CG.HELD_TARGET, regime, relational_on)
            for k, step in enumerate(rollout_steps):
                # Equal-k ties retain gripper priority; only earlier held hazards shorten the grant.
                if first_gripper is not None and k >= first_gripper[1]:
                    break
                moved = _rigid_attached(
                    held_cloud_latch, held_ee_latch, held_rot_latch,
                    step["ee_pos_k"], step["ee_quat_k"])

                def _held_clear(p, m=moved):
                    d = np.linalg.norm(p[:, None, :] - m[None, :, :], axis=2)
                    per_pt = d.min(axis=1)
                    i = int(np.argmin(per_pt))
                    return float(per_pt[i]) - held_point_radius, p[i]

                c, lb, pt = _worst(held_labels, _held_clear)
                if c <= _fh:
                    return True, k, (CG.HELD_TARGET, lb, round(float(c), 5), _xyz(pt))
    return first_gripper if first_gripper is not None else (False, None, None)


def _cloud_world(depth_disp, c2w):
    """Unproject every valid pixel to world + keep (v,u) index for masking."""
    z = np.asarray(depth_disp, float)
    H, W = z.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    ys, xs = vv[valid], uu[valid]
    zz = z[valid]
    cam = np.stack([xs * zz, ys * zz, zz, np.ones_like(zz)], axis=1)
    world = (np.asarray(c2w) @ cam.T).T[:, :3]
    return world, ys, xs


def _voxel(points, size=0.015):
    """Voxel downsample (keeps spatial coverage of the nearest obstacle while
    capping density) -- same role as the base depth shield's _voxel_downsample."""
    if len(points) == 0:
        return points
    keys = np.floor(np.asarray(points) / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def _plane_z(points, percentile):
    """Estimate support height from the z-histogram mode, with a small-cloud percentile fallback."""
    z = np.asarray(points)[:, 2]
    if len(z) < 50:
        return float(np.percentile(z, percentile))
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(z, bins=max(20, int((hi - lo) / 0.01)))
    i = int(np.argmax(hist))
    return float(0.5 * (edges[i] + edges[i + 1]))


def _aabb(pts, margin):
    lo = pts.min(0) - margin
    hi = pts.max(0) + margin
    return lo, hi


def _inside(pts, box):
    lo, hi = box
    return ((pts >= lo) & (pts <= hi)).all(1)


class RelationalShield:
    """Initialize scene roles once and update depth geometry per action chunk.

    Source and destination roles remain static. In CARRY, held-target geometry
    moves with the gripper while remaining target instances stay static. Residual
    points from the current depth frame form the OTHER group."""
    predictor = None                 # Set per instance by __init__.
    trigger_distance_gripper = None  # None follows the arm/wrist standoff.
    trigger_distance_held = None     # None follows the arm/wrist standoff.

    def __init__(self, segmenter, build_labels_fn, fk_all_fn, fk_grip_fn, self_mask_fn,
                 offsets_fn, floor, gripper_closed_width, *, execution_window,
                 trigger_distance=None,
                 trigger_distance_gripper=None, trigger_distance_held=None,
                 relational_on=True, other_aabb_margin_m=0.03,
                 plane_percentile=20.0, float_margin_m=0.03,
                 self_band_m=0.06, tracking_adapter=None, predictor=None):
        self.seg = segmenter
        self.build_labels_fn = build_labels_fn
        self.fk_all_fn = fk_all_fn
        self.fk_grip_fn = fk_grip_fn
        self.self_mask_fn = self_mask_fn        # (proprio, depth_disp, w2p, H, W) -> bool self mask
        self.offsets_fn = offsets_fn
        self.execution_window = int(execution_window)
        if self.execution_window <= 0:
            raise ValueError("execution_window must be positive")
        self.floor = float(floor)
        self.trigger_distance = float(floor if trigger_distance is None else trigger_distance)
        if self.trigger_distance < self.floor:
            raise ValueError("trigger_distance must be >= floor")
        # Actor-specific overrides default to the arm/wrist trigger distance.
        # Keep missing overrides as None so they follow its current value.
        self.trigger_distance_gripper = (
            None if trigger_distance_gripper is None else float(trigger_distance_gripper))
        self.trigger_distance_held = (
            None if trigger_distance_held is None else float(trigger_distance_held))
        for _name, _value in (("gripper", self.trigger_distance_gripper),
                              ("held", self.trigger_distance_held)):
            if _value is None:
                continue
            if _value < self.floor:
                raise ValueError("trigger_distance_%s must be >= floor" % _name)
            if _value > self.trigger_distance:
                raise ValueError(
                    "trigger_distance_%s must not exceed the arm/wrist trigger" % _name)
        self.relational_on = bool(relational_on)
        self._closed_w = float(gripper_closed_width)
        self._margin = float(other_aabb_margin_m)
        self._plane_pct = float(plane_percentile)
        self._float_margin = float(float_margin_m)   # support-plane clearance (as the base depth shield)
        # Use the arm/wrist trigger distance as the self-removal band, measured
        # from the robot-sphere surface independently of actor overrides.
        self._self_band = max(float(self_band_m), self.trigger_distance)
        self._voxel = 0.015                          # Point-cloud voxel size.
        self.tracker = tracking_adapter
        self.predictor = predictor
        self.reset_episode()

    def reset_episode(self):
        self.latch = PL.PhaseLatch(self._closed_w)
        self.fail_closed = None
        # Reset the halt and shield-off states at each episode boundary.
        self.shield_off = None
        self.src_pts = self.dst_pts = self.tgt_pts0 = None
        self.tgt_instances0 = None      # Per-instance TARGET point sets.
        self._held_i = None             # Held instance, locked at the CARRY latch.
        self._grasp_reach = None        # Held-object extent recorded for telemetry.
        self._carry_anchor_ee = None    # End-effector position anchoring the held model at the CARRY latch.
        self._held_cloud_latch = None   # CARRY-latch world cloud.
        self._held_ee_latch = None      # CARRY-latch end-effector position.
        self._held_rot_latch = None     # CARRY-latch end-effector rotation.
        self._tracking_result = None
        self.init_ee = None
        self.c2w = self.w2p = None
        self.sam3_calls = 0
        self.grounding_batches = 0
        self._boxes = {}
        if self.tracker is not None:
            self.tracker.reset_episode()

    def initialize_episode(self, rgb, depth_disp, c2w, w2p, proprio, instruction):
        """One-time SAM3/relational grounding on the initial frame.

        Batch2 keeps the fifth-narrow dispatch priority but restores fail-closed
        for the three declared semantic grounding failures. The runner records
        the original status verbatim and continues the horizon on dummy actions.

        Dispatch priority, single source of truth for the whole system:
            roles failure > N == 0 > M == 0 > N >= 2 > N == 1
        The branch decided here is written to the recorder immediately below
        (B7). No other module may re-derive it: two implementations of one
        priority rule can drift, and the label that drifts is the artifact.
        """
        self.c2w, self.w2p = np.asarray(c2w), np.asarray(w2p)
        if self.tracker is not None:
            try:
                _roles, _plural = instruction
                _targets = _roles.get("TARGET") if _roles else None
            except Exception:
                _targets = None


            self.tracker.configure_targets(_targets)
        lab = self.build_labels_fn(rgb, depth_disp, c2w, w2p, proprio, instruction, self.seg,
                                   self.fk_all_fn)
        self.sam3_calls += int(lab.get("sam3_calls", 0))
        self.grounding_batches += 1
        status = lab.get("status", "FAIL")
        # Dispatch decision.
        if status == "OK":
            # dest_degraded indicates an empty destination mask, not a target candidate count.
            if bool(lab.get("dest_degraded", False)):
                branch = BRANCH_M_EQ_0
            elif len(lab.get("survivors") or []) >= 2:
                branch = BRANCH_N_GE_2
            else:
                branch = BRANCH_N_EQ_1
        elif status == "FAIL_roles_unavailable":
            branch = BRANCH_ROLES_FAIL
        elif status.endswith("FAIL_CLOSED_no_candidates"):
            branch = BRANCH_N_EQ_0
        elif status.endswith("IDENTITY_COLLISION_FAIL_CLOSED"):
            branch = BRANCH_IDENTITY_COLLISION
        else:
            # Undeclared statuses are assembly/method faults, never an ordinary
            # fail-closed episode and never a pass-through.
            raise RuntimeError("%s:%s" % (BRANCH_OTHER_FAIL, status))
        self.fail_closed = branch if branch in {
            BRANCH_ROLES_FAIL, BRANCH_N_EQ_0, BRANCH_M_EQ_0,
            BRANCH_IDENTITY_COLLISION,
        } else None
        self.shield_off = None
        # Every branch leaves usable role point sets.
        # A shield-off branch arrives with no `groups` at all (build_labels
        # returns early). Leaving the role sets at None would crash the first
        # `len(...)` in _current_groups; empty arrays keep every use-site total.
        g = lab.get("groups") or {}
        _empty = np.empty((0, 3))
        self.src_pts = g.get("SOURCE_SUPPORT", _empty)
        self.dst_pts = g.get("DESTINATION", _empty)
        self.tgt_pts0 = g.get("TARGET", _empty)
        self.tgt_instances0 = list(lab.get("target_instances") or [])
        self.init_ee = np.asarray(proprio[7:10], float)
        # Remove role points within one voxel of the support surface,
        # while retaining the role set if filtering would empty it.
        _all = [p for p in (self.tgt_pts0, self.src_pts, self.dst_pts,
                            g.get("OTHER", _empty))
                if p is not None and len(p)]
        if _all:
            plane0 = _plane_z(np.vstack(_all), self._plane_pct)
            band = plane0 + self._voxel
            for _name in ("src_pts", "dst_pts"):
                _p = getattr(self, _name)
                if _p is not None and len(_p):
                    _keep = _p[:, 2] > band
                    if 0 < int(_keep.sum()) < len(_p):
                        setattr(self, _name, _p[_keep])
        for k, pts in (("SOURCE_SUPPORT", self.src_pts), ("DESTINATION", self.dst_pts),
                       ("TARGET", self.tgt_pts0)):
            self._boxes[k] = _aabb(pts, self._margin) if len(pts) else None
        # Read the runner's current record after build_labels_fn; reset() can
        # replace its dictionary. The recorder is optional for other callers.
        _rec = getattr(self, "dest_record", None)
        if _rec is not None:
            _rec.record["dispatch_branch"] = branch
        # Return the status string for the runner to record.
        if not self.fail_closed:
            from cb3_runtime import Destination
            self._cb3 = Destination()
            mask, fallback = self.seg._cb3_initial_destination
            self._cb3.initialize(rgb, mask, fallback)
        return status

    def _current_groups(self, depth_disp, proprio, all_spheres):
        """Build per-chunk scene groups plus the held actor without a model call.

        SOURCE/DEST remain static. PRE_GRASP TARGET instances stay static; in
        CARRY only HELD_TARGET follows the gripper and all remaining TARGET
        instances stay at their grounded coordinates as forbidden hazards.
        OTHER is rebuilt from current residual depth."""
        regime = self.latch.regime()
        destination = self.dst_pts if regime == PL.PRE_GRASP else np.empty((0, 3))
        ee_now = np.asarray(proprio[7:10], float)
        held = np.empty((0, 3))
        if regime == PL.PRE_GRASP:
            tgt = self.tgt_pts0
        else:
            _snapshot = (getattr(self, "_tracking_result", None) or {}).get("snapshot")
            if _snapshot is not None:


                _carry = RSC.tracking_carry_groups(
                    _snapshot["candidate_records"],
                    _snapshot["held_candidate_id"],
                )
                held = _carry["HELD_TARGET"]
                tgt = _carry["TARGET"]
                _insts = None
            else:
                _insts = self.tgt_instances0 or []
            # Select once on the first CARRY chunk and lock until reset_episode.
            if (_insts is not None and self._held_i is None and _insts
                    and not self.latch.released):
                self._held_i = RSC.held_instance_index(_insts, ee_now)












            if _insts is not None and self._carry_anchor_ee is None:
                self._carry_anchor_ee = np.asarray(ee_now, float).copy()
            if _insts is not None and _insts and self._held_i is not None:
                _carry = RSC.carry_groups(_insts, self._held_i,
                                          ee_now, self._carry_anchor_ee)
                held = _carry["HELD_TARGET"]
                tgt = _carry["TARGET"]
            elif _insts is not None:
                # Without per-instance target data, retain single-target rigid-follow protection.
                held = self.tgt_pts0 + (ee_now - self._carry_anchor_ee)
                tgt = np.empty((0, 3))
        if self.latch.released:
            # After measured gripper width confirms release, clear HELD_TARGET
            # and prevent a new held-target latch for the rest of the episode.
            held = np.empty((0, 3))
        if self.predictor is not None and regime == PL.CARRY and len(held):



            if self._held_cloud_latch is None:
                self._held_cloud_latch = np.asarray(held, dtype=np.float64).copy()
                self._held_ee_latch = ee_now.copy()
                self._held_rot_latch = _quat_xyzw_to_mat(proprio[10:14])
            held = _rigid_attached(
                self._held_cloud_latch, self._held_ee_latch,
                self._held_rot_latch, ee_now, proprio[10:14])
        if len(held) and self._grasp_reach is None:
            # Record the held-object extent at the first CARRY chunk, measured from the
            # grasp point. This value is telemetry and does not affect decisions.
            self._grasp_reach = float(
                np.max(np.linalg.norm(held - ee_now, axis=1))) + self._margin
        # current cloud minus robot self, minus the static fixture AABBs -> OTHER
        world, ys, xs = _cloud_world(depth_disp, self.c2w)



        _self_mask_fn = getattr(self, "self_mask_fn", None)
        if len(world) and _self_mask_fn is not None:
            H, W = depth_disp.shape[:2]
            self_mask = np.asarray(
                _self_mask_fn(proprio, depth_disp, self.w2p, H, W), dtype=bool
            )
            if self_mask.shape != (H, W):
                raise ValueError(
                    "SELF_MASK_CURRENT_SELF_MASK_SHAPE:%r!=%r" %
                    (self_mask.shape, (H, W)))
            world = world[~self_mask[ys, xs]]
        world = _voxel(world, 0.015)   # cap density; preserves nearest-obstacle coverage
        if len(world) and all_spheres:
            # world-space robot self-removal: delete points within (radius +
            # self_band) of any FK sphere. The band reaches the actual arm/hand
            # surface, which sits outside the tight curobo spheres.
            keep = np.ones(len(world), bool)
            for c, r in all_spheres:
                keep &= np.linalg.norm(world - np.asarray(c), axis=1) > (float(r) + self._self_band)
            world = world[keep]
        if len(world):
            # Support-plane (table/floor) exclusion: OTHER is the FLOATING hazard
            # set above the dominant low surface -- the same support-surface
            # filter the base depth shield applies. The support surface is not a
            # collision obstacle (the robot operates on it).
            plane_z = _plane_z(world, self._plane_pct)   # Estimate support height from the mode.
            floating = world[:, 2] > plane_z + self._float_margin
            claimed = np.zeros(len(world), bool)
            for k in ("SOURCE_SUPPORT", "DESTINATION"):
                if regime == PL.CARRY and k == "DESTINATION":
                    continue
                box = self._boxes.get(k)
                if box is not None:
                    claimed |= _inside(world, box)
            if regime == PL.PRE_GRASP:
                # Per-instance boxes avoid exempting the empty bridge between
                # separate TARGET instances. N=1 is numerically unchanged.
                _claim_instances = self.tgt_instances0 or (
                    [tgt] if len(tgt) else [])
                for _pts in _claim_instances:
                    if _pts is not None and len(_pts):
                        claimed |= _inside(
                            world, _aabb(np.asarray(_pts), self._margin))
            elif len(held):
                # Only the lifted instance's own depth pixels are excluded.
                # Stationary TARGET instances remain TARGET hazards and may also
                # conservatively appear in OTHER.
                claimed |= _inside(world, _aabb(held, self._margin))












            if regime == PL.CARRY:
                other, destination = self._cb3.apply(world, floating, claimed, depth_disp)
            else:
                other = world[floating & ~claimed]
        else:
            other = np.empty((0, 3))
        return {"HELD_TARGET": held, "TARGET": tgt,
                "SOURCE_SUPPORT": self.src_pts,
                "DESTINATION": destination, "OTHER": other}, regime

    def decide(self, depth_disp, proprio, action_chunk, *, rgb=None, absolute_action_step=None):
        """Per-chunk decision. Returns (mode, evade_action_or_None, telemetry).
        mode in {'pass_through','evade','fail_closed'}."""
        # Declared semantic grounding failures fail closed. An unexpected
        # shield_off latch raises a configuration error.
        if self.fail_closed:
            return "fail_closed", None, {
                "fail_closed": self.fail_closed,
                "floor_m": self.floor,
                "trigger_distance_m": self.trigger_distance,
                "sam3_calls": self.sam3_calls,
                "grounding_batches": self.grounding_batches,
            }
        if self.shield_off:
            raise RuntimeError("BATCH2_UNDECLARED_SHIELD_OFF:%s" % self.shield_off)
        chunk = np.asarray(action_chunk, float)
        grip_w = abs(float(proprio[14]) - float(proprio[15])) if len(proprio) >= 16 else 0.08
        act_grip = float(chunk[0][6]) if chunk.ndim == 2 and chunk.shape[1] > 6 else -1.0
        self.latch.update(grip_w, act_grip)
        if self.latch.regime() == PL.CARRY:
            self._cb3.prepare(rgb, depth_disp, self.w2p, absolute_action_step)
            self.sam3_calls += self._cb3.row["native_model_calls"]
        if self.tracker is not None and self.latch.regime() == PL.CARRY:
            self._tracking_result = self.tracker.maybe_refresh(
                rgb=rgb,
                absolute_action_step=absolute_action_step,
                mask_to_world_fn=lambda mask: RSC.tracking_mask_world(
                    mask, depth_disp, self.c2w
                ),
            )
        elif self.latch.regime() == PL.PRE_GRASP:
            self._tracking_result = None
        all_sph = self.fk_all_fn(proprio)
        grip_sph = self.fk_grip_fn(proprio)
        groups, regime = self._current_groups(depth_disp, proprio, all_sph)
        grip_keys = {(tuple(np.round(c, 6)), round(r, 6)) for c, r in grip_sph}
        static_sph = [(c, r) for c, r in all_sph if (tuple(np.round(c, 6)), round(r, 6)) not in grip_keys]
        if self.predictor is None:



            offsets = list(self.offsets_fn(chunk))[:self.execution_window]
            offset_count = int(len(offsets))
            if offset_count < 1:
                raise RuntimeError("EXECUTION_WINDOW_EMPTY_EXECUTION_WINDOW")
            if offset_count > self.execution_window:
                raise RuntimeError(
                    "EXECUTION_WINDOW_OFFSET_COUNT_OUT_OF_RANGE:%d>%d" %
                    (offset_count, self.execution_window))
            unsafe, k, drv = motion_unsafe(
                groups, regime, static_sph, grip_sph, offsets, self.trigger_distance,
                relational_on=self.relational_on,
                floor_gripper=self.trigger_distance_gripper,
                floor_held=self.trigger_distance_held)
            rollout_steps = None
        else:
            expected_count = min(int(len(chunk)), self.execution_window)
            if expected_count < 1:
                raise RuntimeError("EXECUTION_WINDOW_EMPTY_EXECUTION_WINDOW")
            if expected_count > 5:
                raise RuntimeError("ROLLOUT_EXECUTION_WINDOW_EXCEEDS_SIGNED_FIVE")
            rollout_steps = list(self.predictor.predict(chunk[:expected_count]))
            offset_count = int(len(rollout_steps))
            if offset_count != expected_count:
                raise RuntimeError(
                    "ROLLOUT_COUNT_MISMATCH:%d!=%d" %
                    (offset_count, expected_count))
            unsafe, k, drv = motion_unsafe_rollout(
                groups, regime, static_sph, rollout_steps,
                self.trigger_distance, relational_on=self.relational_on,
                floor_gripper=self.trigger_distance_gripper,
                floor_held=self.trigger_distance_held,
                held_cloud_latch=self._held_cloud_latch,
                held_ee_latch=self._held_ee_latch,
                held_rot_latch=self._held_rot_latch)
        if bool(unsafe) != (k is not None):
            raise RuntimeError(
                "EXECUTION_WINDOW_UNSAFE_K_INCONSISTENT:unsafe=%r:k=%r" %
                (bool(unsafe), k))
        if k is None:
            grant_steps = offset_count
            grant_source = "scanned_execution_window_no_unsafe"
        else:
            k = int(k)
            if k < 0 or k >= offset_count or k >= self.execution_window:
                raise RuntimeError(
                    "EXECUTION_WINDOW_K_OUTSIDE_EXECUTION_WINDOW:k=%d:count=%d:C=%d" %
                    (k, offset_count, self.execution_window))
            # The first unsafe offset determines the
            # safe prefix.  Runner consumers must use this recorded grant rather
            # than recomputing it from k.
            grant_steps = min(k, self.execution_window)
            grant_source = "min(first_unsafe_k,execution_window)"
        tel = {"regime": regime, "driver": (list(drv[:2]) if drv else None),
               "triggering_label": (drv[1] if drv else None),
               "clearance_m": (drv[2] if drv else None),
               "trigger_point_xyz": (drv[3] if drv and len(drv) > 3 else None),
               # un-swept, sphere-surface caliber; see _unswept_self_clearance.
               # Never subtract this from clearance_m (swept for the gripper).
               "self_clearance_m": (
                   _unswept_self_clearance(drv[3], all_sph)
                   if drv and len(drv) > 3 else None),
               "k": k,
               "n_other": int(len(groups["OTHER"])),
               "sam3_calls": self.sam3_calls,
               "grounding_batches": self.grounding_batches,
               "floor_m": self.floor,
               "trigger_distance_m": self.trigger_distance,
               "trigger_distance_gripper_m": (
                   self.trigger_distance if self.trigger_distance_gripper is None
                   else self.trigger_distance_gripper),
               "trigger_distance_held_m": (
                   self.trigger_distance if self.trigger_distance_held is None
                   else self.trigger_distance_held),
               "self_band_m": self._self_band,
               "released": bool(self.latch.released),
               "execution_window": self.execution_window,
               "offset_count": offset_count,
               "grant_steps": int(grant_steps),
               "grant_source": grant_source,
               "action_chunk_sha256": hashlib.sha256(
                   np.ascontiguousarray(chunk, dtype=np.float64).tobytes()
               ).hexdigest(),
               "unsafe": bool(unsafe), "first_step": k}
        if rollout_steps is not None:
            tel.update({
                "predictor": "rollout",
                "ee_pos_k": [np.asarray(step["ee_pos_k"], dtype=float).tolist()
                             for step in rollout_steps],
                "ee_quat_k": [np.asarray(step["ee_quat_k"], dtype=float).tolist()
                              for step in rollout_steps],
                "restore_fidelity_ok": bool(
                    self.predictor.last_restore_fidelity_ok),
                "rollout_latency_ms": float(self.predictor.last_latency_ms),
            })
        if self.tracker is not None and self._tracking_result is not None:
            tel.update({key: value for key, value in self._tracking_result.items()
                        if key != "snapshot"})
        if k is None:
            return "pass_through", None, tel
        if grant_steps > 0:
            return "trim_grant", None, tel
        # k=0 is the only evade trigger.  Direction is computed below from the
        # current decision observation; future-offset scans never cache it.
        # evade: move the gripper away from the nearest forbidden hazard point + up
        ee = np.asarray(proprio[7:10], float)
        haz_labels = _hazard_labels(CG.ROBOT_TERMINAL_GRIPPER, regime, self.relational_on)
        haz = _concat(groups, haz_labels)
        if len(haz):
            nearest = haz[int(np.argmin(np.linalg.norm(haz - ee, axis=1)))]
            direction = ee - nearest
            direction[2] = max(direction[2], 0.0)
            n = np.linalg.norm(direction) + 1e-6
            direction = direction / n
            last_g = float(chunk[0][6]) if chunk.ndim == 2 and chunk.shape[1] > 6 else -1.0
            evade = np.array([direction[0], direction[1], max(direction[2], 0.5), 0.0, 0.0, 0.0, last_g])
        else:
            evade = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0])
        return "evade", evade, tel
