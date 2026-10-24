"""Depth-based motion safety adapter.

Build a metric point cloud, remove robot geometry using FK, attach the held
payload, and identify floating threat components. Check swept gripper/payload
clearance along the predicted action path, then slow the safe prefix or apply
a bounded retreat. Repeated same-hazard handbacks can trigger a brief hold.

The floating-component filter excludes table-level components. Arm geometry
outside the gripper cluster is checked at the current pose."""

from __future__ import annotations

import math
import os
import re

import numpy as np

from carryon.libero_safety.contracts import MotionAssessment, SceneGeometry








FROZEN_OVERRIDES = {
    "SHIELD_V2_ACTION_GAIN_M": "0.05",  # OSC metres per unit action per step
    "SHIELD_V2_FLIP_ROWS": "1",         # OpenGL->OpenCV row order (EE-anchor validated)
    "SHIELD_V2_DEBUG_DIR": "",          # scene npz dump dir; "" = off
}
ACTION_GAIN_M = float(os.environ.get("SHIELD_V2_ACTION_GAIN_M", FROZEN_OVERRIDES["SHIELD_V2_ACTION_GAIN_M"]))
# The observation's vla_action_chunk is ALWAYS the executed-scale chunk (any
# checkpoint-specific unit scaling happens runner-side before the observation
# is built), so the sweep predicts from the chunk verbatim.
VOXEL_M = 0.02
FLOAT_CLEAR_M = 0.08          # component lowest point above plane -> floating






SHELL_HALF_M = 0.01
NEAR_HAZARD_M = 0.03





FLOOR_EPSILON_M = 1e-4
PAYLOAD_BOX_M = 0.09          # gripper-frame half-extent for payload capture
GRIPPER_CLOSED_WIDTH = 0.055  # |q0-q1| below this = holding
HARD_MARGIN_M = 0.015         # swept clearance below this = predicted collision



RETREAT_TARGET_CLEARANCE_M = 2.0 * HARD_MARGIN_M
SLOW_ZONE_M = 0.06            # clearance below this scales the chunk
SLOW_SCALE = 0.4
WAIT_STEPS = 20
RETREAT_MAX_STEPS = 3


def effective_config():

    return {
        name: os.environ.get(name, default)
        for name, default in FROZEN_OVERRIDES.items()
    }


def assert_formal_defaults():
    """Formal legs run with zero overrides; fail closed otherwise."""
    drifted = {
        name: os.environ[name]
        for name, default in FROZEN_OVERRIDES.items()
        if os.environ.get(name, default) != default
    }
    if drifted:
        raise RuntimeError(f"formal leg forbids shield-v2 overrides: {drifted}")

_PANDA = {"model": None}
_DEBUG_STATE = {"n": 0}
_TELEMETRY = []
_ERRORS = []




_EMERGENCY_NEXT_CHECK = {}
_CUROBO_SPHERES = {"map": None}


_CUROBO_YML_RELPATH = os.path.join("safety_eval", "franka_attach32.yml")


def drain_telemetry():
    """Return and clear per-step telemetry + error records (runner hook)."""
    out = {"telemetry": list(_TELEMETRY), "errors": list(_ERRORS)}
    _TELEMETRY.clear()
    _ERRORS.clear()
    return out
_LINK_TO_BODY = {
    "panda_link0": "robot0_link0",
    "panda_link1": "robot0_link1",
    "panda_link2": "robot0_link2",
    "panda_link3": "robot0_link3",
    "panda_link4": "robot0_link4",
    "panda_link5": "robot0_link5",
    "panda_link6": "robot0_link6",
    "panda_link7": "robot0_link7",
    "panda_hand": "robot0_right_hand",
    "panda_leftfinger": "gripper0_leftfinger",
    "panda_rightfinger": "gripper0_rightfinger",
}
_QUERY_LINKS = ("panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger")


def _curobo_sphere_map():
    """Load per-link collision spheres from the supplied robot specification."""
    if _CUROBO_SPHERES["map"] is None:
        import hashlib

        import yaml

        path = os.path.join(os.getcwd(), _CUROBO_YML_RELPATH)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"shield-v2 dependency missing: {_CUROBO_YML_RELPATH} "
                "(run from the repository root)"
            )
        blob = open(path, "rb").read()
        raw = yaml.safe_load(blob.decode("utf-8"))

        def find_spheres(node):
            if isinstance(node, dict):
                if "collision_spheres" in node and isinstance(node["collision_spheres"], dict):
                    return node["collision_spheres"]
                for value in node.values():
                    found = find_spheres(value)
                    if found is not None:
                        return found
            return None

        spheres = find_spheres(raw) or {}
        _CUROBO_SPHERES["map"] = {
            link: [(np.asarray(e["center"], dtype=np.float64), float(e["radius"])) for e in entries]
            for link, entries in spheres.items()
            if link in _LINK_TO_BODY
        }
    return _CUROBO_SPHERES["map"]


def _robot_model():
    """Robot-spec-only FK model (arm + gripper), no scene, no task."""
    if _PANDA["model"] is None:
        import mujoco
        import robosuite
        from robosuite.models.robots import Panda
        from robosuite.models.grippers import gripper_factory

        robot = Panda()
        robot.add_gripper(gripper_factory("PandaGripper"))
        model = mujoco.MjModel.from_xml_string(robot.get_xml())
        data = mujoco.MjData(model)
        _PANDA["model"] = (mujoco, model, data)
    return _PANDA["model"]


def _fk_spheres(proprioception):
    """Return robot spheres (x, y, z, radius) at the observed joint state.

    The proprioception layout is seven joint values, three end-effector position
    values, four quaternion values and two gripper values. FK uses the robot base
    frame; the benchmark places that base at the world origin."""
    mujoco, model, data = _robot_model()
    q = np.asarray(proprioception, dtype=np.float64)
    joints = q[:7]
    grip = q[14:16] if q.size >= 16 else np.array([0.04, -0.04])
    data.qpos[:7] = joints
    if data.qpos.size >= 9:
        data.qpos[7:9] = grip
    mujoco.mj_forward(model, data)
    # The FK model has its base at the origin; the deployment reports EE pose
    # in the world frame. Anchor by aligning the FK EE body to the proprio EE
    # (base offset is a translation for this fixed, unrotated mount).
    ee_world = q[7:10] if q.size >= 10 else None
    shift = np.zeros(3)
    if ee_world is not None:
        try:
            fk_ee = np.array(data.body(model.body("gripper0_eef").id).xpos)
            shift = np.asarray(ee_world, dtype=np.float64) - fk_ee
        except Exception:
            shift = np.zeros(3)
    # Compose the curobo tight sphere spec through the FK link poses. This
    # replaces geom rbound bounding spheres (a palm-mesh rbound is ~9cm and
    # inflates every clearance by that much).
    sphere_map = _curobo_sphere_map()
    spheres = []
    gripper_spheres = []
    for link, entries in sphere_map.items():
        body_name = _LINK_TO_BODY[link]
        try:
            bid = model.body(body_name).id
        except Exception:
            continue
        pos = np.array(data.xpos[bid])
        rot = np.array(data.xmat[bid]).reshape(3, 3)
        for center_local, radius in entries:
            center = pos + rot @ center_local + shift
            spheres.append((center, radius))
            if link in _QUERY_LINKS:
                gripper_spheres.append((center, radius))
    if not spheres or not gripper_spheres:

        raise RuntimeError("curobo sphere spec resolved to an empty robot model")
    return spheres, gripper_spheres


def _real_depth(depth, calibration):
    near = float(calibration["near_m"])
    far = float(calibration["far_m"])
    z = np.clip(np.asarray(depth, dtype=np.float64), 0.0, 1.0)
    return near / (1.0 - z * (1.0 - near / far))








SELF_DEPTH_TOLERANCE_M = 0.02
# Declared robot+mount render asset (stripped from the benchmark model: robot,
# gripper and mount bodies only; scene objects and their welds/contacts/sensors
# removed). Bound by SHA in the packet manifest.
SELF_MODEL_ASSET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "self_model_assets", "robot_mount.xml"
)
SELF_RENDER_STRIDE = 1       # ray-cast every pixel (full-resolution silhouette;











SELF_BACKSTOP_M = 0.02       # Fixed clearance margin.
_SELF_MODEL = {"model": None, "data": None, "robot_geom": None}
_SELF_DEPTH = {}             # per-build handoff from classifier to unprojection


def _load_self_model():
    """Load the robot and mount geometry for the expected-depth model."""
    if _SELF_MODEL["model"] is None:
        import hashlib

        import mujoco

        if not os.path.isfile(SELF_MODEL_ASSET):
            raise RuntimeError(f"self-model asset missing: {SELF_MODEL_ASSET}")
        blob = open(SELF_MODEL_ASSET, "rb").read()
        model = mujoco.MjModel.from_xml_string(blob.decode("utf-8"))
        data = mujoco.MjData(model)
        robot_geom = np.array([
            model.body(model.geom_bodyid[g]).name.startswith(("robot0", "gripper0", "mount0"))
            for g in range(model.ngeom)
        ])
        _SELF_MODEL.update(model=model, data=data, robot_geom=robot_geom)
    return _SELF_MODEL["model"], _SELF_MODEL["data"], _SELF_MODEL["robot_geom"]


def _render_expected_robot_depth(proprioception, calibration, shape):
    """Expected robot+mount depth on the oriented sensor grid (NaN off-robot).

    Policy-visible inputs only: proprioception joints, the declared robot+mount
    geometry, and the recorded camera calibration. The base is EE-anchored so a
    tiny mounting offset cannot shift the silhouette. Rays are cast in the same
    camera frame `_unproject` inverts, so the expected grid registers with the
    observed grid pixel-for-pixel.
    """
    import mujoco

    model, data, robot_geom = _load_self_model()
    q = np.asarray(proprioception, dtype=np.float64)
    ee = q[7:10]
    for k in range(7):
        try:
            adr = model.joint("robot0_joint%d" % (k + 1)).qposadr[0]
            data.qpos[adr] = q[k]
        except Exception:
            pass
    # Gripper finger joints from proprio[14:16] (NAMED addressing; a default-
    # open render of moving fingers otherwise mis-registers the gripper and
    # leaks robot pixels as near-EE threats). qposadr is resolved by joint
    # name, never a hand-computed index.
    if q.size >= 16:
        for name, value in (("gripper0_finger_joint1", q[14]),
                            ("gripper0_finger_joint2", q[15])):
            try:
                data.qpos[model.joint(name).qposadr[0]] = value
            except Exception:
                pass
    mujoco.mj_forward(model, data)
    base = model.body("robot0_base").id
    fk_ee = np.array(data.body(model.body("gripper0_eef").id).xpos)
    model.body_pos[base] = model.body_pos[base] + (ee - fk_ee)   # EE anchor (translation)
    mujoco.mj_forward(model, data)

    h, w = shape
    K = np.asarray(calibration["K"], dtype=np.float64)
    T = np.asarray(calibration["world_from_camera"], dtype=np.float64)
    R = T[:3, :3]
    cam = T[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    stride = SELF_RENDER_STRIDE
    exp = np.full((h, w), np.nan)
    gid = np.zeros(1, dtype=np.int32)
    for v in range(0, h, stride):
        row = exp[v]
        for u in range(0, w, stride):
            dc = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            dw = dc @ R.T
            n = float(np.linalg.norm(dw))
            dist = mujoco.mj_ray(model, data, cam, dw / n, None, 1, -1, gid)
            if dist >= 0 and gid[0] >= 0 and robot_geom[gid[0]]:
                d_perp = dist / n
                row[u] = d_perp
                if stride > 1:  # fill the stride block (nearest-neighbour)
                    row[u:u + stride] = d_perp
                    for vv in range(v + 1, min(v + stride, h)):
                        exp[vv, u:u + stride] = d_perp
    return exp


def _backstop_self_filter(points, spheres):
    """Remove points within SELF_BACKSTOP_M of the nearest robot-sphere surface.

    Return (kept_points, n_deleted). Only points with strictly greater surface
    distance survive; obstacles inside that band are also removed."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or not spheres:
        return points, 0
    surf = np.full(len(points), np.inf)
    for center, radius in spheres:
        surf = np.minimum(surf, np.linalg.norm(points - np.asarray(center), axis=1) - float(radius))
    keep = surf > SELF_BACKSTOP_M
    return points[keep], int((~keep).sum())


def _classify_self_depth(observed_depth_m, expected_robot_depth_m):

    observed = np.asarray(observed_depth_m, dtype=np.float64)
    expected = np.asarray(expected_robot_depth_m, dtype=np.float64)
    valid = np.isfinite(expected)
    diff = observed - expected
    # eps is sub-micron float hygiene only (a `1.0 - tolerance` literal is not
    # exactly representable); it makes the self-side boundary robustly
    # inclusive and never loosens the physical tolerance or certifies safety.
    eps = 1e-9
    foreground = valid & (diff < -(SELF_DEPTH_TOLERANCE_M + eps))
    self_mask = valid & (np.abs(diff) <= SELF_DEPTH_TOLERANCE_M + eps)
    return self_mask, foreground


def relational_robot_mount_self_masks(proprioception, observed_metric_depth, calibration):

    observed = np.asarray(observed_metric_depth, dtype=np.float64)
    if observed.ndim != 2:
        raise ValueError("relational self depth must be a 2-D metric array")
    try:
        expected = _render_expected_robot_depth(
            proprioception, calibration, observed.shape
        )
        return _classify_self_depth(observed, expected)
    except Exception as exc:
        import traceback

        _ERRORS.append({
            "where": "relational_expected_depth_self_model",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        })
        empty = np.zeros(observed.shape, dtype=bool)
        return empty, empty.copy()


def _oriented_metric_depth(depth, calibration):
    """Metric depth in the SAME orientation `_unproject` unprojects from."""
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    rotation = int(calibration.get("image_rotation_deg", 0)) % 360
    if rotation == 180:
        depth = depth[::-1, ::-1]
    elif rotation != 0:
        raise ValueError(f"unsupported image rotation {rotation}")
    if bool(int(os.environ.get("SHIELD_V2_FLIP_ROWS", FROZEN_OVERRIDES["SHIELD_V2_FLIP_ROWS"]))):
        depth = depth[::-1]
    return _real_depth(depth, calibration)


def _unproject(depth, calibration):
    """Metric world-frame cloud from the (possibly rotated) sensor depth."""
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    rotation = int(calibration.get("image_rotation_deg", 0)) % 360
    if rotation == 180:
        depth = depth[::-1, ::-1]
    elif rotation != 0:
        raise ValueError(f"unsupported image rotation {rotation}")
    if bool(int(os.environ.get("SHIELD_V2_FLIP_ROWS", "1"))):
        # The benchmark robot base coincides with the world origin.
        depth = depth[::-1]
    dist = _real_depth(depth, calibration)
    K = np.asarray(calibration["K"], dtype=np.float64)
    T = np.asarray(calibration["world_from_camera"], dtype=np.float64)
    h, w = dist.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    zc = dist
    # robosuite get_camera_extrinsic_matrix applies an OpenCV axis correction:
    # +x right, +y down, +z along the viewpoint.
    xc = (us - cx) / fx * zc
    yc = (vs - cy) / fy * zc
    pts_cam = np.stack([xc, yc, zc], axis=-1).reshape(-1, 3)
    keep = (zc.reshape(-1) > 0.05) & (zc.reshape(-1) < float(calibration["far_m"]) * 0.98)
    pts_cam = pts_cam[keep]
    pts_world = pts_cam @ T[:3, :3].T + T[:3, 3]
    return pts_world


def _voxel_downsample(points, voxel=VOXEL_M):
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(index)]


def _connected_components(points, radius=0.035):
    """Greedy voxel-hash components over the downsampled cloud."""
    if len(points) == 0:
        return []
    cell = radius
    keys = np.floor(points / cell).astype(np.int64)
    from collections import defaultdict

    grid = defaultdict(list)
    for i, key in enumerate(map(tuple, keys)):
        grid[key].append(i)
    seen = np.zeros(len(points), dtype=bool)
    components = []
    for start in range(len(points)):
        if seen[start]:
            continue
        stack, members = [start], []
        seen[start] = True
        while stack:
            i = stack.pop()
            members.append(i)
            kx, ky, kz = keys[i]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in grid.get((kx + dx, ky + dy, kz + dz), ()):
                            if not seen[j]:
                                seen[j] = True
                                stack.append(j)
        components.append(np.asarray(members, dtype=np.int64))
    return components


def _durable(where, fn, *args, **kwargs):
    """Run fn; on exception append a durable traceback record, then re-raise
    (F7: no judge/recovery path may fail without an artifact trace)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        import traceback

        _ERRORS.append({
            "where": where,
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        })
        raise


def semantic_infer(**kwargs):
    return _durable("semantic_infer", _semantic_infer_inner, **kwargs)


def _semantic_infer_inner(**kwargs):
    """Deterministic FSHOA/TSA semantics: proceed; parse task contact roles.

    No VLM call: the motion judge needs no named obstacles for these suites
    (method contract), and refusal semantics belong to the SSR line. The
    manipulation intent is parsed from the instruction text alone.
    """
    instruction = str(kwargs.get("instruction", "")).strip().lower()
    payload, destination = None, None
    # placement-verb-anchored destination first ("... place it on the plate")
    m = re.search(
        r"(?:pick up|put|place|move)\s+(?:both\s+)?the\s+(.+?)(?:\s+on\s+the\s+\w+)?"
        r"\s+and\s+(?:place|put)\s+it\s+(?:in|on|into|onto)\s+(?:the\s+)?(.+)$",
        instruction,
    )
    if not m:
        m = re.search(
            r"(?:pick up|put|place|move)\s+(?:both\s+)?the\s+(.+?)\s+"
            r"(?:in|on|into|onto)\s+(?:the\s+)?(.+)$",
            instruction,
        )
    if m:
        payload = m.group(1).strip()
        destination = re.sub(r"^(?:the|a)\s+", "", m.group(2).strip())
        destination = re.sub(r"\s+and\s+close.*$", "", destination)
    if not payload:
        payload = "task object"
    manipulation = {"payload_query": payload, "destination_query": destination}
    return {"decision": "proceed", "targets": [], "manipulation": manipulation}


def build_scene(observation, semantic_assessment):
    try:
        return _build_scene_inner(observation, semantic_assessment)
    except Exception as exc:
        import traceback

        _ERRORS.append({
            "where": "build_scene",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        })
        raise


def _build_scene_inner(observation, semantic_assessment):
    depth = np.asarray(observation.depth, dtype=np.float64)
    calibration = observation.camera_calibration







    _SELF_DEPTH.clear()
    self_stats = {"n_self_px": 0, "n_foreground_px": 0, "n_invalid_expected": 0,
                  "n_edge_erosion": 0, "render_ok": False}
    try:
        oriented = _oriented_metric_depth(depth, calibration)
        expected = _render_expected_robot_depth(
            observation.proprioception, calibration, oriented.shape
        )
        self_mask, fg_mask = _classify_self_depth(oriented, expected)
        far = float(calibration["far_m"])
        in_range = (oriented > 0.05) & (oriented < far * 0.98)
        _SELF_DEPTH.update(
            self_flat=self_mask.reshape(-1),
            fg_flat=fg_mask.reshape(-1),
            keep_flat=in_range.reshape(-1),
            n_kept=int(in_range.sum()),
        )
        self_stats.update(
            n_self_px=int(self_mask.sum()),
            n_foreground_px=int(fg_mask.sum()),
            n_invalid_expected=int((~np.isfinite(expected)).sum()),
            render_ok=True,
        )
    except Exception as exc:  # fail-closed self model
        import traceback

        _ERRORS.append({
            "where": "expected_depth_self_model",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        })

    spheres, gripper_spheres = _fk_spheres(observation.proprioception)

    raw_points = _unproject(depth, calibration)
    foreground_points = None
    n_backstop_deleted = 0
    # Apply pixel-level self removal only when point count matches the kept pixels.
    # Otherwise, use the point-cloud filtering path.
    sd = _SELF_DEPTH
    if sd and raw_points.shape[0] == sd["n_kept"]:
        kept = sd["keep_flat"]
        pt_self = sd["self_flat"][kept]
        pt_fg = sd["fg_flat"][kept]
        keep_pts = ~pt_self
        foreground_points = raw_points[keep_pts & pt_fg]
        raw_points = raw_points[keep_pts]
        # Round-3 dynamic-self backstop (real depth path only).
        raw_points, n_backstop_deleted = _backstop_self_filter(raw_points, spheres)
    points = _voxel_downsample(raw_points)
    proprio = np.asarray(observation.proprioception, dtype=np.float64)
    ee = proprio[7:10]
    grip_q = proprio[14:16] if proprio.size >= 16 else np.array([0.04, -0.04])
    holding = abs(float(grip_q[0]) - float(grip_q[1])) < GRIPPER_CLOSED_WIDTH

    # support plane = dominant low height (table); robust percentile
    plane_z = float(np.percentile(points[:, 2], 20)) if len(points) else 0.0





    n_pts = len(points)
    s_min = np.full(n_pts, np.inf)
    shell_membership = np.zeros((len(spheres), n_pts), dtype=bool)
    for sphere_index, (center, radius) in enumerate(spheres):
        signed = np.linalg.norm(points - center, axis=1) - radius
        s_min = np.minimum(s_min, signed)
        shell_membership[sphere_index] = np.abs(signed) <= SHELL_HALF_M
    shell_point = np.abs(s_min) <= SHELL_HALF_M
    subsurface_point = s_min < -SHELL_HALF_M

    delete_mask = np.zeros(n_pts, dtype=bool)
    keep_components = []
    if n_pts:
        components = _connected_components(points)
        for members in components:
            member_mask = np.zeros(n_pts, dtype=bool)
            member_mask[members] = True
            # A "surface-shell component": every member point lies in the
            # shell (no subsurface, no outside-shell mass).
            if bool(shell_point[members].all()):
                spans = int(shell_membership[:, members].any(axis=1).sum())
                if spans >= 2:
                    delete_mask |= member_mask
                    continue
            keep_components.append(members)
    self_mask = delete_mask  # provenance name kept for the debug dump
    n_evidence_preserved_in_shell = int(
        ((shell_point | subsurface_point) & ~delete_mask).sum()
    )

    # Floating classification over PRESERVED components only.
    scene_points = points[~delete_mask]
    old_to_new = np.cumsum(~delete_mask) - 1
    scene_float = np.zeros(len(scene_points), dtype=bool)
    for members in keep_components:
        kept = [i for i in members if not delete_mask[i]]
        if len(kept) < 6:
            continue
        component = points[kept]
        if float(component[:, 2].min()) > plane_z + FLOAT_CLEAR_M:
            scene_float[old_to_new[kept]] = True
    float_points = scene_points[scene_float]




    payload_mask_local = np.zeros(len(scene_points), dtype=bool)
    if holding and len(scene_points):
        rel = np.abs(scene_points - ee)
        in_box = (
            (rel[:, 0] < PAYLOAD_BOX_M)
            & (rel[:, 1] < PAYLOAD_BOX_M)
            & (rel[:, 2] < PAYLOAD_BOX_M * 1.5)
        )
        payload_mask_local = in_box & ~scene_float
    payload_points = scene_points[payload_mask_local]

    threat_points = float_points
    # Foreground pixels are NOT force-classified as threats. Per the amendment
    # they must SURVIVE unprojection and clustering (they are never self-
    # deleted, so they remain in `points` above); the unchanged downstream
    # floating-component + floor logic then decides threat-hood. Scene surfaces
    # (table, workspace) that are merely nearer than a phantom side projection
    # of the arm/mount rest at/below the support plane and are correctly not
    # floating, so they do not flood the threat set.
    # Near-hazard measurements: threat points pressed near the gripper
    # or inside the payload box are counted and their minimum EE distance
    # recorded; they remain in the threat set.
    near_hazard = {
        "n_band": 0,
        "n_in_payload_box": 0,
        "min_d_ee": None,
        "n_self_deleted": int(delete_mask.sum()),
        "n_evidence_preserved_in_shell": n_evidence_preserved_in_shell,
    }
    if len(threat_points):
        d_ee = np.linalg.norm(threat_points - ee, axis=1)
        near_hazard["min_d_ee"] = round(float(d_ee.min()), 4)

        for center, radius in gripper_spheres:
            d = np.linalg.norm(threat_points - center, axis=1)
            near_hazard["n_band"] += int(
                ((d >= radius + SHELL_HALF_M) & (d < radius + NEAR_HAZARD_M)).sum()
            )
        if holding:
            rel_t = np.abs(threat_points - ee)
            near_hazard["n_in_payload_box"] = int((
                (rel_t[:, 0] < PAYLOAD_BOX_M)
                & (rel_t[:, 1] < PAYLOAD_BOX_M)
                & (rel_t[:, 2] < PAYLOAD_BOX_M * 1.5)
            ).sum())
    # Durable emission (Clause B): the counter reaches the artifact through
    # the runner's telemetry drain, not only the in-memory scene dict.
    _TELEMETRY.append({
        "check": "scene",
        "n_cloud": int(n_pts),
        "n_self_deleted": int(delete_mask.sum()),
        "n_evidence_preserved_in_shell": n_evidence_preserved_in_shell,
        "n_threat_points": int(len(threat_points)),
        "near_hazard": dict(near_hazard),
        "holding": bool(holding),

        "self_depth": dict(self_stats),
        # Round-3 dynamic-self backstop count (real depth path).
        "n_backstop_deleted": int(n_backstop_deleted),
    })

    debug_dir = os.environ.get("SHIELD_V2_DEBUG_DIR", FROZEN_OVERRIDES["SHIELD_V2_DEBUG_DIR"])
    if debug_dir:
        state = _DEBUG_STATE
        if state["n"] < 12:
            from pathlib import Path

            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                Path(debug_dir) / f"scene_{state['n']:03d}.npz",
                points=points.astype(np.float32),
                self_mask=self_mask,
                threat_points=threat_points.astype(np.float32),
                payload_points=payload_points.astype(np.float32),
                plane_z=plane_z,
                ee=ee,
                holding=holding,
            )
            state["n"] += 1
    return SceneGeometry(
        world={
            "threat_points": threat_points,
            "plane_z": plane_z,
            "n_cloud": int(len(points)),
            "n_self": int(self_mask.sum()),
            "n_evidence_preserved_in_shell": n_evidence_preserved_in_shell,
            "near_hazard": near_hazard,
            "holding": bool(holding),
        },
        grounded_targets={},
        payload_geometry=payload_points if holding else None,
    )


def _swept_query_spheres(proprioception, payload_points):
    _, gripper_spheres = _fk_spheres(proprioception)
    query = [(np.array(c), float(r)) for c, r in gripper_spheres]
    if payload_points is not None and len(payload_points):
        center = payload_points.mean(axis=0)
        radius = float(np.linalg.norm(payload_points - center, axis=1).max())
        query.append((center, min(radius, 0.14)))
    return query


def _predicted_offsets(candidate_actions):
    offsets = [np.zeros(3)]
    acc = np.zeros(3)
    for action in candidate_actions:
        delta = np.clip(np.asarray(action[:3], dtype=np.float64), -1.0, 1.0)
        acc = acc + delta * ACTION_GAIN_M
        offsets.append(acc.copy())
    return offsets[1:]


# Classify each stop-driving threat point geometrically: dynamic robot-surface
# residual, static mount/base residual, or external foreground obstacle.
MOUNT_REGION_X = -0.5        # world x behind this = robot base/mount column
DYNAMIC_SELF_BAND_M = SELF_BACKSTOP_M + 0.02   # on-arm surface + tolerance


def _named_cause(point, all_spheres):
    p = np.asarray(point, dtype=np.float64)
    surf = min(
        (float(np.linalg.norm(p - np.asarray(c))) - float(r) for c, r in all_spheres),
        default=np.inf,
    )
    if surf <= DYNAMIC_SELF_BAND_M:
        return "dynamic_self_leak", round(surf, 4)
    if float(p[0]) < MOUNT_REGION_X:
        return "static_self_leak", round(surf, 4)
    return "foreground_external", round(surf, 4)


def _classify_stop_cause(row):
    """Classify the minimum-clearance threat point by its geometric cause."""
    band = float(DYNAMIC_SELF_BAND_M)
    if not np.isfinite(band):
        raise ValueError("route-B DYNAMIC_SELF_BAND_M must be finite")
    for key in ("bound_by", "approach_delta_m", "static_surf_m", "swept_surf_m", "point"):
        if key not in row or row[key] is None:
            raise KeyError("route-B stop-cause requires %r" % key)
    for key in ("approach_delta_m", "static_surf_m", "swept_surf_m"):
        if not np.isfinite(float(row[key])):
            raise ValueError("route-B stop-cause requires finite %r" % key)
    bound_by = row["bound_by"]
    if bound_by not in ("static_arm", "swept_approach"):
        raise ValueError("route-B unknown bound_by %r" % (bound_by,))
    try:
        pvec = [float(v) for v in row["point"]]
    except (TypeError, ValueError):
        raise ValueError("route-B point must be a numeric sequence")
    if len(pvec) != 3 or not all(np.isfinite(v) for v in pvec):
        raise ValueError("route-B point must be a finite length-3 vector")
    static_surf = float(row["static_surf_m"])
    if static_surf <= band:
        if pvec[0] < MOUNT_REGION_X:
            return "static_self_leak"
        return "dynamic_self_leak"
    if bound_by == "swept_approach" and float(row["approach_delta_m"]) > 0.0:
        return "goal_approach_external"
    return "foreground_external"


def _attribute_threats(threats, all_spheres):
    """Three-way named-cause counts over all threat points + the closest-to-
    any-robot-sphere point (the one most likely to drive a below-floor stop)."""
    counts = {"dynamic_self_leak": 0, "static_self_leak": 0, "foreground_external": 0}
    if not len(threats):
        return counts, None, None
    surf = np.full(len(threats), np.inf)
    for c, r in all_spheres:
        surf = np.minimum(surf, np.linalg.norm(threats - np.asarray(c), axis=1) - float(r))
    xs = threats[:, 0]
    dyn = surf <= DYNAMIC_SELF_BAND_M
    stat = (~dyn) & (xs < MOUNT_REGION_X)
    ext = ~dyn & ~stat
    counts["dynamic_self_leak"] = int(dyn.sum())
    counts["static_self_leak"] = int(stat.sum())
    counts["foreground_external"] = int(ext.sum())
    i = int(np.argmin(surf))
    cause, s = _named_cause(threats[i], all_spheres)
    return counts, cause, {"point": [round(float(v), 3) for v in threats[i]], "surf_to_robot_m": s}


def check_motion(**kwargs):
    try:
        return _check_motion_inner(**kwargs)
    except Exception as exc:  # durable record on every judge path (F7)
        import traceback

        _ERRORS.append({
            "where": "check_motion",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        })
        raise


def _check_motion_inner(**kwargs):
    scene = kwargs["scene_geometry"]
    proprioception = kwargs["proprioception"]
    candidate_actions = kwargs["candidate_actions"]
    threats = scene.world["threat_points"]
    if len(threats) == 0:
        emergency = bool(_EMERGENCY_NEXT_CHECK.pop("armed", False))
        _TELEMETRY.append({
            "check": "emergency_recovery" if emergency else "motion",
            "emergency_recovery": emergency,
            "start_clearance_m": None,
            "min_swept_clearance_m": None,
            "n_threat_points": 0,
            "n_query_spheres": None,
            "status": "safe",
            "first_unsafe_step": None,
            "intervention_steps": len(candidate_actions) if emergency else 0,
            "named_cause": None,
            "named_cause_point": None,
            "cause_counts": {"dynamic_self_leak": 0, "static_self_leak": 0,
                             "foreground_external": 0},
        })
        return MotionAssessment(status="safe")
    query = _swept_query_spheres(proprioception, scene.payload_geometry)
    if not query:
        return MotionAssessment(status="unknown", violation="perception_failure")





    all_spheres, grip_spheres = _fk_spheres(proprioception)
    grip_keys = {(tuple(np.round(c, 6)), round(r, 6)) for c, r in grip_spheres}
    static_spheres = [
        (c, r) for c, r in all_spheres
        if (tuple(np.round(c, 6)), round(r, 6)) not in grip_keys
    ]
    arm_clear = min(
        (
            float(np.min(np.linalg.norm(threats - center, axis=1))) - radius
            for center, radius in static_spheres
        ),
        default=float("inf"),
    )

    def min_clearance(offset):
        best = arm_clear
        for center, radius in query:
            d = float(np.min(np.linalg.norm(threats - (center + offset), axis=1))) - radius
            if d < best:
                best = d
        return best

    # CLEARANCE FLOOR, independent of any velocity/TTC reasoning, boundary
    # INCLUSIVE (gate wording: clearance <= frozen_margin is unsafe). The
    # policy-chunk classifier has NO sequence-level exemption: a predicted
    # step at or below the floor is never safe (Packet 2.2 charter F2).
    # Escapes are a RESPONSE-layer concern: recovery sequences returned by
    # avoid_then_handback/cap_recovery are validated under an explicit
    # emergency-recovery flag with escape semantics (strictly opening
    # clearance, ending above the floor) and are counted as intervention
    # steps in telemetry.
    offsets = _predicted_offsets(candidate_actions)
    clearances = [min_clearance(offset) for offset in offsets]
    start_clearance = min_clearance(np.zeros(3))
    emergency = bool(_EMERGENCY_NEXT_CHECK.pop("armed", False))



    floor = HARD_MARGIN_M + FLOOR_EPSILON_M
    if not clearances:
        verdict = MotionAssessment(status="safe")
    elif emergency:
        verdict = MotionAssessment(status="safe")
        previous = start_clearance
        for k, clearance in enumerate(clearances):
            if clearance < previous - 1e-9:
                verdict = MotionAssessment(
                    status="unsafe",
                    violation="predicted_collision",
                    first_unsafe_step=k,
                )
                break
            previous = clearance
        if verdict.status == "safe" and clearances[-1] <= floor:
            verdict = MotionAssessment(
                status="unsafe",
                violation="predicted_collision",
                first_unsafe_step=len(clearances) - 1,
            )
    else:
        verdict = MotionAssessment(status="safe")
        if start_clearance <= floor:
            # The CURRENT state is already at/below the floor: no policy
            # chunk is safe from here — escape belongs to the flagged
            # response layer, never to the policy classifier.
            verdict = MotionAssessment(
                status="unsafe",
                violation="predicted_collision",
                first_unsafe_step=0,
            )
        else:
            for k, clearance in enumerate(clearances):
                if clearance <= floor:
                    verdict = MotionAssessment(
                        status="unsafe",
                        violation="predicted_collision",
                        first_unsafe_step=k,
                    )
                    break
    # Attribute the point with minimum clearance over both arm-static and swept
    # gripper/payload queries. cause_counts describes the whole-scene composition.
    all_curobo, _ = _fk_spheres(proprioception)
    cause_counts, _, _ = _attribute_threats(np.asarray(threats), all_curobo)
    stop_cause = stop_point = None
    if verdict.status != "safe" and len(threats):
        tpts = np.asarray(threats)
        # Per-threat static arm-link clearance (offset-invariant).
        static_clr = np.full(len(tpts), np.inf)
        for center, radius in static_spheres:
            static_clr = np.minimum(
                static_clr, np.linalg.norm(tpts - np.asarray(center), axis=1) - radius
            )
        # Per-threat swept gripper/payload clearance at the current pose and at
        # its minimum over the predicted sweep.
        def _swept(offset):
            c = np.full(len(tpts), np.inf)
            for center, radius in query:
                c = np.minimum(
                    c, np.linalg.norm(tpts - (np.asarray(center) + offset), axis=1) - radius
                )
            return c

        swept0 = _swept(np.zeros(3))
        swept_min = swept0.copy()
        for offset in offsets:
            swept_min = np.minimum(swept_min, _swept(offset))
        total_min = np.minimum(static_clr, swept_min)
        di = int(np.argmin(total_min))
        driver = tpts[di]
        s = round(
            float(
                min(
                    (float(np.linalg.norm(driver - np.asarray(c))) - float(r)
                     for c, r in all_curobo),
                    default=np.inf,
                )
            ),
            4,
        )
        # Runtime self-vs-approach discriminator (physics, not geometry): a
        # world-fixed obstacle the gripper CLOSES ON has swept clearance that
        # drops along the sweep (approach_delta > 0); a robot-fixed self leak
        # rides with the arm (constant swept clearance) or is bound by the
        # static arm term (approach_delta == 0). This separates goal-as-obstacle
        # (external target in the gripper's approach band) from a true dynamic
        # self leak, which the geometric named_cause alone cannot: both sit
        # within DYNAMIC_SELF_BAND_M of a robot sphere.
        bound_by = (
            "swept_approach" if swept_min[di] < static_clr[di] - 1e-9 else "static_arm"
        )
        approach_delta = float(min(swept0[di], static_clr[di]) - total_min[di])
        # static_surf_m is the driver's clearance to the nearest STATIC (non-
        # gripper) arm sphere. It is the decisive self-vs-external term: a
        # leaked arm-surface point rides on a static link (static_surf ~ 0,
        # i.e. <= the arm visual-surface shell ~0.02); a world-fixed external
        # object the gripper reaches toward is far from every static arm sphere
        # (static_surf large). swept_surf_m is the current-pose clearance to
        # the nearest gripper sphere for reference.
        stop_point = {
            "point": [round(float(v), 3) for v in driver],
            "surf_to_robot_m": s,
            "clearance_m": round(float(total_min[di]), 4),
            "bound_by": bound_by,
            "approach_delta_m": round(approach_delta, 4),
            "static_surf_m": round(float(static_clr[di]), 4),
            "swept_surf_m": round(float(swept0[di]), 4),
        }
        stop_cause = _classify_stop_cause(stop_point)
    _TELEMETRY.append({
        "check": "emergency_recovery" if emergency else "motion",
        "emergency_recovery": emergency,
        "start_clearance_m": round(float(start_clearance), 4),
        "min_swept_clearance_m": round(float(min(clearances)), 4) if clearances else None,
        "n_threat_points": int(len(threats)),
        "n_query_spheres": int(len(query)),
        "status": verdict.status,
        "first_unsafe_step": verdict.first_unsafe_step,
        "intervention_steps": len(candidate_actions) if emergency else 0,
        # named-cause channel
        "named_cause": stop_cause if verdict.status != "safe" else None,
        "named_cause_point": stop_point if verdict.status != "safe" else None,
        "cause_counts": cause_counts,
    })
    return verdict


def _retreat_actions(clearance, nearest, ee, width, gripper):

    travel = max(0.0, RETREAT_TARGET_CLEARANCE_M - float(clearance))
    if travel <= 0.0:
        return []
    away = np.asarray(ee, dtype=np.float64) - np.asarray(nearest, dtype=np.float64)
    away[2] = max(away[2], 0.0)
    norm = float(np.linalg.norm(away))
    if norm < 1e-9:
        away, norm = np.array([0.0, 0.0, 1.0]), 1.0
    away = away / norm
    n_steps = max(1, int(math.ceil(travel / ACTION_GAIN_M - 1e-12)))
    per_step_action = travel / (n_steps * ACTION_GAIN_M)
    if per_step_action > 1.0 + 1e-9:
        raise RuntimeError("retreat step exceeds controller range")
    action = [0.0] * width
    action[0:3] = (away * per_step_action).tolist()
    action[-1] = gripper
    return [list(action) for _ in range(n_steps)]


def _current_clearance(scene, proprioception):
    threats = scene.world["threat_points"]
    if len(threats) == 0:
        return None, None
    # Full-arm current clearance (Clause-B judge reachability); the nearest
    # threat point steers the retreat regardless of which link it presses.
    all_spheres, _ = _fk_spheres(proprioception)
    payload = scene.payload_geometry
    query = list(all_spheres)
    if payload is not None and len(payload):
        center = payload.mean(axis=0)
        radius = float(np.linalg.norm(payload - center, axis=1).max())
        query.append((center, min(radius, 0.14)))
    best, best_point = None, None
    for center, radius in query:
        d = np.linalg.norm(threats - center, axis=1) - radius
        i = int(np.argmin(d))
        if best is None or d[i] < best:
            best, best_point = float(d[i]), threats[i]
    return best, best_point


def avoid_then_handback(observation, semantic, motion, library):
    return _durable(
        "avoid_then_handback", _avoid_then_handback_inner,
        observation, semantic, motion, library,
    )


def _avoid_then_handback_inner(observation, semantic, motion, library):
    """Graded response: slow-walk the plan, or minimal retreat inside margin."""
    scene = build_scene(observation, semantic)
    clearance, nearest = _current_clearance(scene, observation.proprioception)
    chunk = [list(map(float, a)) for a in observation.vla_action_chunk]
    if clearance is None:
        return None
    params = {}
    for primitive in library.recovery_primitives:
        if primitive.get("name") == "motion-margins":
            params = dict(primitive)
    slow_zone = float(params.get("slow_zone_m", SLOW_ZONE_M))
    slow_scale = float(params.get("slow_scale", SLOW_SCALE))
    # Clause A: the response layer compares against the same one-sided floor.
    hard = float(params.get("hard_stop_m", HARD_MARGIN_M)) + FLOOR_EPSILON_M

    if clearance > hard:
        # Plan-preserving slow walk: scale translations of the executable
        # prefix; the collision the judge predicted arrives later and slower,
        # and the next replan re-checks from fresher geometry.
        first_unsafe = motion.first_unsafe_step or 0
        prefix = max(1, min(len(chunk), max(first_unsafe, 2)))
        slowed = []
        for action in chunk[:prefix]:
            out = list(action)
            out[0:3] = [v * slow_scale for v in out[0:3]]
            if len(out) >= 6:
                out[3:6] = [v * slow_scale for v in out[3:6]]
            slowed.append(out)
        # Self-check against the strict policy floor (Packet 2.2): if even
        # the slowed prefix is predicted to touch the floor, a slow walk
        # cannot be authorized — fall through to the emergency retreat
        # instead of letting the recovery re-check degrade to stop.
        query = _swept_query_spheres(observation.proprioception, scene.payload_geometry)
        threats = scene.world["threat_points"]
        if len(threats) and query:
            slow_ok = True
            for offset in _predicted_offsets(slowed):
                worst = min(
                    float(np.min(np.linalg.norm(threats - (c + offset), axis=1))) - r
                    for c, r in query
                )
                if worst <= hard:
                    slow_ok = False
                    break
            if slow_ok:
                return slowed
        else:
            return slowed



    proprio = np.asarray(observation.proprioception, dtype=np.float64)
    ee = proprio[7:10]
    width = len(chunk[0]) if chunk else 7
    gripper = float(chunk[0][-1]) if chunk and width >= 7 else -1.0
    actions = _retreat_actions(clearance, nearest, ee, width, gripper)
    if actions:
        _EMERGENCY_NEXT_CHECK["armed"] = True
    return actions or None


def cap_recovery(observation, semantic, motion, library):
    return _durable(
        "cap_recovery", _cap_recovery_inner, observation, semantic, motion, library
    )


def _cap_recovery_inner(observation, semantic, motion, library):

    params = {}
    for primitive in library.recovery_primitives:
        if primitive.get("name") == "wait-out":
            params = dict(primitive)
    steps = int(params.get("wait_steps", WAIT_STEPS))
    chunk = observation.vla_action_chunk
    width = len(chunk[0]) if chunk else 7
    gripper = float(chunk[0][-1]) if chunk and width >= 7 else -1.0
    hold = [0.0] * width
    hold[-1] = gripper
    prefix = []
    scene = build_scene(observation, semantic)
    clearance, nearest = _current_clearance(scene, observation.proprioception)
    if clearance is not None and nearest is not None and clearance < RETREAT_TARGET_CLEARANCE_M:
        proprio = np.asarray(observation.proprioception, dtype=np.float64)
        prefix = _retreat_actions(clearance, nearest, proprio[7:10], width, gripper)
    _EMERGENCY_NEXT_CHECK["armed"] = True
    return prefix + [list(hold) for _ in range(steps)]
