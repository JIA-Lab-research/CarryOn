"""Policy-visible camera calibration at the benchmark/sensor boundary."""

from __future__ import annotations

from copy import deepcopy


def calibrated_camera(
    sim,
    base_calibration,
    *,
    camera_name="agentview",
    height=256,
    width=256,
    image_rotation_deg=180,
):
    """Return intrinsics/extrinsics for a camera, never an object or task pose.

    In simulation this is the equivalent of reading a calibrated camera driver
    on a real robot.  The method still receives only ``camera_calibration``.
    """
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    value = deepcopy(dict(base_calibration))
    value["K"] = get_camera_intrinsic_matrix(
        sim, camera_name, height, width
    ).tolist()
    value["world_from_camera"] = get_camera_extrinsic_matrix(
        sim, camera_name
    ).tolist()
    value["near_m"] = float(sim.model.vis.map.znear * sim.model.stat.extent)
    value["far_m"] = float(sim.model.vis.map.zfar * sim.model.stat.extent)
    value["image_rotation_deg"] = int(image_rotation_deg)
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {"previous": provenance}
    provenance["camera"] = (
        "runtime calibrated camera pose/intrinsics; no object or task pose"
    )
    value["provenance"] = provenance
    return value
