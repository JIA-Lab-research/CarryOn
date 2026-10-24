#!/usr/bin/env python3
"""Ground target instances and evaluate their geometric support relation."""
import re

import numpy as np

RELATIONAL_XY_OVERLAP_MIN = 0.5     # Minimum fraction of candidate world-xy points inside the source footprint.
RELATIONAL_ABOVE_MARGIN_M = 0.02    # Allowed candidate median-z offset below the source median-z.


# Receive parsed roles from the planner; this module applies geometry only.
def mask_world(mask, depth_disp, c2w):
    """Unproject a bool mask to world points: world = c2w @ [u*z, v*z, z, 1]
    (matches robosuite transform_from_pixels_to_world). depth_disp and the mask
    share the obs[::-1] display orientation; c2w = inv(get_camera_transform)."""
    a = np.asarray(mask, bool)
    ys, xs = np.where(a)
    if len(xs) == 0:
        return None
    z = np.asarray(depth_disp, float)[ys, xs]
    good = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    if good.sum() == 0:
        return None
    xs, ys, z = xs[good], ys[good], z[good]
    cam = np.stack([xs * z, ys * z, z, np.ones_like(z)], axis=1)
    return (np.asarray(c2w) @ cam.T).T[:, :3]


def xy_overlap_frac(cand_xyz, src_xyz):
    if cand_xyz is None or src_xyz is None or len(cand_xyz) == 0 or len(src_xyz) == 0:
        return 0.0
    lo, hi = src_xyz[:, :2].min(0), src_xyz[:, :2].max(0)
    inside = ((cand_xyz[:, :2] >= lo) & (cand_xyz[:, :2] <= hi)).all(1)
    return float(inside.mean())


def is_on(cand_xyz, src_xyz):
    """On(candidate, source): candidate world-xy footprint overlaps the source
    footprint AND the candidate rests at/above the source surface."""
    if cand_xyz is None or src_xyz is None:
        return False
    overlap = xy_overlap_frac(cand_xyz, src_xyz)
    above = float(np.median(cand_xyz[:, 2])) >= float(np.median(src_xyz[:, 2])) - RELATIONAL_ABOVE_MARGIN_M
    return bool(overlap >= RELATIONAL_XY_OVERLAP_MIN and above)


def relational_target_set(cand_masks, src_mask, depth_disp, c2w, plural):
    """Return target survivor indices and the grounding status.

    With a source, keep candidates satisfying On(candidate, source); a singular
    role requires exactly one survivor. Without a source, keep every non-empty
    candidate of the same noun phrase: zero candidates fail closed, one is unique,
    and multiple candidates form a union. Plural and singular unions retain their
    respective status labels."""
    src_xyz = mask_world(src_mask, depth_disp, c2w) if src_mask is not None else None
    if src_xyz is None:
        nonempty = [i for i, m in enumerate(cand_masks or [])
                    if np.asarray(m, bool).sum() > 0]
        if not nonempty:
            return nonempty, "FAIL_CLOSED_no_candidates"
        if plural:
            return nonempty, "OK_PLURAL_NO_SOURCE"
        if len(nonempty) == 1:
            return nonempty, "OK_SINGULAR_UNIQUE"
        return nonempty, "OK_SINGULAR_UNION_NO_SOURCE"
    survivors = []
    for i, m in enumerate(cand_masks or []):
        if np.asarray(m, bool).sum() == 0:
            continue
        if is_on(mask_world(m, depth_disp, c2w), src_xyz):
            survivors.append(i)
    if plural:
        return survivors, ("OK_PLURAL" if survivors else "FAIL_CLOSED_plural_empty")
    if len(survivors) == 1:
        return survivors, "OK_SINGULAR"
    return survivors, "FAIL_CLOSED_singular_count_%d" % len(survivors)
