"""Compute same-frame destination-surface tolerance."""
import numpy as np
from scipy.spatial import cKDTree


def complete_boundary(original_mask, depth, world_to_pixel):
    original = np.asarray(original_mask, dtype=bool)
    depth = np.asarray(depth, dtype=float)
    if original.ndim != 2 or depth.shape != original.shape:
        raise ValueError("Mask and depth must have the same two-dimensional shape")
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    v, u = np.indices(depth.shape)
    safe_depth = np.where(valid, depth, 0.0)
    projected = np.stack((u * safe_depth, v * safe_depth, safe_depth,
                          np.ones_like(depth)), axis=-1)
    xyz = (projected @ np.linalg.inv(np.asarray(world_to_pixel, float)).T)[..., :3]
    anchors = original & valid
    claimed = np.zeros_like(original)
    if anchors.any():
        # Query measured points once. Newly claimed points never seed the tree.
        distances = cKDTree(xyz[anchors]).query(xyz[valid], k=1, eps=0)[0]
        claimed[valid] = distances <= 0.030
    return claimed, claimed & ~original
