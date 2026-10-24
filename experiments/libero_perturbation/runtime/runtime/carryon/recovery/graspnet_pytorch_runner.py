"""Subprocess runner for PyTorch Contact-GraspNet.

This script intentionally lives outside CarryOn's import path assumptions: it is
launched by ``ContactGraspNetDetector`` using the contact-graspnet conda env, so
the main CarryOn process never imports torch or the graspnet package.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
from pathlib import Path

def _jsonable_matrix(mat):
    return [[float(v) for v in row] for row in mat]

def _remove_table_plane(pc_full, pc_segments, *, dist_thresh, min_inlier_frac, iters):
    """RANSAC-remove the dominant support plane (table) from ``pc_full`` while
    KEEPING all segment (object) points.

    Contact-GraspNet struggles with thin objects lying flat on a table: the table
    plane dominates the local geometry, so grasps whose contacts land on the object
    collide with the table and get filtered out (filter_grasps=True -> "no grasps").
    Deleting the support plane lets the object stand out as graspable. Returns the
    filtered cloud plus an info dict (``applied`` False if no dominant plane found).
    """
    import numpy as np
    if pc_full is None or len(pc_full) < 100:
        return (pc_full, {'applied': False, 'reason': 'too few points'})
    pts = np.asarray(pc_full, dtype=np.float64)
    n = len(pts)
    m = int(min(n, 20000))
    sample = pts[np.random.choice(n, m, replace=False)]
    best_count, best_plane = (0, None)
    for _ in range(int(iters)):
        tri = sample[np.random.choice(m, 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = float(np.linalg.norm(normal))
        if nn < 1e-09:
            continue
        normal = normal / nn
        d = -float(normal.dot(tri[0]))
        cnt = int((np.abs(sample @ normal + d) < dist_thresh).sum())
        if cnt > best_count:
            best_count, best_plane = (cnt, (normal, d))
    if best_plane is None:
        return (pc_full, {'applied': False, 'reason': 'no plane'})
    normal, d = best_plane
    frac = best_count / m
    if frac < min_inlier_frac:
        return (pc_full, {'applied': False, 'reason': 'plane too small', 'inlier_frac': float(frac)})
    keep = np.abs(pts @ normal + d) >= dist_thresh
    pc_kept = pts[keep]
    seg_pts = [np.asarray(s, dtype=np.float64) for s in pc_segments.values() if len(s)]
    pc_final = np.concatenate([pc_kept, *seg_pts], axis=0) if seg_pts else pc_kept
    info = {'applied': True, 'inlier_frac': float(frac), 'removed': int((~keep).sum()), 'kept': int(len(pc_final)), 'plane_normal': [float(v) for v in normal]}
    return (pc_final.astype(np.asarray(pc_full).dtype), info)

def _camera_pose_looking_at(position, target):
    """Return 4x4 transform (target/original frame <- virtual camera frame)."""
    import numpy as np
    p = np.asarray(position, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    z_c = t - p
    zn = float(np.linalg.norm(z_c))
    if zn < 1e-09:
        z_c = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_c = z_c / zn
    down_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_c = down_world - float(np.dot(down_world, z_c)) * z_c
    yn = float(np.linalg.norm(y_c))
    if yn < 1e-09:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_c = helper - float(np.dot(helper, z_c)) * z_c
        y_c = y_c / (float(np.linalg.norm(y_c)) or 1.0)
    else:
        y_c = y_c / yn
    x_c = np.cross(y_c, z_c)
    xn = float(np.linalg.norm(x_c))
    if xn < 1e-09:
        x_c = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_c = x_c / xn
    y_c = np.cross(z_c, x_c)
    y_c = y_c / (float(np.linalg.norm(y_c)) or 1.0)
    tf_wc = np.eye(4, dtype=np.float64)
    tf_wc[:3, :3] = np.column_stack([x_c, y_c, z_c])
    tf_wc[:3, 3] = p
    return tf_wc

def _sample_random_viewpoint(target_centroid, *, xy_extent_meters):
    import numpy as np
    position = np.random.uniform(-xy_extent_meters, xy_extent_meters, size=(3,))
    return _camera_pose_looking_at(position, target_centroid)

def _sample_cone_viewpoints_evenly(target_centroid, *, num_samples, max_angle_deg, radius_hint=0.5):
    """Sample camera poses on a cone around the current camera axis."""
    import numpy as np
    n = max(1, int(num_samples))
    max_angle_rad = float(max_angle_deg) * math.pi / 180.0
    min_z = math.cos(max_angle_rad)
    phi = (math.sqrt(5.0) + 1.0) / 2.0
    poses = []
    for i in range(n):
        t = 0.0 if n <= 1 else i / float(n - 1)
        z_local = 1.0 - t * (1.0 - min_z)
        theta = 2.0 * math.pi * i / phi
        sin_theta = math.sqrt(max(0.0, 1.0 - z_local * z_local))
        x = sin_theta * math.cos(theta)
        y = sin_theta * math.sin(theta)
        z = z_local
        position = np.asarray([x, y, z], dtype=np.float64) * float(radius_hint)
        poses.append(_camera_pose_looking_at(position, target_centroid))
    return poses

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-dir', required=True)
    parser.add_argument('--ckpt-dir', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--segmap-id', type=int, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--score-threshold', type=float, default=0.4)
    parser.add_argument('--filter-grasps', type=int, choices=(0, 1), default=1)
    parser.add_argument('--local-regions', type=int, choices=(0, 1), default=1)
    parser.add_argument('--remove-table', type=int, choices=(0, 1), default=0, help='RANSAC-remove the dominant support plane (table) from the scene cloud before grasp prediction, keeping object points. Helps thin objects lying flat on a table.')
    parser.add_argument('--table-dist-thresh', type=float, default=0.01, help='distance (m) within which a point counts as on the table plane')
    parser.add_argument('--table-min-inlier-frac', type=float, default=0.3, help='min fraction of sampled points on the plane to treat it as the table')
    parser.add_argument('--table-ransac-iters', type=int, default=200)
    parser.add_argument('--z-min', type=float, default=0.2, help='minimum depth for point cloud extraction (meters)')
    parser.add_argument('--z-max', type=float, default=2.0, help='maximum depth for point cloud extraction (meters)')
    parser.add_argument('--forward-passes', type=int, default=2, help='Contact-GraspNet forward passes per prediction')
    parser.add_argument('--max-retries', type=int, default=1, help='max virtual-viewpoint retries when initial pass finds no grasps')
    parser.add_argument('--viewpoint-mode', choices=('random', 'even'), default='even', help='virtual viewpoint sampling mode used during retries')
    parser.add_argument('--num-viewpoints', type=int, default=40, help='number of evenly sampled viewpoints when --viewpoint-mode=even')
    parser.add_argument('--max-angle-deg', type=float, default=45.0, help='cone max angle for --viewpoint-mode=even')
    parser.add_argument('--viewpoint-xy-extent', type=float, default=0.25, help='random viewpoint sampling extent in original-camera frame (meters)')
    parser.add_argument('--top-k', type=int, default=1, help='number of top-scored grasp candidates to emit (sorted descending). Used by the hybrid + cuRobo goalset planner to try multiple candidates if the argmax-1 grasp is unreachable / in collision. 1 keeps the v0 single-best output.')
    args = parser.parse_args()
    repo = Path(args.repo_dir)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'Pointnet_Pointnet2_pytorch'))
    os.chdir(repo)
    import numpy as np
    import torch
    import types as _types
    if 'contact_graspnet_pytorch.scene_renderer' not in sys.modules:
        _stub = _types.ModuleType('contact_graspnet_pytorch.scene_renderer')

        class _SceneRendererUnavailable:

            def __init__(self, *a, **k):
                raise RuntimeError('SceneRenderer is stubbed out in the CarryOn graspnet runner (headless inference does not need offline scene rendering).')
        _stub.SceneRenderer = _SceneRendererUnavailable
        sys.modules['contact_graspnet_pytorch.scene_renderer'] = _stub
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch.data import load_available_input_data
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        _torch_load = torch.load

        def _cpu_load(*load_args, **load_kwargs):
            load_kwargs.setdefault('map_location', torch.device('cpu'))
            return _torch_load(*load_args, **load_kwargs)
        torch.load = _cpu_load
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = config_utils.load_config(args.ckpt_dir, batch_size=1, arg_configs=[])
            estimator = GraspEstimator(cfg)
            checkpoint_io = CheckpointIO(checkpoint_dir=str(Path(args.ckpt_dir) / 'checkpoints'), model=estimator.model)
            checkpoint_io.load('model.pt')
            estimator.model.eval()
            segmap, rgb, depth, cam_k, pc_full, pc_colors = load_available_input_data(args.input, K=None)
            if pc_full is None:
                pc_full, pc_segments, _ = estimator.extract_point_clouds(depth, cam_k, segmap=segmap, rgb=rgb, skip_border_objects=False, z_range=[float(args.z_min), float(args.z_max)])
            else:
                pc_segments = {}
            if args.segmap_id not in {int(k) for k in pc_segments.keys()}:
                raise RuntimeError(f'segmap id {args.segmap_id} has no depth points')
            table_info = {'applied': False, 'reason': 'disabled'}
            if args.remove_table:
                pc_full, table_info = _remove_table_plane(pc_full, pc_segments, dist_thresh=args.table_dist_thresh, min_inlier_frac=args.table_min_inlier_frac, iters=args.table_ransac_iters)
            with torch.no_grad():
                grasps, scores, contact_pts, openings = estimator.predict_scene_grasps(pc_full, pc_segments=pc_segments, local_regions=bool(args.local_regions), filter_grasps=bool(args.filter_grasps), forward_passes=max(1, int(args.forward_passes)))
            retry_attempts = 0
            key0 = next((k for k in grasps.keys() if int(k) == args.segmap_id), args.segmap_id)
            while len(grasps.get(key0, [])) == 0 and retry_attempts < max(0, int(args.max_retries)):
                if args.segmap_id in pc_segments and len(pc_segments[args.segmap_id]) > 0:
                    target_centroid = np.mean(pc_segments[args.segmap_id], axis=0)
                else:
                    target_centroid = np.mean(pc_full, axis=0)
                if args.viewpoint_mode == 'even':
                    poses = _sample_cone_viewpoints_evenly(target_centroid, num_samples=max(1, int(args.num_viewpoints)), max_angle_deg=float(args.max_angle_deg), radius_hint=max(0.05, float(np.linalg.norm(target_centroid))))
                else:
                    poses = [_sample_random_viewpoint(target_centroid, xy_extent_meters=max(0.01, float(args.viewpoint_xy_extent)))]
                found = False
                for tf_wc in poses:
                    tf_cw = np.linalg.inv(tf_wc)
                    pc_full_h = np.hstack([pc_full, np.ones((pc_full.shape[0], 1), dtype=pc_full.dtype)])
                    pc_segments_h = {seg_id: np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)]) for seg_id, pts in pc_segments.items()}
                    pc_full_cam = (tf_cw @ pc_full_h.T).T[:, :3]
                    pc_segments_cam = {seg_id: (tf_cw @ pts_h.T).T[:, :3] for seg_id, pts_h in pc_segments_h.items()}
                    with torch.no_grad():
                        g_new, s_new, c_new, _ = estimator.predict_scene_grasps(pc_full_cam, pc_segments=pc_segments_cam, local_regions=bool(args.local_regions), filter_grasps=bool(args.filter_grasps), forward_passes=max(1, int(args.forward_passes)))
                    key_new = next((k for k in g_new.keys() if int(k) == args.segmap_id), args.segmap_id)
                    if len(g_new.get(key_new, [])) == 0:
                        continue
                    grasps_cam = np.asarray(g_new[key_new])
                    grasps_world = np.matmul(tf_wc, grasps_cam)
                    pts_cam = np.asarray(c_new[key_new])
                    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=pts_cam.dtype)])
                    pts_world = (tf_wc @ pts_cam_h.T).T[:, :3]
                    grasps[key0] = grasps_world
                    scores[key0] = s_new[key_new]
                    contact_pts[key0] = pts_world
                    found = True
                    break
                retry_attempts += 1
                if found:
                    break
            pc_full_count = int(len(pc_full)) if pc_full is not None else 0
            pc_segment_count = int(len(pc_segments.get(args.segmap_id, [])))
            stats = {'pc_full_count': pc_full_count, 'pc_segment_count': pc_segment_count, 'filter_grasps': bool(args.filter_grasps), 'local_regions': bool(args.local_regions), 'forward_passes': int(max(1, args.forward_passes)), 'z_range': [float(args.z_min), float(args.z_max)], 'remove_table': table_info, 'max_retries': int(max(0, args.max_retries)), 'retry_attempts': int(retry_attempts), 'viewpoint_mode': str(args.viewpoint_mode), 'num_viewpoints': int(max(1, args.num_viewpoints))}
        key = next((k for k in grasps.keys() if int(k) == args.segmap_id), args.segmap_id)
        grasp_arr = np.asarray(grasps.get(key, []))
        score_arr = np.asarray(scores.get(key, [])).reshape(-1)
        if grasp_arr.size == 0 or score_arr.size == 0:
            result = {'ok': False, 'reason': 'no grasps', **stats}
        else:
            order = np.argsort(-score_arr)
            best_idx = int(order[0])
            best_score = float(score_arr[best_idx])
            if best_score < args.score_threshold:
                result = {'ok': False, 'reason': f'score {best_score:.3f} below threshold', 'best_score': best_score, 'count': int(len(grasp_arr)), **stats}
            else:
                k_emit = max(1, int(args.top_k))
                top_idx = order[:k_emit]
                top_grasps = [_jsonable_matrix(grasp_arr[int(i)]) for i in top_idx]
                top_scores = [float(score_arr[int(i)]) for i in top_idx]
                result = {'ok': True, 'score': best_score, 'grasp_cam': _jsonable_matrix(grasp_arr[best_idx]), 'grasps_cam': top_grasps, 'scores': top_scores, 'count': int(len(grasp_arr)), **stats}
    except Exception as exc:
        result = {'ok': False, 'reason': f'{type(exc).__name__}: {exc}'}
    Path(args.output).write_text(json.dumps(result), encoding='utf-8')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
