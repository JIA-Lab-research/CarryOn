"""Isolated selected-cloud preparation; no GG/P calls or geometry centre rule.

Runtime integration must use either the existing call's result with
prepare_from_selected_cloud, or prepare_detector_input as its replacement.
Never call both acquisition paths for one selected mask. This file does not
install anything.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import uuid
import numpy as np
SCHEMA = 'common-perception-input/v3'
MIN_POINTS = 50
Z_TOP_PCT = 98.0
TABLE_CLEAR_M = 0.005
PAD_HEIGHT_M = 0.016
MID_DROP_M = 0.008

class PreparationError(ValueError):
    """No usable preparation; caller must not fall back to GG or stale input."""

@dataclass(frozen=True)
class PreparedGraspInput:
    cloud: tuple
    seed_z_m: float
    evidence: dict

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def _snapshot(value):
    """JSON metadata snapshot. Arrays retain hash/shape/dtype, not huge pixels."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {'type': 'ndarray', 'status': 'OBJECT_ARRAY_NOT_SERIALIZED'}
        return {'type': 'ndarray', 'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256_c_order_bytes': _sha(np.ascontiguousarray(value).tobytes())}
    if isinstance(value, np.generic):
        return _snapshot(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'nonfinite': str(value)}
    if isinstance(value, (tuple, list)):
        return [_snapshot(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _snapshot(v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return {'type': 'bytes', 'length': len(value), 'sha256': _sha(value)}
    return {'type': type(value).__name__, 'status': 'NOT_SERIALIZED'}

def native_seed_z(point_cloud, table_z_world):
    """Derive a deterministic sampling seed for the candidate."""
    try:
        pc = np.asarray(point_cloud, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PreparationError('INVALID_CLOUD') from exc
    if pc.ndim != 2 or pc.shape[1] != 3 or len(pc) < MIN_POINTS:
        raise PreparationError('INVALID_CLOUD_SHAPE_OR_TOO_SMALL')
    if not np.all(np.isfinite(pc)):
        raise PreparationError('NONFINITE_CLOUD')
    try:
        table_z = 0.0 if table_z_world is None else float(table_z_world)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PreparationError('INVALID_TABLE_Z') from exc
    if not math.isfinite(table_z):
        raise PreparationError('INVALID_TABLE_Z')
    z_top = float(np.percentile(pc[:, 2], Z_TOP_PCT))
    z_mid = (z_top + table_z) / 2.0
    seed = max(table_z + TABLE_CLEAR_M, min(z_mid - MID_DROP_M, z_top - PAD_HEIGHT_M))
    if not all((math.isfinite(v) for v in (z_top, z_mid, seed))):
        raise PreparationError('UNREPRESENTABLE_SEED_Z')
    ev = {'source': 'existing native_topdown default formula; geometric seed, not a VLM learning result', 'table_z_m': table_z, 'table_z_source': 'config.table_z_world' if table_z_world is not None else 'native existing fallback 0.0 when config is unset', 'z_top_m': z_top, 'z_mid_m': z_mid, 'seed_z_m': seed, 'z_rule_branch': 'table_floor' if seed == table_z + TABLE_CLEAR_M else 'mid_drop' if z_mid - MID_DROP_M <= z_top - PAD_HEIGHT_M else 'below_top', 'constants': {'MIN_POINTS': MIN_POINTS, 'Z_TOP_PCT': Z_TOP_PCT, 'TABLE_CLEAR_M': TABLE_CLEAR_M, 'PAD_HEIGHT_M': PAD_HEIGHT_M, 'MID_DROP_M': MID_DROP_M}}
    return (seed, ev)

def prepare_from_selected_cloud(point_cloud, *, segmap_id, table_z_world, selection_binding, camera_source, archive_root):
    """Package the already selected call result, never select a mask or call P.

    archive_root must be the caller's row-local output directory.
    selection_binding/camera_source describe real known sources; missing camera
    identity is recorded as unknown, never inferred from a view or object.
    """
    if not isinstance(selection_binding, dict) or not isinstance(camera_source, dict):
        raise PreparationError('BINDING_METADATA_MUST_BE_DICT')
    if isinstance(segmap_id, (bool, np.bool_)) or not isinstance(segmap_id, (int, np.integer)):
        raise PreparationError('INVALID_SELECTION_ID')
    try:
        raw = np.array(point_cloud, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise PreparationError('INVALID_CLOUD') from exc
    if raw.dtype.hasobject:
        raise PreparationError('OBJECT_CLOUD_NOT_ARCHIVABLE')
    seed, seed_ev = native_seed_z(raw, table_z_world)
    frozen_cloud = tuple((tuple((float(v) for v in row)) for row in raw))
    prepare_id = uuid.uuid4().hex
    ev = {'schema': SCHEMA, 'prepare_id': prepare_id, 'segmap_id': int(segmap_id), 'seed': seed_ev, 'selection_binding': _snapshot(selection_binding), 'camera_source': _snapshot(camera_source), 'cloud_shape': list(raw.shape), 'cloud_dtype': str(raw.dtype), 'cloud_sha256_original_c_order_bytes': _sha(np.ascontiguousarray(raw).tobytes()), 'cloud_float32_sha256': _sha(np.ascontiguousarray(raw.astype(np.float32)).tobytes()), 'mask_binding': 'segmap_id and point_cloud returned by the same _object_point_cloud_world invocation', 'mask_pixels_status': 'not separately returned by this native exit; any available detector debug retained in selection_binding', 'center_rule': 'not evaluated here; only common_grasp_geometry_v2.build_candidates may construct centre', 'p_read': False, 'graspgen_server_called': False, 'candidate_generation_called': False, 'runtime_installed_by_this_module': False}
    directory = Path(archive_root) / 'common_perception_raw' / prepare_id
    directory.mkdir(parents=True, exist_ok=False)
    cloud_file = directory / 'POINT_CLOUD.npy'
    with cloud_file.open('xb') as handle:
        np.save(handle, raw, allow_pickle=False)
    ev['archive'] = {'dir': str(directory), 'cloud_path': str(cloud_file), 'point_cloud_file_sha256': _sha(cloud_file.read_bytes()), 'prepare_input_path': str(directory / 'PREPARE_INPUT.json')}
    with (directory / 'PREPARE_INPUT.json').open('x', encoding='utf-8') as handle:
        json.dump(ev, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    return PreparedGraspInput(cloud=frozen_cloud, seed_z_m=seed, evidence=ev)

def prepare_detector_input(detector, *, archive_root, rgb, depth_m, intrinsics, object_hint, target_xy=None, target_px=None, grounding=None, sam_text_prompts=None, scene_reference=None, exclude_masks=None, camera_source=None):
    """Replacement for native detect_many acquisition, with exactly one call.

    Uses the configured mask and depth providers.
    """
    detector.last_error = None
    detector.last_server_result = None
    detector.last_debug = {}
    if grounding is not None:
        if getattr(grounding, 'normalized_point', None) is not None:
            target_px = tuple(grounding.normalized_point)
        if getattr(grounding, 'world_point', None) is not None and len(grounding.world_point) >= 2:
            target_xy = tuple(grounding.world_point[:2])
    output = detector._object_point_cloud_world(rgb=rgb, depth_m=depth_m, intrinsics=intrinsics, object_hint=object_hint, target_xy=target_xy, target_px=target_px, sam_text_prompts=sam_text_prompts, scene_reference=scene_reference, exclude_masks=exclude_masks)
    if output is None:
        raise PreparationError('NATIVE_TOPDOWN_NO_CLOUD: ' + str(detector.last_error))
    point_cloud, segmap_id = output
    source = dict(camera_source or {})
    source.setdefault('identity_status', 'CALLER_PROVIDED' if camera_source else 'NOT_EXPOSED_BY_NATIVE_DETECT_MANY')
    source['rgb_input'] = _snapshot(rgb)
    source['depth_input'] = _snapshot(depth_m)
    source['intrinsics_input'] = _snapshot(intrinsics)
    binding = {'object_hint': object_hint, 'effective_target_xy': target_xy, 'effective_target_px': target_px, 'sam_text_prompts': sam_text_prompts, 'scene_reference': scene_reference, 'exclude_masks': exclude_masks, 'detector_debug_from_same_call': detector.last_debug, 'cloud_call_count_in_adapter': 1}
    prepared = prepare_from_selected_cloud(point_cloud, segmap_id=segmap_id, table_z_world=getattr(detector.config, 'table_z_world', None), selection_binding=binding, camera_source=source, archive_root=archive_root)
    detector.last_debug['common_perception_prepare'] = prepared.evidence
    return prepared
