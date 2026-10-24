"""Classify recorded evidence and bind its images without model calls."""
import base64, copy, hashlib, json, math
from pathlib import Path
from decimal import Decimal

def wire(v):
    return (json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf8')

def digest(b):
    return hashlib.sha256(b).hexdigest()

def sha(p):
    return digest(Path(p).read_bytes())

def read(p):
    return json.loads(Path(p).read_bytes())

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(v if isinstance(v, bytes) else wire(v))
    return sha(p)

def finite(v):
    return type(v) in (int, float) and math.isfinite(v)

def mm(x):
    return float(Decimal(str(x)) * 1000)

def lengths_mm(value):
    if isinstance(value, dict):
        return {k + '_mm' if k in {'w_meas', 'w_cmd'} else k[:-2] + '_mm' if k.endswith('_m') else k: mm(v) if (k.endswith('_m') or k in {'w_meas', 'w_cmd'}) and finite(v) else lengths_mm(v) for k, v in value.items()}
    if isinstance(value, list):
        return [lengths_mm(v) for v in value]
    return value

def walk(v, path=()):
    yield (path, v)
    if isinstance(v, dict):
        for k, x in v.items():
            if k not in {'trajectory', 'judge_timeline'}:
                yield from walk(x, path + (k,))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from walk(x, path + (i,))

def bound_file(item):
    if not isinstance(item, dict) or not Path(item.get('path', '')).is_file() or sha(item['path']) != item.get('sha256'):
        raise ValueError('source file absent/SHA drift')
    return Path(item['path'])

def pointer(v, path):
    for k in path:
        v = v[k]
    return v

def source_value(item):
    return pointer(read(bound_file(item)), item.get('pointer', []))

def mechanically_classify(binding, descent, phases):
    """Consume only existing recorded phase classifier outputs, not new cutoffs."""
    classes = []
    was_nonempty = False
    out = []
    executed = all((finite(descent.get(k)) for k in ('started_wall_ts', 'finished_wall_ts'))) and descent['started_wall_ts'] <= descent['finished_wall_ts']
    for phase in phases:
        op = phase.get('op')
        data = phase.get('data') or {}
        h = data.get('holding_check')
        if op in {'goto_grasp_topdown_family', 'goto_grasp_goalset', 'goto_grasp'} and executed:
            reason = phase.get('reason') or ''
            if phase.get('ok') is False and ('physical_descent_jam_requires_fresh_grounding:' in reason or 'goalset descent failed: descent jammed' in reason):
                classes.append('DESCENT_JAM_ACTUALLY_EXECUTED')
        if not h:
            continue
        if h.get('executed_selection') != binding or h.get('execution_binding_status') != 'MATCH':
            out.append({'status': 'UNKNOWN', 'reason_code': 'HOLD_BINDING_MISSING_OR_MISMATCH'})
            continue
        legal = {k: copy.deepcopy(h[k]) for k in ('status', 'reason_code', 'trace_phase', 'calibration_sha256', 'measured_width_m', 'producer_ts', 'lift_width_trace', 'evaluated_wall_ts', 'state_after', 'execution_binding_status', 'terminal_empty_observation') if k in h}
        out.append(legal)
        status = h.get('status')
        reason = h.get('reason_code')
        if status == 'width_hold_proxy' and reason == 'NONEMPTY_STABLE_WIDTH':
            was_nonempty = True
        elif status == 'empty_closed' and reason == 'MEASURED_EMPTY_CLOSED':
            classes.append('MEASURED_LOSS_AFTER_NONEMPTY' if was_nonempty else 'TERMINAL_MEASURED_EMPTY')
        elif status == 'unstable' and reason in {'LIFT_WIDTH_UNSTABLE', 'TRACE_CROSSES_EMPTY_INTERVAL'}:
            classes.append('MEASURED_WIDTH_UNSTABLE')
        terminal = h.get('terminal_empty_observation') or {}
        if status in {'unstable', 'empty_closed'} and reason in {'LIFT_WIDTH_UNSTABLE', 'TRACE_CROSSES_EMPTY_INTERVAL', 'MEASURED_EMPTY_CLOSED'} and (terminal.get('schema') == 'terminal_width_observation/v1') and (terminal.get('current_and_last_dynamic_empty') is True):
            classes.append('MEASURED_LOSS_AFTER_NONEMPTY' if terminal.get('earlier_dynamic_nonempty') is True or was_nonempty else 'TERMINAL_MEASURED_EMPTY')
    return (sorted(set(classes)), out, executed)

def discover_attempts(episode):
    """Enumerate all distinct captured attempts, including repeated nesting aliases."""
    found = {}
    for path, v in walk(episode.get('recovery', {}), ('recovery',)):
        if isinstance(v, dict) and isinstance(v.get('parameter_attempt_evidence'), dict):
            e = v['parameter_attempt_evidence']
            key = digest(wire({'binding': e.get('binding'), 'descent': e.get('descent', {}).get('started_wall_ts')}))
            found.setdefault(key, {'key': key, 'evidence': e, 'pointer': list(path + ('parameter_attempt_evidence',))})
    return list(found.values())

def derive_images(item, outdir):
    """Original pixels + exact mask bbox8 crop; no replacement segmentation."""
    import numpy as np
    from PIL import Image
    ref = bound_file(item['reference'])
    rgbp = bound_file(item['rgb'])
    maskp = bound_file(item['mask'])
    rgb = np.asarray(Image.open(rgbp).convert('RGB'))
    mask = np.load(maskp, allow_pickle=False)
    if mask.shape != rgb.shape[:2] or not np.any(mask > 0):
        raise ValueError('selected mask/RGB dimensions or empty')
    boolmask = mask > 0
    array_sha = digest(mask.astype(np.int32).tobytes())
    bool_sha = digest(boolmask.tobytes())
    declared = item['selection_binding']
    if declared.get('rgb_sha256') != sha(rgbp) or declared.get('mask_file_sha256') != sha(maskp):
        raise ValueError('same selection RGB/mask binding')
    if declared.get('mask_int32_sha256') != array_sha:
        raise ValueError('mask array binding')
    ys, xs = np.where(boolmask)
    bbox = [max(0, int(xs.min()) - 8), max(0, int(ys.min()) - 8), min(rgb.shape[1], int(xs.max()) + 9), min(rgb.shape[0], int(ys.max()) + 9)]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cp = outdir / 'SELECTED_CROP.png'
    Image.fromarray(rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]).save(cp)
    overlay = rgb.copy()
    overlay[boolmask] = (rgb[boolmask] * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
    op = outdir / 'SELECTED_OVERLAY.png'
    Image.fromarray(overlay).save(op)
    roles = []
    for name, p, phase in [('TARGET_REFERENCE', ref, 'frozen registered reference'), ('CURRENT_SCENE', rgbp, 'same perception'), ('ALGORITHM_SELECTED_MASK_OVERLAY', op, 'derived from same RGB and selected mask'), ('ALGORITHM_SELECTED_CROP_DERIVED', cp, 'same RGB bbox plus 8 pixels, original scale')]:
        roles.append(dict(role=name, phase=phase, local_path=str(p.resolve()), sha256=sha(p), mime=Image.open(p).get_format_mimetype()))
    return (roles, dict(mask_file_sha256=sha(maskp), mask_int32_sha256=array_sha, mask_bool_sha256=bool_sha, rgb_sha256=sha(rgbp), crop_bbox_xyxy=bbox, crop_padding_pixels=8, crop_resized=False))
