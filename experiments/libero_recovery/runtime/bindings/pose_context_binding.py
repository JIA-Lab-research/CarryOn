"""Same-perception visual evidence only. No network or semantic ground truth."""
from runtime_paths import bound_path, resource
from pathlib import Path
import hashlib, io, json
import numpy as np
from PIL import Image
from pose_memory import prepare_request

def build_context(*, family_id, object_key, rgb_bytes, selected_mask, reference_path, directory):
    image = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
    pixels = np.asarray(image)
    mask = np.asarray(selected_mask, dtype=bool)
    if mask.shape != pixels.shape[:2] or not mask.any():
        raise ValueError('selected mask must match current frame and be nonempty')
    reference = Path(reference_path).read_bytes()
    Image.open(io.BytesIO(reference)).verify()
    overlay = pixels.copy()
    overlay[mask] = (overlay[mask] * 0.55 + np.array([255, 40, 40]) * 0.45).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(overlay).save(stream, format='PNG')
    images = {'TARGET_REFERENCE': reference, 'CURRENT_RGB': bytes(rgb_bytes), 'SELECTED_MASK_OVERLAY': stream.getvalue()}
    request = prepare_request(object_key, images)
    folder = Path(directory) / 'POSE_INPUT'
    folder.mkdir(exist_ok=False)
    for role, data in images.items():
        (folder / (role + '.bin')).write_bytes(data)
    binding = {'family_id': family_id, 'object_key': object_key, 'request_sha256': request['sha256'], 'image_sha256': request['binding']['image_sha256'], 'selected_mask_sha256_bool_bytes': hashlib.sha256(mask.tobytes()).hexdigest(), 'mask_shape': list(mask.shape), 'source': 'same live selected-mask perception'}
    (folder / 'BINDING.json').write_text(json.dumps(binding, indent=2))
    return {'family_id': family_id, 'object_key': object_key, 'images': images, 'binding': binding}

def write_isolated_candidate(path, *, object_key, corrected_z_m, revision, sources, p_request_sha256, p_response_sha256):
    """Candidate provenance only, not a replacement for production P admission."""
    from pose_memory import make_record, parse_pose, prepare_request
    if not sources or not p_request_sha256 or (not p_response_sha256):
        raise ValueError('bound P response and experiences required')
    poses = set()
    records = []
    for source in sources:
        context = source['context']
        answer = source['pose_answer']
        request = prepare_request(context['object_key'], context['images'])
        if context['object_key'] != object_key or answer['request_sha256'] != request['sha256']:
            raise ValueError('source object/request mismatch')
        pose = parse_pose(answer['text'])['pose']
        if pose == 'UNKNOWN':
            raise ValueError('UNKNOWN cannot label a candidate')
        poses.add(pose)
        records.append({'family_id': context['family_id'], 'pose': pose, 'pose_request_sha256': request['sha256'], 'pose_response_text_sha256': hashlib.sha256(answer['text'].encode()).hexdigest()})
    if len(poses) != 1:
        raise ValueError('different postures cannot share one pose P candidate')
    row = make_record(object_key, {'pose': next(iter(poses))}, corrected_z_m, p_request_sha256, revision)
    row.update(status='ISOLATED_CANDIDATE_NOT_ADMITTED', sources=records, p_response_sha256=p_response_sha256, source_response_sha=p_response_sha256)
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(row, f, indent=2)
    return row
