"""Represent pose-classification requests and pose-indexed parameter memory."""
from runtime_paths import bound_path, resource
import hashlib
import json
import math
import re
POSES = {'UPRIGHT', 'LYING', 'UNKNOWN'}
PROMPT = 'Describe the selected target object\'s current posture using the reference\nimage, current scene and selected-mask overlay. UPRIGHT means standing in its\ncanonical upright orientation; LYING includes resting on a side or lying flat.\nThese are object postures, not gripper opening directions. Use UNKNOWN if the\nselected target or posture cannot be established visually. Do not infer posture\nfrom object name, scene ID, prior episodes or grasp success. Return JSON only:\n{"pose":"UPRIGHT|LYING|UNKNOWN","visual_basis":"brief visible evidence"}.\nDo not propose a grasp or height.'
SCHEMA = 'isolated-object-pose-p/v1'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def prepare_request(object_key, images):
    required = {'TARGET_REFERENCE', 'CURRENT_RGB', 'SELECTED_MASK_OVERLAY'}
    if not isinstance(object_key, str) or not object_key.strip():
        raise ValueError('object key required')
    if set(images) != required or any((not isinstance(v, bytes) or not v for v in images.values())):
        raise ValueError('three bound image roles required')
    binding = {'object_key': object_key, 'image_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in images.items()}, 'prompt': PROMPT}
    return {'binding': binding, 'sha256': digest(binding), 'images': dict(images)}

def parse_pose(raw):
    if isinstance(raw, str):
        text = raw.strip()
        fenced = re.fullmatch('```(?:json)?[ \\t]*\\r?\\n([\\s\\S]*?)\\r?\\n```', text, flags=re.IGNORECASE)
        if fenced is not None:
            raw = fenced.group(1)
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {'pose', 'visual_basis'}:
        raise ValueError('invalid pose structure')
    if value['pose'] not in POSES or not isinstance(value['visual_basis'], str) or (not value['visual_basis'].strip()):
        raise ValueError('pose and visual evidence required')
    return value

def make_record(object_key, pose_result, corrected_z_m, source_request_sha, revision):
    pose = pose_result['pose']
    if pose not in POSES - {'UNKNOWN'}:
        raise ValueError('UNKNOWN cannot be a memory key')
    if isinstance(corrected_z_m, bool) or not isinstance(corrected_z_m, (int, float)):
        raise ValueError('z must be numeric')
    if not math.isfinite(corrected_z_m) or not 0 <= corrected_z_m <= 0.25:
        raise ValueError('outside existing absolute-z range')
    if not object_key or not source_request_sha or (not revision):
        raise ValueError('source and revision required')
    return dict(schema=SCHEMA, object_key=object_key, pose=pose, corrected_z_m=corrected_z_m, source_request_sha=source_request_sha, revision=revision)

class FamilyResolver:
    """Injected perception caller used during preparation, once per perception family."""

    def __init__(self):
        self.families = {}

    def resolve(self, family_id, request, seed_z_m, records, caller):
        if not family_id or not math.isfinite(seed_z_m):
            raise ValueError('invalid family')
        signature = digest(dict(request_sha=request['sha256'], seed_z_m=seed_z_m, records=records))
        if family_id in self.families:
            old_signature, result = self.families[family_id]
            if old_signature != signature:
                raise ValueError('family inputs changed')
            return dict(result)
        pose = parse_pose(caller(request))
        key = request['binding']['object_key']
        hits = [r for r in records if r.get('schema') == SCHEMA and r.get('object_key') == key and (r.get('pose') == pose['pose']) and (pose['pose'] != 'UNKNOWN')]
        if len(hits) > 1:
            raise ValueError('resolve active revision before lookup')
        if hits:
            r = hits[0]
            make_record(key, pose, r['corrected_z_m'], r['source_request_sha'], r['revision'])
            z = r['corrected_z_m']
            if abs(z - seed_z_m) > 0.2:
                raise ValueError('outside existing maximum adjustment')
        else:
            z = seed_z_m
        result = dict(pose=pose['pose'], visual_basis=pose['visual_basis'], effective_z_m=z, P_applied=bool(hits), request_sha=request['sha256'], family_id=family_id)
        self.families[family_id] = (signature, dict(result))
        return result
