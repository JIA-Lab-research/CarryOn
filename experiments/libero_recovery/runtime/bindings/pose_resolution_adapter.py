"""Isolated synchronous adapter. The perception caller supplies live bound images.
No network code, inference of posture from geometry, PI labels or object exceptions."""
from runtime_paths import bound_path, resource
import copy
import math
from pose_memory import prepare_request, parse_pose, make_record

def make_resolver(classify_once, load_pose_records):

    def resolve(*, object_key, family_id, seed_z_m, context):
        if not isinstance(context, dict) or context.get('family_id') != family_id or context.get('object_key') != object_key:
            raise ValueError('perception family/object binding mismatch')
        request = prepare_request(object_key, context['images'])
        request['call_id'] = str(family_id)
        answer = classify_once(request)
        if answer['request_sha256'] != request['sha256']:
            raise ValueError('response not bound to request')
        pose = parse_pose(answer['text'])
        rows = load_pose_records()
        hits = [r for r in rows if r.get('schema') == 'isolated-object-pose-p/v1' and r.get('object_key') == object_key and (r.get('pose') == pose['pose']) and (pose['pose'] != 'UNKNOWN')]
        if len(hits) > 1:
            raise ValueError('ambiguous active memory revision')
        row = copy.deepcopy(hits[0]) if hits else None
        if row:
            make_record(object_key, pose, row['corrected_z_m'], row['source_request_sha'], row['revision'])
        z = float(row['corrected_z_m'] if row else seed_z_m)
        if not math.isfinite(z) or (row and abs(z - float(seed_z_m)) > 0.2):
            raise ValueError('invalid effective z')
        return {'row': row, 'effective_z_m': z, 'binding': {'family_id': family_id, 'object_key': object_key, 'pose': pose['pose'], 'visual_basis': pose['visual_basis'], 'request_sha256': request['sha256'], 'image_sha256': request['binding']['image_sha256'], 'wire_sha256': answer.get('wire_sha256'), 'response_sha256': answer.get('response_sha256'), 'archive_path': answer.get('archive_path')}}
    return resolve
