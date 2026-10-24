"""Validate object and pose parameters with package-local parsers."""
from pathlib import Path
import hashlib,json,copy,math

def imports():
 from .parsers import parse_p
 from .pose_contract import prepare_request,parse_pose,make_record
 return parse_p,prepare_request,parse_pose,make_record

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def hexsha(v):
    if not isinstance(v, str) or len(v) != 64 or any((c not in '0123456789abcdef' for c in v)):
        raise ValueError('invalid source digest')
    return v

def source_poses(object_key, sources):
    _, prepare_request, parse_pose, _ = imports()
    if not sources:
        raise ValueError('same-perception pose source required')
    records = []
    seen = set()
    for source in sources:
        context = source['context']
        answer = source['pose_answer']
        if context['object_key'] != object_key or not context.get('family_id'):
            raise ValueError('source object/family mismatch')
        if context['family_id'] in seen:
            raise ValueError('duplicate family source')
        seen.add(context['family_id'])
        req = prepare_request(object_key, context['images'])
        if answer['request_sha256'] != req['sha256']:
            raise ValueError('pose answer binding mismatch')
        hexsha(answer['response_sha256'])
        hexsha(answer['wire_sha256'])
        mode = source['classification_timing']
        if mode not in {'LIVE_BEFORE_FAMILY', 'POSTHOC_SAME_PERCEPTION'}:
            raise ValueError('unrecognized classification timing')
        pose = parse_pose(answer['text'])
        records.append({'family_id': context['family_id'], 'object_key': object_key, 'pose': pose['pose'], 'classification_timing': mode, 'pose_request_binding_sha256': req['sha256'], 'pose_wire_sha256': answer['wire_sha256'], 'pose_response_sha256': answer['response_sha256'], 'pose_response_text_sha256': digest(answer['text'].encode()), 'image_sha256': req['binding']['image_sha256']})
    return records

def validate_snapshot(raw):
    _, _, _, make_record = imports()
    v = json.loads(raw)
    if v.get('schema') != 'isolated-pose-snapshot/v1' or v.get('coordinate') != 'world_contact_tcp_z_m':
        raise ValueError('snapshot contract mismatch')
    seen = set()
    for r in v['records']:
        k = (r.get('object_key'), r.get('pose'))
        if k in seen:
            raise ValueError('duplicate active object/pose')
        seen.add(k)
        if r.get('schema') != 'isolated-object-pose-p/v1':
            raise ValueError('legacy object-only P not auto-converted')
        make_record(r['object_key'], {'pose': r['pose']}, r['corrected_z_m'], hexsha(r['source_request_sha']), r['revision'])
        hexsha(r['source_response_sha'])
    return v

def admit_isolated(*, parent_bytes, object_key, pose_sources, raw_p_response, anchor_m, request_sha256, response_sha256, revision):
    """Return immutable proposed snapshot bytes; caller explicitly chooses if/where to save.
 NO_UPDATE returns the exact old bytes and no candidate. Numeric pass is not physical success.
 """
    if not isinstance(object_key, str) or not object_key or object_key != object_key.strip().lower():
        raise ValueError('canonical object key required')
    hexsha(request_sha256)
    hexsha(response_sha256)
    if not isinstance(revision, str) or not revision:
        raise ValueError('revision required')
    parent = validate_snapshot(parent_bytes)
    sources = source_poses(object_key, pose_sources)
    poses = {s['pose'] for s in sources}
    if len(poses) != 1:
        raise ValueError('mixed pose source batch')
    pose = next(iter(poses))
    same = [r for r in parent['records'] if (r['object_key'], r['pose']) == (object_key, pose)]
    if same and anchor_m != same[0]['corrected_z_m']:
        raise ValueError('existing same-pose P must be the current anchor')
    parse_p, _, _, make_record = imports()
    decision = parse_p(raw_p_response, anchor_m)
    if decision['applied'] is not True:
        return {'status': 'NO_UPDATE_RETAINED', 'candidate': None, 'snapshot_bytes': parent_bytes, 'decision': decision}
    if pose == 'UNKNOWN':
        raise ValueError('UNKNOWN cannot create a pose memory record')
    row = make_record(object_key, {'pose': pose}, decision['corrected_z_m'], request_sha256, revision)
    row.update(source_response_sha=response_sha256, rationale=decision['rationale'], status='ISOLATED_CANDIDATE_NOT_PUBLISHED', parent_snapshot_sha256=digest(parent_bytes), prior_record_sha256=digest(json.dumps(same[0], sort_keys=True).encode()) if same else None, sources=sources)
    new = copy.deepcopy(parent)
    new['records'] = [r for r in new['records'] if (r['object_key'], r['pose']) != (object_key, pose)] + [row]
    raw = (json.dumps(new, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    validate_snapshot(raw)
    return {'status': 'ISOLATED_CANDIDATE_NOT_PUBLISHED', 'candidate': row, 'snapshot_bytes': raw, 'decision': decision}

def admit_parameter(batch, raw, parent, response):
    from ..files import bound_file, json_bytes
    sources = copy.deepcopy(batch['pose_sources'])
    for source in sources:
        source['context']['images'] = {role: bound_file(binding).read_bytes()
                                      for role, binding in source['context']['images'].items()}
    result = admit_isolated(parent_bytes=json_bytes(parent), object_key=batch['object_key'],
                            pose_sources=sources, raw_p_response=raw, anchor_m=batch['anchor_z_m'],
                            request_sha256=batch['request']['sha256'],
                            response_sha256=response['response']['sha256'],
                            revision='M' + str(batch['reflection_round']))
    if result['candidate']['pose'] != batch['pose']:
        raise ValueError('Response pose differs from prepared batch')
    return result['candidate']
