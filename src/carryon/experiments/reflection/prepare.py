"""Project newly collected failure evidence into reflection inputs."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

from ..files import bound_file, json_bytes, read_json, sha256, write_json
from .evidence import walk
from .memory import load
from .pose import source_poses
from .projections import object_m1, object_m2, recorded
from .requests import save_requests


def poses_for_batch(batch, roots):
    poses, audit, sources = {}, [], {}
    for episode in batch['episodes']:
        rid = episode['episode_id']
        root = Path(roots[rid])
        families = list((root / 'common_perception_raw').glob('*/FAMILY.json'))
        if not families:
            poses[rid] = 'UNAVAILABLE_NO_SELECTION'
            continue
        if len(families) != 1:
            raise ValueError('Pose family binding is not unique: ' + rid)
        family_file = families[0]
        binding = read_json(family_file)['common_family']['pose_binding']
        call = root / 'pose_calls' / Path(binding['archive_path']).name
        receipt = read_json(call / 'RESULT.json')
        if sha256(call / 'request.body.bin') != binding['wire_sha256'] or binding['wire_sha256'] != receipt['wire_sha256']:
            raise ValueError('Pose request binding mismatch')
        if sha256(call / 'response.body.bin') != binding['response_sha256'] or binding['response_sha256'] != receipt['response_sha256']:
            raise ValueError('Pose response binding mismatch')
        if binding['family_id'] != family_file.parent.name or binding['object_key'] != batch['object_key']:
            raise ValueError('Pose object/family binding mismatch')
        images = {role: {'path': str(family_file.parent / 'POSE_INPUT' / (role + '.bin')), 'sha256': digest}
                  for role, digest in binding['image_sha256'].items()}
        for image in images.values():
            bound_file(image)
        text = (call / 'MODEL_TEXT.txt').read_text(encoding='utf-8').strip()
        if text.startswith('```json\n') and text.endswith('```'):
            text = text[8:-3].strip()
        elif text.startswith('```\n') and text.endswith('```'):
            text = text[4:-3].strip()
        if json.loads(text) != {'pose': binding['pose'], 'visual_basis': binding['visual_basis']}:
            raise ValueError('Archived pose text differs from the executed family pose')
        source = {'context': {'object_key': batch['object_key'], 'family_id': binding['family_id'], 'images': images},
                  'classification_timing': 'LIVE_BEFORE_FAMILY',
                  'pose_answer': {'request_sha256': binding['request_sha256'], 'wire_sha256': binding['wire_sha256'],
                                  'response_sha256': binding['response_sha256'], 'text': text}}
        check = copy.deepcopy(source)
        check['context']['images'] = {role: bound_file(image).read_bytes() for role, image in images.items()}
        source_poses(batch['object_key'], [check])
        poses[rid] = binding['pose']
        audit.append(dict(episode_id=rid, **binding, sha_verified=True))
        sources[rid] = source
    return poses, audit, sources


def unselected_context(batch, roots, reference):
    """Use retained source observations, never create replacement evidence."""
    contexts, roles, missing = [], [], []
    for episode in batch['episodes']:
        if episode['attempt_ids']:
            continue
        if 'all_recovery_episode_files' in episode and not episode['all_recovery_episode_files']:
            continue
        rid = episode['episode_id']
        root = Path(roots[rid])
        files = list((root / 'visual_object_memory').glob('*/ALL_CANDIDATE_COMPARATIVE_EVIDENCE.json'))
        if len(files) == 1:
            evidence = read_json(files[0])
            if evidence['selection_status'] != 'ALL_CANDIDATE_COMPARATIVE_REJECTED' or evidence['selected_candidate_id'] is not None or evidence['parsed']['matches_reference'] is not False:
                raise ValueError('Unselected episode is not a recorded comparative rejection')
            image = files[0].parent / 'COMPARATIVE_UNANNOTATED_CURRENT.png'
            if sha256(image) != evidence['COMPARATIVE_UNANNOTATED_CURRENT.png']['sha256'] or reference['sha256'] != evidence['reference_sha256']:
                raise ValueError('Unselected scene/reference binding mismatch')
            contexts.append({'episode_id': rid, 'grounding_mode': 'SELECTOR_REJECTED_BEFORE_EXECUTED_SELECTION',
                'selection_status': evidence['selection_status'], 'selected_mask': 'NOT_APPLICABLE_NO_ACCEPTED_SELECTION',
                'parameter_attempts': 0, 'online_pose': 'UNAVAILABLE_STAGE_NOT_REACHED',
                'scene_is_exact_comparison_input': True, 'evidence_sha256': sha256(files[0])})
            for role, path in [('TARGET_REFERENCE', bound_file(reference)), ('CURRENT_SCENE', image)]:
                roles.append({'local_path': str(path), 'sha256': sha256(path), 'mime': 'image/png',
                    'role': role, 'row_id': rid, 'phase': 'frozen registered reference' if role == 'TARGET_REFERENCE'
                    else 'same recovery comparison input; no accepted selected mask'})
            continue
        context_file = root / 'NO_POINT_CONTEXT.json'
        if not context_file.is_file():
            missing.append(rid + ':NO_RETAINED_NO_POINT_CONTEXT')
            continue
        retained = read_json(context_file)
        episodes = [read_json(p) for p in (root / 'recovery_episodes').glob('*.json')]
        errors = [node['reason'] for ep in episodes for _, node in walk(ep.get('recovery', {}))
                  if isinstance(node, dict) and node.get('op') == 'perceive_grasps' and node.get('ok') is False]
        if len(errors) != 1 or 'REFERENCE_CONDITIONED_POINTER_FAILURE:' not in errors[0]:
            raise ValueError('No-point context lacks its matching recorded grounding failure')
        context = dict(retained['facts'])
        if context['episode_id'] != rid or context['runtime_exception'] != errors[0]:
            raise ValueError('No-point exception binding mismatch')
        contexts.append(context)
        for image in retained['roles']:
            path = bound_file({'path': image['local_path'], 'sha256': image['sha256']}, root)
            roles.append({**image, 'local_path': str(path)})
    return contexts, roles, missing


def prepare(collection, parent, model_config, output):
    collection = Path(collection).resolve()
    previous = load(parent)
    batches = read_json(collection / 'BATCHES.json')
    roots = read_json(collection / 'SOURCE_ROOTS.json')
    reference_registry = read_json(collection / 'REFERENCES.json')
    specs, skipped = [], []
    generation = previous['manifest']['generation'] + 1
    suite = previous['manifest']['suite']
    old_s = {r['object_key']: r for r in previous['S']}
    old_pose = {(r['object_key'], r['pose']): r for r in previous['pose']['records']}
    registry_digest = hashlib.sha256(json_bytes({k: r['sha256'] for k, r in reference_registry.items()})).hexdigest()
    for key, batch in batches.items():
        if batch['parent_snapshot_sha256'] != previous['snapshot_sha256'] or batch['reflection_round'] != generation:
            raise ValueError('Collection parent/generation mismatch')
        references = [r for r in batch['roles'] if r['role'] == 'TARGET_REFERENCE']
        if references:
            if len({r['sha256'] for r in references}) != 1:
                raise ValueError('One object has conflicting reference pixels')
            reference = {'path': references[0]['local_path'], 'sha256': references[0]['sha256']}
        elif key in old_s:
            reference = {'path': str(Path(parent) / old_s[key]['reference_path']), 'sha256': old_s[key]['reference_sha256']}
        elif key in reference_registry:
            reference = reference_registry[key]
        else:
            skipped.append({'object_key': key, 'role': 'S/P', 'reason': 'NO_BOUND_TARGET_REFERENCE'})
            continue
        bound_file(reference)
        contexts, extra_roles, no_context = unselected_context(batch, roots, reference)
        common = {'object_key': key, 'reflection_round': generation,
                  'parent_snapshot_sha256': previous['snapshot_sha256'],
                  'reference': reference, 'registry_sha256': registry_digest,
                  'mechanical_source_classes': sorted({c for a in batch['attempts'] for c in a['mechanical_source_classes']})}
        if suite == 'libero_perturbation':
            specialised = generation == 2 and batch['episodes'] and 'R1_global_ordinal' in batch['episodes'][0]
            for role in ('S', 'P'):
                if specialised:
                    facts, roles = object_m2(batch, role)
                    profile = 'object_m2'
                else:
                    actual = [a for a in batch['attempts'] if a['executed'] and a['planned_contact_TCP_mm']]
                    excluded = any(not ep['attempt_ids'] for ep in batch['episodes'])
                    revised = excluded or (role == 'P' and actual and not any(a['mechanical_source_classes'] for a in actual))
                    problems = [x for x in batch.get(role + '_missing', []) if not x.endswith(
                        (':NO_RECOVERY_EPISODE', ':NO_BOUND_ATTEMPT_IMAGES', ':NO_ACTUAL_PARAMETER_EVIDENCE'))]
                    if problems or (role == 'P' and not actual) or (role == 'S' and no_context):
                        skipped.append({'object_key': key, 'role': role, 'reason': 'NO_QUALIFYING_BOUND_EVIDENCE',
                                        'missing': problems + (no_context if role == 'S' else [])})
                        continue
                    facts, roles = object_m1(batch, role, revised=revised, unselected=contexts, extra_roles=extra_roles)
                    if not facts['episodes']:
                        skipped.append({'object_key': key, 'role': role, 'reason': 'NO_ELIGIBLE_FAILED_EPISODE'})
                        continue
                    profile = 'object_m1_v3' if revised else 'object_m1_v2'
                spec = {**common, 'role': role, 'facts': facts, 'roles': roles, 'prompt_profile': profile}
                if role == 'P':
                    spec['anchor_z_m'] = facts['memory_context']['anchor_z_mm'] / 1000
                specs.append(spec)
        else:
            poses, pose_audit, pose_sources = poses_for_batch(batch, roots)
            combinations = [('S', None, batch['attempts'], batch['episodes'], batch['roles'] + extra_roles)]
            for pose in ('UPRIGHT', 'LYING'):
                attempts = [a for a in batch['attempts'] if poses[a['episode_id']] == pose]
                if not any(a['executed'] and a['planned_contact_TCP_mm'] for a in attempts):
                    continue
                ids = {a['episode_id'] for a in attempts}
                aids = {a['attempt_id'] for a in attempts}
                combinations.append(('P', pose, attempts, [ep for ep in batch['episodes'] if ep['episode_id'] in ids],
                                     [r for r in batch['roles'] if r['row_id'] in aids]))
            for role, pose, attempts, episodes, roles in combinations:
                if role == 'S' and no_context:
                    skipped.append({'object_key': key, 'role': role, 'reason': 'NO_RETAINED_NO_POINT_CONTEXT', 'missing': no_context})
                    continue
                if role == 'P' and any(a['missing'] for a in attempts):
                    skipped.append({'object_key': key, 'role': role, 'pose': pose, 'reason': 'INCOMPLETE_BOUND_ATTEMPT_EVIDENCE'})
                    continue
                facts, selected_roles = recorded(batch, role, pose, attempts, episodes, roles,
                    pose_audit=pose_audit, unselected=contexts, prior_pose=old_pose.get((key, pose)),
                    parent_pose_sha256=sha256(Path(parent) / 'POSE_MEMORY.json'))
                spec = {**common, 'role': role, 'pose': pose, 'facts': facts, 'roles': selected_roles,
                        'prompt_profile': 'recovery_m' + str(generation)}
                if role == 'P':
                    spec['anchor_z_m'] = facts['memory_context']['anchor_z_mm'] / 1000
                    spec['pose_sources'] = [pose_sources[ep['episode_id']] for ep in episodes]
                specs.append(spec)
            for rid, pose in poses.items():
                if pose not in ('UPRIGHT', 'LYING'):
                    skipped.append({'object_key': key, 'role': 'P', 'episode_id': rid, 'pose': pose,
                                    'reason': 'UNKNOWN_OR_UNSELECTED_POSE_NOT_BORROWED'})
    bindings = save_requests(specs, parent, model_config, output)
    write_json(Path(output) / 'NO_CALLS.json', skipped)
    return bindings
