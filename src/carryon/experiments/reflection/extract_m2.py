"""Project second-round Perturbation evidence."""
from pathlib import Path
import hashlib,math
import numpy as np
from .evidence import discover_attempts,walk,mechanically_classify,lengths_mm,derive_images,sha,mm,read
def valid_xyz(value):
    return isinstance(value, list) and len(value) == 3 and all((type(v) in (int, float) and math.isfinite(v) for v in value))

def extract_object_m2(sample,source_roots,references,output,param_state,semantic_state):
    OUT=Path(output)
    image_index=dict(references)
    for root in source_roots.values():
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bin'):
                image_index.setdefault(sha(p),p)
    eligible=sorted({r['object_key'] for r in sample})
    coordinate={'frame':'planner world','reference':'contact_TCP','positive_z':'up','length_unit':'mm','parser_bounds_mm':[0,250],'maximum_delta_from_anchor_mm':200}
    batches,binding_audit,source_bindings={ },[],[]
    for object_key in eligible:
        pstate = param_state[object_key]
        sstate = semantic_state[object_key]
        batch = {'object_key': object_key, 'reflection_round': 2, 'source_round': 'R1', 'memory_context': {'P_current_z_mm': mm(pstate['corrected_z_m']), 'anchor_z_mm': mm(pstate['corrected_z_m']), 'anchor_source': 'frozen M1 active parameter for this object', 'S_entry_present': True, 'active_template': sstate['template']}, 'coordinate': coordinate, 'episodes': [], 'attempts': [], 'roles': []}
        if not sstate['template'] == 'REGROUND_V7_ALL_CANDIDATE_V1':
            raise ValueError('Evidence binding mismatch')
        if not type(pstate['corrected_z_m']) in (int, float):
            raise ValueError('Evidence binding mismatch')
        for selected in [x for x in sample if x['object_key'] == object_key]:
            row_dir = source_roots[selected['row_id']]
            result = read(row_dir / 'RESULT.json')
            if not result['success'] is False:
                raise ValueError('Evidence binding mismatch')
            episode_files = sorted((row_dir / 'recovery_episodes').glob('*.json'))
            if not len(episode_files) == 1:
                raise ValueError('Evidence binding mismatch')
            ep = read(episode_files[0])
            attempts = discover_attempts(ep)
            episode_fact = {'episode_id': selected['row_id'], 'R1_global_ordinal': selected['global_ordinal'], 'official_task_failure': True, 'terminal_feedback_scope': 'whole task only; not identity/grasp/z diagnosis', 'recovery_triggered': True, 'recovery_failure_type': ep.get('recovery', {}).get('failure_type', 'MISSING'), 'recovery_outcome': ep.get('recovery', {}).get('outcome', 'MISSING'), 'recovery_notes': ep.get('recovery', {}).get('notes', 'MISSING'), 'attempt_ids': [], 'selected_evidence_status': 'PRESENT' if attempts else 'MISSING: recovery ended before a bound selected parameter attempt', 'parameter_attempt_status': 'PRESENT' if attempts else 'MISSING: no bound executed parameter attempt'}
            grounding = ep.get('recovery', {}).get('grounding') or {}
            episode_fact['grounding'] = {'mode': grounding.get('mode', 'MISSING'), 'provenance': grounding.get('provenance', 'MISSING'), 'confidence': grounding.get('confidence', 'MISSING'), 'normalized_point': grounding.get('normalized_point', 'MISSING'), 'region': grounding.get('region', 'MISSING'), 'world_point': grounding.get('world_point', 'MISSING'), 'attempt_count': len(grounding.get('attempts') or [])}
            batch['episodes'].append(episode_fact)
            source_bindings.append({'object_key': object_key, 'episode_id': selected['row_id'], 'R1_global_ordinal': selected['global_ordinal'], 'RESULT_sha256': sha(row_dir / 'RESULT.json'), 'recovery_episode_sha256': sha(episode_files[0]), 'attempt_count': len(attempts)})
            imgbysha = {}
            for p in row_dir.rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bin'):
                    imgbysha.setdefault(sha(p), p)
            pvis = [(p, read(p)) for p in (row_dir / 'parameter_visual').glob('*/INPUT_EVIDENCE.json')]
            stages = [(p, read(p)) for p in (row_dir / 'attempt_evidence').glob('*/*.json')]
            families = [(p, read(p)) for p in (row_dir / 'common_perception_raw').glob('*/FAMILY.json')]
            if not attempts:
                ref = image_index.get(sstate['reference_sha256'])
                scene_candidates = list((row_dir / 'visual_object_memory').glob('*/COMPARATIVE_UNANNOTATED_CURRENT.png'))
                if not (ref is not None and len(scene_candidates) == 1):
                    raise ValueError('Evidence binding mismatch')
                scene = scene_candidates[0]
                roles = [{'role': 'TARGET_REFERENCE', 'phase': 'frozen registered reference', 'local_path': str(ref), 'sha256': sha(ref), 'mime': 'image/png', 'row_id': selected['row_id']}, {'role': 'CURRENT_SCENE', 'phase': 'same recovery scene retained before selection failed', 'local_path': str(scene), 'sha256': sha(scene), 'mime': 'image/png', 'row_id': selected['row_id']}]
                batch['roles'].extend(roles)
                continue
            for attempt in attempts:
                evidence = attempt['evidence']
                bind = evidence['binding']
                attempt_id = selected['row_id'] + '/' + attempt['key']
                episode_fact['attempt_ids'].append(attempt_id)
                v_matches = [(p, v) for p, v in pvis if v.get('binding') == bind and v.get('descent', {}).get('started_wall_ts') == evidence.get('descent', {}).get('started_wall_ts')]
                stage_matches = [(p, v) for p, v in stages if v.get('executed_selection') == bind]
                family_ids = {v.get('common_family_id') for _, v in stage_matches if v.get('common_family_id')}
                family_matches = [(p, v) for p, v in families if p.parent.name in family_ids]
                missing = []
                roles = []
                if len(v_matches) != 1:
                    missing.append('MISSING: parameter visual binding not unique')
                if len(family_matches) != 1:
                    missing.append('MISSING: selected family binding not unique')
                else:
                    family_path, family = family_matches[0]
                    prep = read(family_path.parent / 'PREPARE_INPUT.json')
                    rgb_sha = prep.get('camera_source', {}).get('rgb_input', {}).get('sha256')
                    rgb = imgbysha.get(rgb_sha) or image_index.get(rgb_sha)
                    ref_sha = family.get('reference_sha256')
                    ref = imgbysha.get(ref_sha) or image_index.get(ref_sha)
                    mask = family_path.parent / 'SELECTED_MASK.npy'
                    if not ref_sha == sstate['reference_sha256']:
                        raise ValueError((object_key, ref_sha, sstate['reference_sha256']))
                    if ref is None:
                        missing.append('MISSING: reference image hash binding')
                    if rgb is None:
                        missing.append('MISSING: current RGB hash binding')
                    if not mask.is_file():
                        missing.append('MISSING: selected mask')
                    if ref is not None and rgb is not None and mask.is_file():
                        arr = np.load(mask, allow_pickle=False)
                        expected = prep['camera_source']['intrinsics_input']['mask']['sha256_c_order_bytes']
                        actual_mask_int32 = hashlib.sha256(arr.astype(np.int32).tobytes()).hexdigest()
                        if not actual_mask_int32 == expected:
                            raise ValueError('Evidence binding mismatch')
                        item = {'reference': {'path': str(ref), 'sha256': sha(ref)}, 'rgb': {'path': str(rgb), 'sha256': sha(rgb)}, 'mask': {'path': str(mask), 'sha256': sha(mask)}, 'selection_binding': {'rgb_sha256': sha(rgb), 'mask_file_sha256': sha(mask), 'mask_int32_sha256': expected}}
                        roles, maskcheck = derive_images(item, OUT / 'DERIVED_IMAGES' / attempt['key'])
                        executed_candidate_matches = any((z['position'] == bind['executed_candidate']['position'] and z['quaternion_wxyz'] == bind['executed_candidate']['quaternion'] for z in family['common_family']['candidates']))
                        if not executed_candidate_matches:
                            raise ValueError('Evidence binding mismatch')
                        binding_audit.append({'attempt_id': attempt_id, 'family_path': str(family_path), 'family_sha256': sha(family_path), 'mask_binding': maskcheck, 'executed_candidate_matches_family': True, 'reference_matches_frozen_M1': True})
                for p, stage in stage_matches:
                    if stage.get('capture_state') == 'CAPTURED':
                        image = imgbysha.get(stage.get('image_sha256'))
                        if image is None:
                            missing.append('MISSING: stage image hash ' + p.name)
                        else:
                            roles.append({'role': 'STAGE_' + stage['phase'].upper(), 'phase': stage.get('image_phase', stage['phase']), 'local_path': str(image), 'sha256': sha(image), 'mime': stage.get('mime', 'image/jpeg'), 'record_sha256': sha(p)})
                phases = []
                for _, value in walk(ep.get('recovery', {})):
                    if not isinstance(value, dict) or 'op' not in value:
                        continue
                    holding_check = (value.get('data') or {}).get('holding_check')
                    same_holding = holding_check and holding_check.get('executed_selection') == bind
                    same_binding = any((isinstance(z, dict) and z.get('binding') == bind for _, z in walk(value.get('data') or {})))
                    if same_holding or same_binding:
                        phases.append(value)
                classes, holding, executed = mechanically_classify(bind, evidence.get('descent', {}), phases)
                actual = evidence.get('actual_post_descent', {})
                plan = evidence.get('planned_contact_TCP_xyz_m')
                measured = actual.get('TCP_xyz_m')
                if evidence.get('coordinate_frame') != 'planner world' or evidence.get('position_units') != 'm' or (not valid_xyz(plan)):
                    missing.append('MISSING: planned coordinate')
                for image in roles:
                    image['row_id'] = attempt_id
                rec = {'episode_id': selected['row_id'], 'attempt_id': attempt_id, 'source_round': 'R1', 'generation': 1, 'chronology': evidence.get('descent', {}).get('started_wall_ts'), 'executed': executed, 'candidate_direction': bind.get('executed_candidate', {}).get('source', 'MISSING'), 'planned_contact_TCP_mm': [mm(v) for v in plan] if valid_xyz(plan) else 'MISSING', 'actual_contact_TCP_mm': [mm(v) for v in measured] if valid_xyz(measured) else 'MISSING', 'actual_contact_TCP_time': actual.get('wall_ts', 'MISSING'), 'signed_actual_minus_planned_TCP_mm': [mm(x - y) for x, y in zip(measured, plan)] if valid_xyz(measured) and valid_xyz(plan) else 'MISSING', 'actual_endpoint_unavailable_reason': None if valid_xyz(measured) else 'MISSING: no same-attempt archived contact TCP endpoint', 'mechanical_source_classes': classes, 'holding': lengths_mm(holding), 'executed_phases': [{'op': v['op'], 'ok': v.get('ok')} for v in phases], 'image_roles': [{k: v for k, v in image.items() if k != 'local_path'} for image in roles], 'consumed_memory': {'P_current_z_mm': mm(pstate['corrected_z_m']), 'S_entry_present': True, 'active_template': sstate['template']}, 'missing': missing}
                batch['attempts'].append(rec)
                batch['roles'].extend(roles)
        batch['episodes'].sort(key=lambda x: x['R1_global_ordinal'])
        batch['attempts'].sort(key=lambda x: (x['episode_id'], x['chronology'] or -1))
        batches[object_key] = batch
    return batches,binding_audit
