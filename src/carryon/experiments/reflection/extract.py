"""Evidence extraction derived from the two formal collectors."""

from pathlib import Path

import json,hashlib,math

import numpy as np

from .evidence import discover_attempts,walk,mechanically_classify,lengths_mm,derive_images,wire,sha,mm,read

from PIL import Image

def extract_object(sources, source_roots, references, output):
    O = Path(output)
    batches, audit = {}, []
    for ob in sources['objects']:
        key = ob['object_name']
        b = {'object_key': key, 'reflection_round': 1, 'episodes': [], 'attempts': [], 'roles': [], 'S_missing': [], 'P_missing': [], 'memory_context': {'P_current_z_mm': None, 'S_entry_present': False, 'active_template': None}, 'coordinate': {'frame': 'planner world', 'reference': 'contact_TCP', 'positive_z': 'up', 'length_unit': 'mm', 'parser_bounds_mm': [0, 250], 'maximum_delta_from_anchor_mm': 200}}
        batches[key] = b
        for src in ob['selected_sources']:
            rid = src['row_id']
            r = source_roots[rid]
            eps = list((r / 'recovery_episodes').glob('*.json'))
            epctx = {'episode_id': rid, 'official_task_failure': True, 'terminal_feedback_scope': 'whole task only; not identity/grasp/z diagnosis', 'all_recovery_episode_files': [], 'attempt_ids': []}
            b['episodes'].append(epctx)
            imgbysha = {sha(p): p for p in r.rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bin']}
            pvis = [(p, read(p)) for p in (r / 'parameter_visual').glob('*/INPUT_EVIDENCE.json')]
            stages = [(p, read(p)) for p in (r / 'attempt_evidence').glob('*/*.json')]
            families = [(p, read(p)) for p in (r / 'common_perception_raw').glob('*/FAMILY.json')]
            if not eps:
                b['S_missing'].append(rid + ':NO_RECOVERY_EPISODE')
                b['P_missing'].append(rid + ':NO_RECOVERY_EPISODE')
                continue
            for epfile in eps:
                ep = read(epfile)
                epctx['all_recovery_episode_files'].append({'path': str(epfile), 'sha256': sha(epfile)})
                ats = discover_attempts(ep)
                if not ats:
                    epctx['no_parameter_attempt_capture'] = True
                for a in ats:
                    e = a['evidence']
                    bind = e['binding']
                    aid = rid + '/' + a['key']
                    epctx['attempt_ids'].append(aid)
                    matches = [(p, v) for p, v in pvis if v.get('binding') == bind and v.get('descent', {}).get('started_wall_ts') == e.get('descent', {}).get('started_wall_ts')]
                    vm = matches[0][1] if len(matches) == 1 else None
                    st = [(p, v) for p, v in stages if v.get('executed_selection') == bind]
                    famids = {v.get('common_family_id') for p, v in st if v.get('common_family_id')}
                    fm = [(p, v) for p, v in families if p.parent.name in famids]
                    miss = []
                    roles = []
                    if len(fm) != 1:
                        miss.append('FAMILY_BINDING_NOT_UNIQUE')
                    else:
                        fp, f = fm[0]
                        prep = read(fp.parent / 'PREPARE_INPUT.json')
                        ref = imgbysha.get(f.get('reference_sha256')) or references.get(f.get('reference_sha256'))
                        mask = fp.parent / 'SELECTED_MASK.npy'
                        cur = r / 'visual_object_memory/CURRENT_RGB.png'
                        selection = r / 'visual_object_memory/SELECTION_EVIDENCE.json'
                        rgb = None
                        if cur.is_file() and selection.is_file() and mask.is_file():
                            sel = read(selection)
                            sm = r / 'visual_object_memory' / f"candidate_{sel['selected_candidate_id']:02d}_mask.npy"
                            sv = r / 'visual_object_memory/SELECTED_OVERLAY.png'
                            if not (sel['object_name'] == key and sel['selected_candidate_id'] == f['selected_candidate_id']):
                                raise ValueError('Evidence binding mismatch')
                            ar0 = np.load(sm, allow_pickle=False) > 0
                            ar1 = np.load(mask, allow_pickle=False) > 0
                            if not np.array_equal(ar0, ar1):
                                raise ValueError('Evidence binding mismatch')
                            rr = np.array(Image.open(cur).convert('RGB'))
                            ov = rr.copy()
                            ov[ar0] = (0.55 * ov[ar0] + 0.45 * np.array([40, 255, 40])).astype(np.uint8)
                            if not np.array_equal(ov, np.array(Image.open(sv).convert('RGB'))):
                                raise ValueError('Evidence binding mismatch')
                            rgb = cur
                        if ref is None:
                            miss.append('REFERENCE_IMAGE_MISSING')
                        if rgb is None:
                            miss.append('CURRENT_RGB_HASH_BINDING_MISSING')
                        if not mask.is_file():
                            miss.append('SELECTED_MASK_MISSING')
                        if ref and rgb and mask.is_file():
                            ar = np.load(mask, allow_pickle=False)
                            expected = prep['camera_source']['intrinsics_input']['mask']['sha256_c_order_bytes']
                            if not hashlib.sha256(ar.astype(np.int32).tobytes()).hexdigest() == expected:
                                raise ValueError('Evidence binding mismatch')
                            item = {'reference': {'path': str(ref), 'sha256': sha(ref)}, 'rgb': {'path': str(rgb), 'sha256': sha(rgb)}, 'mask': {'path': str(mask), 'sha256': sha(mask)}, 'selection_binding': {'rgb_sha256': sha(rgb), 'mask_file_sha256': sha(mask), 'mask_int32_sha256': expected}}
                            roles, maskcheck = derive_images(item, O / 'DERIVED_IMAGES' / a['key'])
                            for z in roles:
                                if z['role'] != 'TARGET_REFERENCE':
                                    z['phase'] = 'selector-stage current RGB and mask, original algorithm selection; not the detector later JPEG'
                            audit.append({'attempt_id': aid, 'family_path': str(fp), 'family_sha256': sha(fp), 'mask_binding': maskcheck, 'executed_candidate_matches_family': any((z['position'] == bind['executed_candidate']['position'] and z['quaternion_wxyz'] == bind['executed_candidate']['quaternion'] for z in f['common_family']['candidates']))})
                            if not audit[-1]['executed_candidate_matches_family']:
                                raise ValueError('Evidence binding mismatch')
                    for p, v in st:
                        if v.get('capture_state') == 'CAPTURED':
                            im = imgbysha.get(v.get('image_sha256'))
                            if im is None:
                                miss.append('STAGE_IMAGE_SHA_MISSING:' + p.name)
                            else:
                                roles.append({'role': 'STAGE_' + v['phase'].upper(), 'phase': v.get('image_phase', v['phase']), 'local_path': str(im), 'sha256': sha(im), 'mime': v.get('mime', 'image/jpeg'), 'record_sha256': sha(p)})
                    if not any((z['role'].startswith('STAGE_GOTO') for z in roles)):
                        miss.append('DESCENT_STAGE_IMAGE_MISSING')
                    phases = []
                    for _, v in walk(ep.get('recovery', {})):
                        if not isinstance(v, dict) or 'op' not in v:
                            continue
                        h = (v.get('data') or {}).get('holding_check')
                        same = h and h.get('executed_selection') == bind
                        if same or any((isinstance(z, dict) and z.get('binding') == bind for _, z in walk(v.get('data') or {}))):
                            phases.append(v)
                    classes, holding, executed = mechanically_classify(bind, e.get('descent', {}), phases)
                    actual = e.get('actual_post_descent', {})
                    plan = e.get('planned_contact_TCP_xyz_m')
                    act = actual.get('TCP_xyz_m')
                    validnum = lambda x: isinstance(x, list) and len(x) == 3 and all((type(v) in (int, float) and math.isfinite(v) for v in x))
                    if e.get('coordinate_frame') != 'planner world' or e.get('position_units') != 'm' or (not validnum(plan)):
                        miss.append('PLANNED_COORDINATE_MISSING')
                    for im in roles:
                        im['row_id'] = aid
                    rec = {'episode_id': rid, 'attempt_id': aid, 'source_round': 'R0', 'generation': 0, 'chronology': e.get('descent', {}).get('started_wall_ts'), 'executed': executed, 'candidate_direction': bind.get('executed_candidate', {}).get('source'), 'planned_contact_TCP_mm': [mm(v) for v in plan] if validnum(plan) else None, 'actual_contact_TCP_mm': [mm(v) for v in act] if validnum(act) else None, 'actual_contact_TCP_time': actual.get('wall_ts'), 'signed_actual_minus_planned_TCP_mm': [mm(x - y) for x, y in zip(act, plan)] if validnum(act) and validnum(plan) else None, 'actual_endpoint_unavailable_reason': None if validnum(act) else 'No same-attempt archived contact TCP endpoint', 'mechanical_source_classes': classes, 'holding': lengths_mm(holding), 'executed_phases': [{'op': v['op'], 'ok': v.get('ok')} for v in phases], 'image_roles': [{k: v for k, v in im.items() if k != 'local_path'} for im in roles], 'consumed_semantic_state': {'S_entry_present': False, 'active_template': None}, 'missing': miss}
                    b['attempts'].append(rec)
                    b['roles'].extend(roles)
                    b['S_missing'].extend((aid + ':' + m for m in miss if not m.startswith(('PLANNED_', 'DESCENT_', 'STAGE_'))))
                    b['P_missing'].extend((aid + ':' + m for m in miss))
            if not epctx['attempt_ids']:
                b['S_missing'].append(rid + ':NO_BOUND_ATTEMPT_IMAGES')
                b['P_missing'].append(rid + ':NO_ACTUAL_PARAMETER_EVIDENCE')
        b['attempts'].sort(key=lambda x: (x['episode_id'], x['chronology'] or -1))
        actual = [x for x in b['attempts'] if x['executed'] and x['planned_contact_TCP_mm']]
        b['memory_context']['anchor_z_mm'] = actual[-1]['planned_contact_TCP_mm'][2] if actual else None
        b['memory_context']['anchor_source'] = 'last canonical-order actual attempt planned z; not a recommended value'
        b['S_status'] = 'NO_CALL_NO_FAILED_EPISODES' if not b['episodes'] else 'NO_CALL_MISSING_EVIDENCE' if b['S_missing'] else 'READY_CHECK_ONLY'
        b['P_status'] = 'NO_CALL_NO_FAILED_EPISODES' if not b['episodes'] else 'NO_CALL_MISSING_EVIDENCE' if b['P_missing'] else 'READY_CHECK_ONLY' if actual and any((a['mechanical_source_classes'] for a in actual)) else 'NO_CALL_NO_QUALIFYING_MECHANICAL_FAILURE'
    return batches, audit

def extract_recorded(sources, source_roots, references, output):
    O = Path(output)
    batches, audit = {}, []
    for ob in sources['objects']:
        key = ob['object_name']
        b = {'object_key': key, 'reflection_round': 1, 'episodes': [], 'attempts': [], 'roles': [], 'S_missing': [], 'P_missing': [], 'memory_context': {'P_current_z_mm': None, 'S_entry_present': False, 'active_template': None}, 'coordinate': {'frame': 'planner world', 'reference': 'contact_TCP', 'positive_z': 'up', 'length_unit': 'mm', 'parser_bounds_mm': [0, 250], 'maximum_delta_from_anchor_mm': 200}}
        batches[key] = b
        for src in ob['selected_sources']:
            rid = src['row_id']
            r = source_roots[rid]
            eps = list((r / 'recovery_episodes').glob('*.json'))
            epctx = {'episode_id': rid, 'official_task_failure': True, 'terminal_feedback_scope': 'whole task only; not identity/grasp/z diagnosis', 'all_recovery_episode_files': [], 'attempt_ids': []}
            b['episodes'].append(epctx)
            imgbysha = {sha(p): p for p in r.rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.bin']}
            pvis = [(p, read(p)) for p in (r / 'parameter_visual').glob('*/INPUT_EVIDENCE.json')]
            stages = [(p, read(p)) for p in (r / 'attempt_evidence').glob('*/*.json')]
            families = [(p, read(p)) for p in (r / 'common_perception_raw').glob('*/FAMILY.json')]
            if not eps:
                b['S_missing'].append(rid + ':NO_RECOVERY_EPISODE')
                b['P_missing'].append(rid + ':NO_RECOVERY_EPISODE')
                continue
            for epfile in eps:
                ep = read(epfile)
                epctx['all_recovery_episode_files'].append({'path': str(epfile), 'sha256': sha(epfile)})
                ats = discover_attempts(ep)
                if not ats:
                    epctx['no_parameter_attempt_capture'] = True
                for a in ats:
                    e = a['evidence']
                    bind = e['binding']
                    aid = rid + '/' + a['key']
                    epctx['attempt_ids'].append(aid)
                    matches = [(p, v) for p, v in pvis if v.get('binding') == bind and v.get('descent', {}).get('started_wall_ts') == e.get('descent', {}).get('started_wall_ts')]
                    vm = matches[0][1] if len(matches) == 1 else None
                    st = [(p, v) for p, v in stages if v.get('executed_selection') == bind]
                    famids = {v.get('common_family_id') for p, v in st if v.get('common_family_id')}
                    fm = [(p, v) for p, v in families if p.parent.name in famids]
                    miss = []
                    roles = []
                    if len(fm) != 1:
                        miss.append('FAMILY_BINDING_NOT_UNIQUE')
                    else:
                        fp, f = fm[0]
                        prep = read(fp.parent / 'PREPARE_INPUT.json')
                        rgbs = prep.get('camera_source', {}).get('rgb_input', {}).get('sha256')
                        rgb = imgbysha.get(rgbs)
                        ref = imgbysha.get(f.get('reference_sha256')) or references.get(f.get('reference_sha256'))
                        mask = fp.parent / 'SELECTED_MASK.npy'
                        if rgb is None:
                            cur = r / 'visual_object_memory/CURRENT_RGB.png'
                            if cur.is_file() and sha(cur) == rgbs:
                                rgb = cur
                        if ref is None:
                            miss.append('REFERENCE_IMAGE_MISSING')
                        if rgb is None:
                            miss.append('CURRENT_RGB_HASH_BINDING_MISSING')
                        if not mask.is_file():
                            miss.append('SELECTED_MASK_MISSING')
                        if ref and rgb and mask.is_file():
                            ar = np.load(mask, allow_pickle=False)
                            expected = prep['camera_source']['intrinsics_input']['mask']['sha256_c_order_bytes']
                            if not hashlib.sha256(ar.astype(np.int32).tobytes()).hexdigest() == expected:
                                raise ValueError('Evidence binding mismatch')
                            item = {'reference': {'path': str(ref), 'sha256': sha(ref)}, 'rgb': {'path': str(rgb), 'sha256': sha(rgb)}, 'mask': {'path': str(mask), 'sha256': sha(mask)}, 'selection_binding': {'rgb_sha256': sha(rgb), 'mask_file_sha256': sha(mask), 'mask_int32_sha256': expected}}
                            roles, maskcheck = derive_images(item, O / 'DERIVED_IMAGES' / a['key'])
                            audit.append({'attempt_id': aid, 'family_path': str(fp), 'family_sha256': sha(fp), 'mask_binding': maskcheck, 'executed_candidate_matches_family': any((z['position'] == bind['executed_candidate']['position'] and z['quaternion_wxyz'] == bind['executed_candidate']['quaternion'] for z in f['common_family']['candidates']))})
                            if not audit[-1]['executed_candidate_matches_family']:
                                raise ValueError('Evidence binding mismatch')
                    for p, v in st:
                        if v.get('capture_state') == 'CAPTURED':
                            im = imgbysha.get(v.get('image_sha256'))
                            if im is None:
                                miss.append('STAGE_IMAGE_SHA_MISSING:' + p.name)
                            else:
                                roles.append({'role': 'STAGE_' + v['phase'].upper(), 'phase': v.get('image_phase', v['phase']), 'local_path': str(im), 'sha256': sha(im), 'mime': v.get('mime', 'image/jpeg'), 'record_sha256': sha(p)})
                    if not any((z['role'].startswith('STAGE_GOTO') for z in roles)):
                        miss.append('DESCENT_STAGE_IMAGE_MISSING')
                    phases = []
                    for _, v in walk(ep.get('recovery', {})):
                        if not isinstance(v, dict) or 'op' not in v:
                            continue
                        h = (v.get('data') or {}).get('holding_check')
                        same = h and h.get('executed_selection') == bind
                        if same or any((isinstance(z, dict) and z.get('binding') == bind for _, z in walk(v.get('data') or {}))):
                            phases.append(v)
                    for sp, sv in sorted(st):
                        h = sv.get('data', {}).get('holding_check')
                        if h and (not any((ph.get('data', {}).get('holding_check') == h for ph in phases))):
                            phases.append({'op': sv['phase'], 'ok': None, 'data': sv['data']})
                    classes, holding, executed = mechanically_classify(bind, e.get('descent', {}), phases)
                    actual = e.get('actual_post_descent', {})
                    plan = e.get('planned_contact_TCP_xyz_m')
                    act = actual.get('TCP_xyz_m')
                    validnum = lambda x: isinstance(x, list) and len(x) == 3 and all((type(v) in (int, float) and math.isfinite(v) for v in x))
                    if e.get('coordinate_frame') != 'planner world' or e.get('position_units') != 'm' or (not validnum(plan)):
                        miss.append('PLANNED_COORDINATE_MISSING')
                    for im in roles:
                        im['row_id'] = aid
                    rec = {'episode_id': rid, 'attempt_id': aid, 'source_round': 'R0', 'generation': 0, 'chronology': e.get('descent', {}).get('started_wall_ts'), 'executed': executed, 'candidate_direction': bind.get('executed_candidate', {}).get('source'), 'planned_contact_TCP_mm': [mm(v) for v in plan] if validnum(plan) else None, 'actual_contact_TCP_mm': [mm(v) for v in act] if validnum(act) else None, 'actual_contact_TCP_time': actual.get('wall_ts'), 'signed_actual_minus_planned_TCP_mm': [mm(x - y) for x, y in zip(act, plan)] if validnum(act) and validnum(plan) else None, 'actual_endpoint_unavailable_reason': None if validnum(act) else 'No same-attempt archived contact TCP endpoint', 'mechanical_source_classes': classes, 'holding': lengths_mm(holding), 'executed_phases': [{'op': v['op'], 'ok': v.get('ok')} for v in phases], 'image_roles': [{k: v for k, v in im.items() if k != 'local_path'} for im in roles], 'consumed_semantic_state': {'S_entry_present': False, 'active_template': None}, 'missing': miss}
                    b['attempts'].append(rec)
                    b['roles'].extend(roles)
                    b['S_missing'].extend((aid + ':' + m for m in miss if not m.startswith(('PLANNED_', 'DESCENT_', 'STAGE_'))))
                    b['P_missing'].extend((aid + ':' + m for m in miss))
            if not epctx['attempt_ids']:
                b['S_missing'].append(rid + ':NO_BOUND_ATTEMPT_IMAGES')
                b['P_missing'].append(rid + ':NO_ACTUAL_PARAMETER_EVIDENCE')
        b['attempts'].sort(key=lambda x: (x['episode_id'], x['chronology'] or -1))
        actual = [x for x in b['attempts'] if x['executed'] and x['planned_contact_TCP_mm']]
        b['memory_context']['anchor_z_mm'] = actual[-1]['planned_contact_TCP_mm'][2] if actual else None
        b['memory_context']['anchor_source'] = 'last canonical-order actual attempt planned z; not a recommended value'
        b['S_status'] = 'NO_CALL_NO_FAILED_EPISODES' if not b['episodes'] else 'NO_CALL_MISSING_EVIDENCE' if b['S_missing'] else 'READY_CHECK_ONLY'
        b['P_status'] = 'NO_CALL_NO_FAILED_EPISODES' if not b['episodes'] else 'NO_CALL_MISSING_EVIDENCE' if b['P_missing'] else 'READY_CHECK_ONLY' if actual else 'NO_CALL_NO_ACTUAL_PARAMETER_EVIDENCE'
    return batches, audit

