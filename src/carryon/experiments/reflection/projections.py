"""Project parameter and semantic roles for reflection requests."""
import copy

SEMANTIC_FIELDS = ('episode_id', 'attempt_id', 'source_round', 'generation',
                   'executed', 'image_roles', 'consumed_semantic_state')


def object_m1(batch, role, *, revised=False, unselected=(), extra_roles=()):
    memory = {k: copy.deepcopy(v) for k, v in batch['memory_context'].items()
              if k != 'parent_snapshot_sha256'}
    if not revised:
        facts = {k: copy.deepcopy(batch[k]) for k in
                 ('object_key', 'reflection_round', 'episodes', 'attempts', 'memory_context')}
        facts['memory_context'] = memory
        facts['episodes'] = [{k: v for k, v in ep.items() if k != 'all_recovery_episode_files'}
                             for ep in facts['episodes']]
        if role == 'S':
            facts['attempts'] = [{k: v for k, v in a.items() if k in SEMANTIC_FIELDS}
                                 for a in facts['attempts']]
        else:
            facts['coordinate'] = copy.deepcopy(batch['coordinate'])
        roles = batch['roles'] if role == 'P' else [x for x in batch['roles'] if not x['role'].startswith('STAGE_')]
        return facts, copy.deepcopy(roles)
    ids = {ep['episode_id'] for ep in unselected} if role == 'S' else set()
    episodes = [ep for ep in batch['episodes'] if ep['attempt_ids'] or ep['episode_id'] in ids]
    facts = {'object_key': batch['object_key'], 'reflection_round': batch['reflection_round'],
             'memory_context': memory}
    facts['episodes'] = [{k: v for k, v in ep.items() if k not in
                         ('all_recovery_episode_files', 'no_parameter_attempt_capture')} for ep in episodes]
    facts['attempts'] = copy.deepcopy(batch['attempts'])
    facts['grounding_modes'] = [dict(episode_id=ep['episode_id'], mode='SELECTED_CANDIDATE')
                                for ep in episodes if ep['attempt_ids']]
    roles = copy.deepcopy(batch['roles'])
    if role == 'S':
        fields = (*SEMANTIC_FIELDS, 'executed_phases', 'holding')
        facts['attempts'] = [{k: v for k, v in a.items() if k in fields} for a in facts['attempts']]
        for attempt in facts['attempts']:
            attempt['holding'] = [{k: v for k, v in h.items() if k in
                ('status', 'reason_code', 'trace_phase', 'evaluated_wall_ts', 'execution_binding_status')}
                for h in attempt['holding']]
        facts['memory_context'] = {k: v for k, v in memory.items() if k in ('S_entry_present', 'active_template')}
        if unselected:
            facts['no_point_episodes'] = list(unselected)
            roles.extend(extra_roles)
    else:
        facts['coordinate'] = copy.deepcopy(batch['coordinate'])
        facts['mechanical_failure_requirement'] = False
    return facts, roles


def recorded(batch, role, pose, attempts, episodes, roles, *, pose_audit,
             unselected=(), prior_pose=None, parent_pose_sha256=None):
    generation = batch['reflection_round']
    no_selection_mode = 'NO_POINT_BEFORE_SELECTION' if generation == 1 else 'SELECTOR_REJECTED_BEFORE_EXECUTED_SELECTION'
    facts = {'object_key': batch['object_key'], 'reflection_round': generation,
             'episodes': [{k: v for k, v in ep.items() if k not in
                          ('all_recovery_episode_files', 'no_parameter_attempt_capture')} for ep in episodes],
             'attempts': copy.deepcopy(attempts),
             'memory_context': {k: copy.deepcopy(v) for k, v in batch['memory_context'].items()
                                if k != 'parent_snapshot_sha256'},
             'grounding_modes': [{'episode_id': ep['episode_id'], 'mode': 'SELECTED_CANDIDATE'
                                 if ep['attempt_ids'] else no_selection_mode} for ep in episodes]}
    if role == 'S':
        fields = (*SEMANTIC_FIELDS, 'executed_phases', 'holding')
        facts['attempts'] = [{k: v for k, v in a.items() if k in fields} for a in facts['attempts']]
        for attempt in facts['attempts']:
            attempt['holding'] = [{k: v for k, v in h.items() if k in
                ('status', 'reason_code', 'trace_phase', 'evaluated_wall_ts', 'execution_binding_status')}
                for h in attempt['holding']]
        facts['memory_context'] = {k: v for k, v in facts['memory_context'].items()
                                   if k in ('S_entry_present', 'active_template')}
        if unselected:
            facts['no_point_episodes' if generation == 1 else 'unselected_episodes'] = list(unselected)
    else:
        executed = [a for a in attempts if a['executed'] and a['planned_contact_TCP_mm']]
        if not executed:
            raise ValueError('P requires an executed parameter observation')
        facts['memory_context']['anchor_z_mm'] = executed[-1]['planned_contact_TCP_mm'][2]
        if generation == 2:
            facts['memory_context']['P_current_z_mm'] = prior_pose['corrected_z_m'] * 1000 if prior_pose else None
            facts['memory_context']['anchor_z_mm'] = prior_pose['corrected_z_m'] * 1000 if prior_pose else executed[-1]['planned_contact_TCP_mm'][2]
            facts['memory_context']['anchor_source'] = ('frozen M1 active parameter for this object and pose'
                if prior_pose else 'last canonical-order actual attempt planned z; no existing same-pose P record')
            facts['memory_context']['parent_pose_snapshot_sha256'] = parent_pose_sha256
        facts['coordinate'] = copy.deepcopy(batch['coordinate'])
        facts['mechanical_failure_requirement'] = False
        ids = {ep['episode_id'] for ep in episodes}
        facts['pose_context'] = {'memory_key': {'object_key': batch['object_key'], 'pose': pose},
            'sources': [{'episode_id': r['episode_id'], 'pose': r['pose'], 'family_id': r['family_id'],
                         'request_sha256': r['wire_sha256'], 'response_sha256': r['response_sha256'],
                         'visual_basis': r['visual_basis']} for r in pose_audit if r['episode_id'] in ids],
            'scope': 'VLM posture indexing of the selected target observation; not an identity confirmation, gripper orientation, collision truth or cause certification. Same-pose P only; UNKNOWN is not borrowed.'}
    return facts, copy.deepcopy(roles)

def object_m2(batch, role):
    object_key = batch['object_key']
    coordinate = batch['coordinate']
    facts = {'object_key': object_key, 'reflection_round': 2, 'source_round': 'R1', 'episodes': copy.deepcopy(batch['episodes'])}
    if role == 'S':
        facts['episodes'] = [{k: e[k] for k in ('episode_id', 'R1_global_ordinal', 'official_task_failure', 'terminal_feedback_scope', 'recovery_triggered', 'recovery_failure_type', 'recovery_outcome', 'grounding', 'selected_evidence_status')} for e in facts['episodes']]
        facts['memory_context'] = {'S_entry_present': True, 'active_template': batch['memory_context']['active_template']}
        facts['attempts'] = [{'episode_id': a['episode_id'], 'attempt_id': a['attempt_id'], 'source_round': a['source_round'], 'generation': a['generation'], 'executed': a['executed'], 'image_roles': [x for x in a['image_roles'] if not x['role'].startswith('STAGE_')], 'consumed_semantic_state': {'S_entry_present': True, 'active_template': batch['memory_context']['active_template']}} for a in batch['attempts']]
        selected_roles = [x for x in batch['roles'] if not x['role'].startswith('STAGE_')]
    else:
        facts['memory_context'] = {'P_current_z_mm': batch['memory_context']['P_current_z_mm'], 'anchor_z_mm': batch['memory_context']['anchor_z_mm'], 'anchor_source': batch['memory_context']['anchor_source']}
        facts['coordinate'] = coordinate
        facts['attempts'] = copy.deepcopy(batch['attempts'])
        selected_roles = copy.deepcopy(batch['roles'])
    return facts, selected_roles
