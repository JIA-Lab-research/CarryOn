"""Attribute unresolved recovery failures using recorded execution evidence."""
import copy
import math
from carryon.recovery.backend import GelloRecoveryBackend as AcceptedBackend
GRASP_OPS = frozenset({'goto_grasp_topdown_family', 'goto_grasp_goalset', 'goto_grasp'})
RESOLVING_RETRIES = frozenset({'assert_holding_pose_retry', 'retry'})

def physical_target_record(item):
    """Bind only the final recorded physical attempt; never search older matches."""
    data = getattr(item, 'data', None) or {}
    ok = bool(getattr(item, 'ok', False))
    if not isinstance(data, dict):
        return None
    if 'fresh_attempts' in data:
        attempts = data['fresh_attempts']
        if not isinstance(attempts, list) or not attempts or (not isinstance(attempts[-1], dict)):
            return None
        last = attempts[-1]
        if type(last.get('ok')) is not bool or last['ok'] != ok:
            return None
        family = last.get('topdown_family')
        path = 'data.fresh_attempts[%d].topdown_family.frozen_target_xyz' % (len(attempts) - 1)
        attempt_number = last.get('attempt')
        pose_index = last.get('pose_index')
    else:
        family = data.get('topdown_family')
        path = 'data.topdown_family.frozen_target_xyz'
        attempt_number = None
        pose_index = None
    target = family.get('frozen_target_xyz') if isinstance(family, dict) else None
    valid = isinstance(target, (list, tuple)) and len(target) == 3 and all((isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x) for x in target))
    if not valid:
        return None
    outer = data.get('topdown_family')
    if 'fresh_attempts' in data and isinstance(outer, dict) and ('frozen_target_xyz' in outer):
        if outer['frozen_target_xyz'] != target:
            return None
    return {'target': list(target), 'target_path': path, 'attempt': attempt_number, 'pose_index': pose_index, 'ok': ok}

def failure_view(steps):
    """Project existing evidence; never mutate the original StepResult tree."""
    nodes = []
    resolved = []
    flat = []

    def visit(items, prefix=(), retry_resolved=False):
        for i, item in enumerate(items or []):
            path = prefix + (i,)
            op = str(getattr(item, 'op', ''))
            ok = bool(getattr(item, 'ok', False))
            if op.startswith('fixv2_postrow'):
                continue
            children = getattr(item, 'children', None) or []
            desc_resolved = retry_resolved or (op in RESOLVING_RETRIES and ok)
            node = {'path': '.'.join(map(str, path)), 'item': item, 'op': op, 'ok': ok, 'resolved': retry_resolved}
            nodes.append(node)
            if children:
                visit(children, path, desc_resolved)
            else:
                flat.append(node)
            if not ok and retry_resolved:
                resolved.append(node['path'])
    visit(steps)
    active = None
    nohold = None
    last_goto = None
    for node in flat:
        op = node['op']
        item = node['item']
        ok = node['ok']
        if op in GRASP_OPS:
            binding = physical_target_record(item)
            active = {'goto_path': node['path'], 'goto_ok': ok, 'target': None if binding is None else binding['target'], 'target_binding': binding, 'closed': False, 'ever_held': False}
            last_goto = node
            nohold = None
        elif op == 'close_gripper':
            if active is not None:
                active['closed'] = ok
                active['ever_held'] = False
                active['close_path'] = node['path']
        elif op == 'assert_holding' and active is not None:
            if ok:
                active['ever_held'] = True
                nohold = None
            elif active['goto_ok'] and active['closed'] and (not active['ever_held']) and (not node['resolved']):
                nohold = {**active, 'assert_path': node['path'], 'assert_item': item}
    semantic = []
    for node in nodes:
        if node['ok'] or node['resolved']:
            continue
        item = copy.copy(node['item'])
        item.children = []
        semantic.append(item)
    return {'semantic_steps': semantic, 'unresolved_nohold': nohold, 'last_goto': last_goto, 'raw_failed_paths': [n['path'] for n in nodes if not n['ok']], 'resolved_failed_paths': resolved, 'unresolved_failed_paths': [n['path'] for n in nodes if not n['ok'] and (not n['resolved'])]}

class FailureEvidenceAttributionBackend(AcceptedBackend):

    def reflect_parameter_memory_at_recovery_end(self, steps):
        view = failure_view(steps)
        chosen = view['unresolved_nohold']
        context = copy.copy(self)
        context._last_executed_grasp_xyz_m = None
        context._last_descent_jam_signal = None
        selected_steps = []
        mode = 'NO_UNRESOLVED_BOUND_FAILURE'
        jam_binding = None
        if chosen is not None and chosen['target'] is not None:
            context._last_executed_grasp_xyz_m = chosen['target']
            from types import SimpleNamespace
            selected_steps = [SimpleNamespace(op='close_gripper', ok=True, children=[]), copy.copy(chosen['assert_item'])]
            selected_steps[-1].children = []
            mode = 'SAME_ATTEMPT_GRASP_NO_HOLD'
        else:
            node = view['last_goto']
            signal = getattr(self, '_last_descent_jam_signal', None)
            if node is not None and (not node['ok']) and (not node['resolved']) and (signal is not None):
                binding = physical_target_record(node['item'])
                target = None if binding is None else binding['target']
                reason = str(getattr(node['item'], 'reason', '')).lower()
                if isinstance(target, (list, tuple)) and len(target) == 3 and (float(target[2]) == float(signal.current_z_m)) and ('descent' in reason) and ('jam' in reason):
                    context._last_descent_jam_signal = signal
                    mode = 'SAME_ATTEMPT_DESCENT_JAM'
                    jam_binding = {**binding, 'goto_path': node['path']}
        evidence = AcceptedBackend.reflect_parameter_memory_at_recovery_end(context, selected_steps)
        evidence['failure_attribution'] = {'schema': 'carryon-failure-evidence-attribution/v1', 'mode': mode, 'raw_failed_paths': view['raw_failed_paths'], 'resolved_failed_paths': view['resolved_failed_paths'], 'unresolved_failed_paths': view['unresolved_failed_paths'], 'source_logs_unchanged': True, 'bound_goto_path': (None if jam_binding is None else jam_binding['goto_path']) if chosen is None else chosen['goto_path'], 'bound_assert_path': None if chosen is None else chosen['assert_path'], 'bound_target_xyz_m': (None if jam_binding is None else jam_binding['target']) if chosen is None else chosen['target'], 'physical_target_binding': jam_binding if chosen is None else chosen['target_binding']}
        self.last_jam_z_reflection_debug = dict(evidence)
        return evidence

    def reflect_semantic_memory_at_recovery_end(self, steps):
        view = failure_view(steps)
        evidence = AcceptedBackend.reflect_semantic_memory_at_recovery_end(self, view['semantic_steps'])
        evidence['failure_attribution'] = {'schema': 'carryon-failure-evidence-attribution/v1', 'raw_failed_paths': view['raw_failed_paths'], 'resolved_failed_paths': view['resolved_failed_paths'], 'unresolved_failed_paths': view['unresolved_failed_paths'], 'source_logs_unchanged': True}
        self.last_semantic_reflection_debug = dict(evidence)
        return evidence

def bind_failure_evidence_hooks(backend_class):
    """CPU/future explicit assembly hook: retain the actual adapter class and all other methods."""
    before = {name: getattr(backend_class, name) for name in ('perceive_grasps', 'goto_grasp_topdown_family')}
    backend_class.reflect_parameter_memory_at_recovery_end = FailureEvidenceAttributionBackend.reflect_parameter_memory_at_recovery_end
    backend_class.reflect_semantic_memory_at_recovery_end = FailureEvidenceAttributionBackend.reflect_semantic_memory_at_recovery_end
    assert all((getattr(backend_class, k) is v for k, v in before.items()))
    return backend_class
