"""Per-call execution provenance; no candidate search, state machine or new retry."""
import copy
from carryon.recovery.schema import GraspProposal

def proposal(g):
    return {'position': list(g.position), 'quaternion': list(g.quaternion), 'score': g.score, 'source': g.source, 'predicted_width_m': g.predicted_width_m}

def register(backend, child, parent):
    scope = getattr(backend, '_selection_origin_scope', None)
    if scope is None:
        return child
    entry = scope['origins'].get(id(parent))
    if entry is None or entry[0] is not parent:
        raise ValueError('candidate origin missing: cannot infer from source/pose')
    scope['origins'][id(child)] = (child, entry[1], proposal(child))
    return child

def begin(backend, ctx, frozen):
    pool = ctx.grasp_candidates
    j = ctx.grasp_candidate_index
    if type(j) is not int or not 0 <= j < len(pool):
        raise ValueError('preselected original pool index missing')
    if len(pool) > 15 or len({id(g) for g in pool}) != len(pool):
        raise ValueError('candidate pool invalid or larger than frozen top_k=15')
    scope = {'pool': list(pool), 'pool_snapshot': [proposal(g) for g in pool], 'ctx_target': ctx.grasp, 'ctx_target_snapshot': proposal(ctx.grasp), 'preselected_original_index': j, 'executed': None, 'physical_count': 0, 'origins': {id(g): (g, i, proposal(g)) for i, g in enumerate(pool)}}
    backend._selection_origin_scope = scope
    backend.last_goalset_execution_debug = {}
    scope['origins'][id(frozen)] = (frozen, j, proposal(frozen))
    return scope

def capture_before_motion(backend, candidate, local_index):
    scope = getattr(backend, '_selection_origin_scope', None)
    if scope is None:
        return None
    if [proposal(g) for g in scope['pool']] != scope['pool_snapshot']:
        raise ValueError('frozen original pool mutated before motion')
    entry = scope['origins'].get(id(candidate))
    if entry is None or entry[0] is not candidate or proposal(candidate) != entry[2]:
        raise ValueError('executed selection has missing/changed original binding')
    if scope['physical_count'] != 0:
        raise ValueError('more than one physical execution inside family forbidden')
    record = {'schema': 'carryon-executed-candidate-origin/v1', 'original_pool_index': entry[1], 'local_goalset_index': int(local_index), 'preselected_original_index': scope['preselected_original_index'], 'original_pool_count': len(scope['pool']), 'original_candidate': copy.deepcopy(scope['pool_snapshot'][entry[1]]), 'executed_candidate': proposal(candidate), 'physical_execution_ordinal': 1}
    scope['physical_count'] = 1
    scope['executed'] = copy.deepcopy(record)
    return record

def finish(backend, ctx, scope, family_debug, motion_ok):
    try:
        if len(ctx.grasp_candidates) != len(scope['pool']) or any((a is not b for a, b in zip(ctx.grasp_candidates, scope['pool']))) or [proposal(g) for g in ctx.grasp_candidates] != scope['pool_snapshot'] or (ctx.grasp is not scope['ctx_target']) or (proposal(ctx.grasp) != scope['ctx_target_snapshot']):
            raise ValueError('frozen target or original candidate pool mutated during attempt')
        record = scope['executed']
        debug = getattr(backend, 'last_goalset_execution_debug', {}) or {}
        if motion_ok and record is None:
            raise ValueError('successful motion lacks original candidate binding')
        if bool(debug.get('physical_motion_started')) != (record is not None):
            raise ValueError('physical motion and candidate provenance disagree')
        family_debug['preselected_target_xyz'] = list(scope['ctx_target_snapshot']['position'])
        family_debug['preselected_original_index'] = scope['preselected_original_index']
        family_debug['original_pool_count'] = len(scope['pool'])
        family_debug['physical_execution_count'] = scope['physical_count']
        family_debug['executed_selection'] = copy.deepcopy(record)
        if record is not None:
            ctx.grasp = GraspProposal(**copy.deepcopy(record['executed_candidate']))
            ctx.grasp_candidate_index = record['original_pool_index']
            ctx.grasp_candidate_index_history.append(ctx.grasp_candidate_index)
            ctx.executed_grasp_binding = copy.deepcopy(record)
            family_debug['frozen_target_xyz'] = list(ctx.grasp.position)
        elif getattr(ctx, 'executed_grasp_binding', None) is not None:
            old = ctx.executed_grasp_binding
            ctx.grasp = GraspProposal(**copy.deepcopy(old['executed_candidate']))
            ctx.grasp_candidate_index = old['original_pool_index']
        backend.last_topdown_family_debug = copy.deepcopy(family_debug)
        return record
    finally:
        backend._selection_origin_scope = None

def binding(ctx):
    return copy.deepcopy(getattr(ctx, 'executed_grasp_binding', None))

def invoke(backend, family, *args, **kwargs):
    from carryon.recovery.backend import MotionResult
    try:
        return family(*args, **kwargs)
    except Exception as exc:
        return MotionResult(ok=False, reason='family exception: ' + type(exc).__name__ + ': ' + str(exc))
