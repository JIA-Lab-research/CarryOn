"""Common direction retry: two primitive sequences, no perception/P in retry."""
import copy
import functools
MODE = 'common_short_long_v1'
BODY_OPS = ('open_gripper', 'goto_grasp_topdown_family', 'close_gripper', 'lift', 'assert_holding')

def terminal_empty_after_unstable(hold, *, now):
    import math
    if not isinstance(hold, dict):
        return False
    if not (hold.get('reason_code') == 'LIFT_WIDTH_UNSTABLE' and hold.get('status') == 'unstable' and (hold.get('semantics') == 'width_hold_proxy/v1') and (hold.get('execution_binding_status') == 'MATCH') and (hold.get('trace_phase', 'post_lift') == 'post_lift')):
        return False
    obs = hold.get('terminal_empty_observation')
    if not isinstance(obs, dict) or obs.get('schema') != 'terminal_width_observation/v1':
        return False
    if obs.get('current_and_last_dynamic_empty') is not True or obs.get('earlier_dynamic_nonempty') is not True:
        return False
    max_age = obs.get('measurement_max_age_s')
    numeric = lambda x: isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x)
    if not numeric(now) or not numeric(max_age) or max_age <= 0:
        return False
    for key in ['state_producer_ts', 'last_dynamic_ts', 'evaluated_wall_ts']:
        stamp = obs.get(key)
        if not numeric(stamp) or not 0 <= now - stamp <= max_age:
            return False
    return True

def install(guard):
    cls = guard.STATE['class']
    candidate = guard.STATE['candidate']
    old = cls._run_retry
    Step = candidate.RecoveryStep
    Result = candidate.StepResult

    @functools.wraps(old)
    def run(self, backend, ctx, p):
        if p.get('mode') != MODE:
            return old(self, backend, ctx, p)
        children = []
        ledger = {'schema': MODE, 'max_attempts': 2, 'attempts': [], 'P_read_in_retry': False, 'depth_retry': False}

        def finish(ok, reason):
            ledger['index_history'] = list(ctx.grasp_candidate_index_history)
            r = Result('retry', bool(ok), reason, data={'common_direction_retry': ledger})
            r.children = children
            return r
        body = guard.STATE['canonical'].coerce_steps(p.get('body'))
        if type(p.get('max_attempts')) is not int or p['max_attempts'] != 2 or tuple((s.op for s in body)) != BODY_OPS:
            return finish(False, 'common retry requires exact five primitives and max_attempts=2')
        if body[1].params.get('fresh_retry_max_attempts') != 1 or body[1].params.get('frozen_candidate_only') is not True:
            return finish(False, 'common family primitive must be one frozen physical attempt')
        f = getattr(backend, '_common_geometry_family', None)
        if not f or len(ctx.grasp_candidates) != 2 or len(f['candidates']) != 2:
            return finish(False, 'common two-candidate family unavailable')
        from selection_origin_binding import proposal
        if [proposal(g) for g in ctx.grasp_candidates] != f['proposal_snapshots']:
            return finish(False, 'context pool differs from frozen common family')
        if ctx.grasp_candidate_index_history:
            return finish(False, 'common retry may start only before its first physical attempt')
        for index in (0, 1):
            info = f['candidates'][index]
            if info['axis_choice'] != ('short' if index == 0 else 'long'):
                return finish(False, 'common axis order mismatch')
            if info['opening_limit_status'] != 'PASS_ESTIMATE':
                if info['opening_limit_status'] == 'INFEASIBLE_KNOWN_CLOUD':
                    ledger.setdefault('opening_estimate_advisories', []).append({'index': index, 'phase': 'before_attempt', 'status': info['opening_limit_status'], 'predicted_width_m': info.get('predicted_width_m'), 'required_symmetric_opening_m': info.get('required_symmetric_opening_m'), 'max_opening_m': info.get('max_opening_m'), 'veto': False})
                else:
                    ledger['skip'] = {'index': index, 'reason': info['opening_limit_status']}
                    return finish(False, 'common candidate exceeds observed opening estimate')
            ctx.grasp_candidate_index = index
            ctx.grasp = ctx.grasp_candidates[index]
            ctx._common_attempt_ordinal = index + 1
            ctx._common_family_id = f['family_id']
            for attr in ('_retention_attempt_dir', '_retention_binding', '_retention_sequence'):
                if hasattr(ctx, attr):
                    delattr(ctx, attr)
            ctx._retention_pending_lift = False
            attempt = {'ordinal': index + 1, 'candidate_index': index, 'effective_z_m': f['effective_z_m'], 'steps': []}
            ledger['attempts'].append(attempt)
            failed = None
            for step in body:
                import time
                _started = time.monotonic()
                result = self._run_step(backend, ctx, step)
                result.dt_s = time.monotonic() - _started
                if step.op == 'open_gripper':
                    result.data = dict(result.data or {}, gripper_command_m=float(backend.cfg.gripper_open_width), measured_fully_open_not_asserted=True)
                children.append(result)
                attempt['steps'].append({'op': step.op, 'ok': bool(result.ok), 'reason': str(result.reason)})
                if not result.ok:
                    failed = result
                    break
            if failed is None:
                return finish(True, 'common candidate holding passed')
            if index == 1:
                return finish(False, 'long backup exhausted')
            hold = (failed.data or {}).get('holding_check', {})
            explicit_empty = failed.op == 'assert_holding' and hold.get('reason_code') == 'MEASURED_EMPTY_CLOSED' and (hold.get('status') == 'empty_closed') and (hold.get('semantics') == 'width_hold_proxy/v1') and (hold.get('execution_binding_status') == 'MATCH') and (hold.get('p_source_eligible') is True) and (hold.get('trace_phase', 'post_lift') == 'post_lift')
            jam = failed.op == 'goto_grasp_topdown_family' and 'physical_descent_jam_requires_fresh_grounding:' in str(failed.reason)
            terminal_empty = failed.op == 'assert_holding' and terminal_empty_after_unstable(hold, now=time.time())
            if terminal_empty:
                attempt['retry_eligibility'] = {'reason': 'UNSTABLE_THEN_MEASURED_EMPTY', 'holding_status_unchanged': True, 'initial_empty_grasp_certified': False, 'P_source_eligible': False}
            if not jam and (not explicit_empty) and (not terminal_empty):
                return finish(False, 'failure does not authorize another direction')
            if f['candidates'][1]['opening_limit_status'] != 'PASS_ESTIMATE':
                info = f['candidates'][1]
                if info['opening_limit_status'] == 'INFEASIBLE_KNOWN_CLOUD':
                    ledger.setdefault('opening_estimate_advisories', []).append({'index': 1, 'phase': 'qualified_backup', 'status': info['opening_limit_status'], 'predicted_width_m': info.get('predicted_width_m'), 'required_symmetric_opening_m': info.get('required_symmetric_opening_m'), 'max_opening_m': info.get('max_opening_m'), 'veto': False})
                else:
                    ledger['skip'] = {'index': 1, 'reason': info['opening_limit_status']}
                    return finish(False, 'long backup opening estimate unavailable')
            if jam:
                name = 'retreat_after_topdown_physical_failure'
                retreat = getattr(backend, name, None)
                if not callable(retreat):
                    return finish(False, 'exit primitive unavailable: ' + name)
                motion = retreat()
                r = candidate._from_motion(name, motion)
                children.append(r)
                attempt['exit'] = {'primitive': name, 'ok': bool(r.ok), 'reason': str(r.reason)}
                if not r.ok:
                    return finish(False, 'contact exit verification failed')
            else:
                long_pose = ctx.grasp_candidates[1]
                raised = list(long_pose.position)
                raised[2] += float(body[1].params.get('z_approach', ctx.config.pre_grasp_dis))
                closed_width = getattr(ctx, 'last_close_width', None)
                if closed_width is None:
                    closed_width = getattr(backend.cfg, 'gripper_close_width', None)
                import math
                if not isinstance(closed_width, (int, float)) or isinstance(closed_width, bool) or (not math.isfinite(closed_width)) or (closed_width < 0):
                    return finish(False, 'last close command unavailable for standoff')
                motion = backend.goto_pose(raised, list(long_pose.quaternion), z_approach=0.0, verify=True, gripper_width=float(closed_width))
                r = candidate._from_motion('goto_pose', motion, data={'purpose': 'empty_grasp_long_standoff', 'contact_tcp_position': raised, 'quaternion': list(long_pose.quaternion), 'z_approach': 0.0, 'verify': True, 'gripper_command_m': float(closed_width), 'rotation_may_occur_during_path': True})
                children.append(r)
                attempt['exit'] = r.to_dict()
                if not r.ok:
                    return finish(False, 'long standoff preparation failed')
        return finish(False, 'common retry exhausted')
    cls._run_retry = candidate.SequentialPolicy._run_retry = run
    return {'mode': MODE, 'max_attempts': 2, 'class_identity_preserved': True, 'old_run_retry': old.__code__.co_filename, 'new_run_retry': run.__code__.co_filename}
