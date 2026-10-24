"""Isolated robot-width proxy. No object-specific thresholds or oracle inputs."""
import copy, math, time, hashlib, json, functools
from pathlib import Path
from width_sample_semantics import dynamic_samples, width_sample
ORACLE_KEYS = frozenset({'grasp_contact', 'contact_ok', 'all_contact_true', 'post_lift_grasp_contact', 'target_gt', 'gt_point', 'gt_mask', 'offline_gt_timeline'})
WIDTH_FIELDS = frozenset({'command_width', 'w_cmd', 'w_meas', 'width_m', 'measured_width_m', 'ts', 'producer_ts', 'measurement_source', 'measurement_valid', 'sample_count', 'delta_width_m', 'state_after', 'lift_width_trace', 'samples', 'status', 'semantics', 'p_source_eligible', 'reason_code', 'calibration_sha256', 'state_source'})

def finite(x):
    return isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(x)

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in ORACLE_KEYS}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

def widths_only(value):
    if isinstance(value, dict):
        return {k: widths_only(v) for k, v in value.items() if k in WIDTH_FIELDS}
    if isinstance(value, list):
        return [widths_only(v) for v in value]
    return value

def valid_binding(record):
    if not isinstance(record, dict) or record.get('schema') != 'carryon-executed-candidate-origin/v1':
        return False
    i = record.get('original_pool_index')
    n = record.get('original_pool_count')
    if type(i) is not int or type(n) is not int or (not 0 <= i < n <= 15) or (record.get('physical_execution_ordinal') != 1):
        return False
    for k in ('original_candidate', 'executed_candidate'):
        g = record.get(k)
        if not isinstance(g, dict):
            return False
        if not (isinstance(g.get('position'), list) and len(g['position']) == 3 and all((finite(x) for x in g['position']))):
            return False
        if not (isinstance(g.get('quaternion'), list) and len(g['quaternion']) == 4 and all((finite(x) for x in g['quaternion']))):
            return False
    return True

def execution_errors(steps):
    """Validate actual grasp binding even when the width/source verdict is unavailable."""
    errors = []
    active = None

    def visit(nodes):
        nonlocal active
        for item in nodes:
            op = getattr(item, 'op', '')
            data = getattr(item, 'data', {}) or {}
            if op in ('goto_grasp_topdown_family', 'goto_grasp_goalset', 'goto_grasp'):
                family = data.get('topdown_family')
                if 'fresh_attempts' in data:
                    attempts = data['fresh_attempts']
                    family = attempts[-1].get('topdown_family') if attempts else None
                active = family if isinstance(family, dict) else {}
            if op == 'assert_holding' and active is not None:
                hc = data.get('holding_check', {})
                hb = hc.get('executed_selection')
                expected = active.get('executed_selection')
                capture = active.get('parameter_attempt_evidence') or {}
                if not valid_binding(hb) or not valid_binding(expected):
                    errors.append('EXECUTED_SELECTION_MISSING_OR_INVALID')
                elif hb != expected or (capture and capture.get('binding') != hb) or hb['executed_candidate']['position'] != active.get('frozen_target_xyz'):
                    errors.append('EXECUTED_SELECTION_MISMATCH')
            visit(getattr(item, 'children', None) or [])
    visit(steps)
    return errors

def _calibration_valid(c, *, cpu=False):
    if not isinstance(c, dict):
        return False
    kind = c.get('kind')
    if kind == 'CPU_SYNTHETIC_CALIBRATION':
        if not cpu:
            return False
    elif kind != 'robot_empty_close_calibration/v1':
        return False
    nums = ['empty_width_m', 'closed_command_m', 'hold_margin_m', 'max_open_width_m', 'max_open_margin_m', 'lift_width_epsilon_m', 'measurement_max_age_s']
    if not all((finite(c.get(k)) for k in nums)):
        return False
    if not (0 <= c['empty_width_m'] < c['max_open_width_m'] and 0 < c['hold_margin_m'] and (0 < c['max_open_margin_m']) and (0 < c['lift_width_epsilon_m']) and (0 < c['measurement_max_age_s'])):
        return False
    if c['empty_width_m'] + c['hold_margin_m'] >= c['max_open_width_m'] - c['max_open_margin_m']:
        return False
    if kind != 'CPU_SYNTHETIC_CALIBRATION':
        if not c.get('robot_controller_identity') or c.get('task_conditioned', True):
            return False
        p = Path(c.get('source_path', ''))
        if not p.is_file():
            return False
        source = json.loads(p.read_text())
        if source.get('kind') != 'empty_close_no_object_measurement' or source.get('robot_controller_identity') != c['robot_controller_identity']:
            return False
        if source.get('used_GT_or_outcomes', True) or source.get('closed_command_m') != c['closed_command_m']:
            return False
        samples = source.get('samples', [])
        if len(samples) < 2 or not all((finite(s.get('w_meas')) and finite(s.get('producer_ts')) for s in samples)):
            return False
        if len({s['producer_ts'] for s in samples}) != len(samples):
            return False
        import statistics
        if c['empty_width_m'] != statistics.median((s['w_meas'] for s in samples)):
            return False
    return True

def calibration_valid(c, *, cpu=False):
    try:
        return _calibration_valid(c, cpu=cpu)
    except (OSError, ValueError, TypeError, KeyError):
        return False

def evaluate(state, trace, closed_command, c, *, now=None, cpu=False):
    """Pure legal-observation decision; oracle fields cannot affect any branch."""
    now = time.time() if now is None else now
    diag = {'semantics': 'width_hold_proxy/v1', 'status': 'verification_unavailable', 'p_source_eligible': False}

    def unavailable(reason):
        diag['reason_code'] = reason
        return (False, reason, diag)
    if not calibration_valid(c, cpu=cpu):
        return unavailable('CALIBRATION_UNAVAILABLE')
    diag['calibration_sha256'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    if not finite(closed_command) or closed_command != c['closed_command_m']:
        return unavailable('CLOSED_COMMAND_NOT_CALIBRATED')
    if not isinstance(state, dict):
        return unavailable('MEASURED_WIDTH_MISSING')
    w = state.get('w_meas')
    ts = state.get('ts')
    if not finite(w) or w < 0 or (not finite(ts)):
        return unavailable('MEASURED_WIDTH_MISSING_OR_INVALID')
    if not finite(now) or not 0 <= now - ts <= c['measurement_max_age_s']:
        return unavailable('MEASURED_WIDTH_STALE')
    observed = trace.get('samples', []) if isinstance(trace, dict) else []
    samples = dynamic_samples(observed)
    diag['observed_width_samples'] = copy.deepcopy(observed)
    diag['dynamic_sample_count'] = len(samples)
    if len(samples) < 2:
        return unavailable('INSUFFICIENT_PRODUCER_SAMPLES')
    times = [s.get('ts') for s in samples]
    ws = [s.get('width_m') for s in samples]
    if not all((finite(t) for t in times)) or not all((finite(x) and x >= 0 for x in ws)):
        return unavailable('TRACE_INVALID')
    if not all((a < b for a, b in zip(times, times[1:]))):
        return unavailable('DUPLICATE_OR_REVERSED_PRODUCER_TIME')
    if not 0 <= now - times[-1] <= c['measurement_max_age_s']:
        return unavailable('TRACE_STALE')
    if any((s.get('measurement_source') != 'w_meas' for s in samples)):
        return unavailable('TRACE_SOURCE_UNPROVEN')
    low = c['empty_width_m'] + c['hold_margin_m']
    high = c['max_open_width_m'] - c['max_open_margin_m']
    delta = max(ws) - min(ws)
    stable = delta < c['lift_width_epsilon_m']
    diag.update(measured_width_m=w, producer_ts=ts, lift_width_trace={'samples': [{'ts': t, 'width_m': x, 'measurement_source': 'w_meas', 'state_source': s.get('state_source')} for t, x, s in zip(times, ws, samples)], 'sample_count': len(ws), 'delta_width_m': delta})
    if not stable:
        diag.update(status='unstable', reason_code='LIFT_WIDTH_UNSTABLE')
        return (False, diag['reason_code'], diag)
    if w <= low and max(ws) <= low:
        diag.update(status='empty_closed', reason_code='MEASURED_EMPTY_CLOSED', p_source_eligible=True)
        return (False, diag['reason_code'], diag)
    if w >= high or max(ws) >= high:
        diag.update(status='open_interval', reason_code='OPEN_WIDTH_INTERVAL')
        return (False, diag['reason_code'], diag)
    if min(ws) <= low or w <= low or max((abs(w - x) for x in ws)) >= c['lift_width_epsilon_m']:
        diag.update(status='unstable', reason_code='TRACE_CROSSES_EMPTY_INTERVAL')
        return (False, diag['reason_code'], diag)
    diag.update(status='width_hold_proxy', reason_code='NONEMPTY_STABLE_WIDTH')
    return (True, '', diag)

def bind_holding_entry(guard, c, *, cpu=False):
    """Preserve cached factory class; replace only assert_holding branch."""
    if not calibration_valid(c, cpu=cpu):
        raise ValueError('CALIBRATION_UNAVAILABLE')
    cls = guard.STATE['class']
    candidate = guard.STATE['candidate']
    old = cls._run_step
    from carryon.recovery.policy import StepResult

    @functools.wraps(old)
    def step(self, backend, ctx, step):
        op = getattr(step, 'op', None)
        if op in ('goto_grasp', 'goto_grasp_goalset', 'goto_grasp_topdown_family', 'close_gripper', 'lift'):
            ctx._preplace_width_capture = None
        if op == 'goto_place':
            from selection_origin_binding import binding
            cap = {'phase': 'pre_place', 'started_wall_ts': time.time(), 'executed_selection': binding(ctx), 'motion_ok': False, 'trace': {}, 'capture_error': None}
            ctx._preplace_width_capture = cap
            previous_trace = copy.deepcopy(backend.last_lift_width_trace())
            result = None
            try:
                try:
                    backend._begin_lift_width_trace()
                except Exception as e:
                    cap['capture_error'] = repr(e)
                result = old(self, backend, ctx, step)
                cap['motion_ok'] = bool(result.ok)
                return result
            finally:
                try:
                    backend._finish_lift_width_trace()
                    cap['finished_wall_ts'] = time.time()
                    raw = copy.deepcopy(backend.last_lift_width_trace())
                    cap['raw_trace'] = raw
                    cap['trace'] = {'samples': [s for s in raw.get('samples', []) if finite(s.get('ts')) and cap['started_wall_ts'] <= s['ts'] <= cap['finished_wall_ts']]}
                except Exception as e:
                    cap['capture_error'] = repr(e)
                finally:
                    backend._lift_width_trace_active = False
                    backend._last_lift_width_trace = previous_trace
                if result is not None:
                    if result.data is None:
                        result.data = {}
                    result.data['preplace_width_capture'] = copy.deepcopy(cap)
        if op != 'assert_holding':
            return old(self, backend, ctx, step)
        state = backend._read_state()
        closed = getattr(ctx, 'last_close_width', None)
        phase = getattr(ctx, '_preplace_width_capture', None)
        trace = phase.get('trace', {}) if phase is not None else backend.last_lift_width_trace()
        from selection_origin_binding import binding as current_binding
        phase_ok = phase is None or (phase.get('motion_ok') and (not phase.get('capture_error')) and valid_binding(phase.get('executed_selection')) and (phase['executed_selection'] == current_binding(ctx)))
        if not phase_ok:
            trace = {}
        evaluated_wall_ts = time.time()
        from gripper_width import evaluate as terminal_width_evaluate
        ok, reason, diag = terminal_width_evaluate(state, trace, closed, c, now=evaluated_wall_ts, cpu=cpu)
        diag['trace_phase'] = 'pre_place' if phase is not None else 'post_lift'
        diag['phase_capture'] = copy.deepcopy(phase)
        if not phase_ok:
            diag['reason_code'] = 'PREPLACE_SOURCE_UNAVAILABLE'
            reason = diag['reason_code']
            ok = False
            diag['p_source_eligible'] = False
        diag['evaluated_wall_ts'] = evaluated_wall_ts
        diag['evaluation_clock'] = 'host wall clock, same producer clock required'
        from selection_origin_binding import binding, proposal
        actual = binding(ctx)
        diag['executed_selection'] = actual
        diag['state_after'] = {k: state.get(k) for k in ('ts', 'w_meas', 'w_cmd', 'state_source') if isinstance(state, dict) and k in state}
        consistent = valid_binding(actual) and getattr(ctx, 'grasp_candidate_index', None) == actual['original_pool_index'] and (getattr(ctx, 'grasp', None) is not None) and (proposal(ctx.grasp) == actual['executed_candidate'])
        diag['execution_binding_status'] = 'MATCH' if consistent else 'MISSING_OR_MISMATCH'
        if not consistent:
            diag['p_source_eligible'] = False
        return StepResult('assert_holding', ok, reason, data={'holding_check': diag})
    cls._run_step = candidate.SequentialPolicy._run_step = step
    return {'before_file': old.__code__.co_filename, 'after_file': step.__code__.co_filename, 'class_identity_preserved': True}

def bind_measurement_paths(backend_cls):
    """Sanitize at bridge-read entry, preserve real producer times for lift trace."""
    old_read = backend_cls._read_state

    @functools.wraps(old_read)
    def read(self, *a, **kw):
        return clean(old_read(self, *a, **kw))

    def sample(self, state):
        if not self._lift_width_trace_active:
            return
        width = state.get('w_meas')
        ts = state.get('ts')
        if not finite(width) or width < 0 or (not finite(ts)):
            return
        if self._lift_width_trace_samples and ts <= self._lift_width_trace_samples[-1]['ts']:
            return
        self._lift_width_trace_samples.append(width_sample(state))

    def finish(self):
        try:
            self._read_state(timeout_s=min(0.25, float(self.cfg.state_timeout_s)))
        finally:
            self._lift_width_trace_active = False
        samples = copy.deepcopy(self._lift_width_trace_samples)
        ws = [s['width_m'] for s in samples]
        self._last_lift_width_trace = {'schema': 'carryon-lift-width-trace/widths-v2', 'sample_count': len(samples), 'samples': samples, 'delta_width_m': max(ws) - min(ws) if ws else None}
    backend_cls._read_state = read
    backend_cls._record_lift_width_sample = sample
    backend_cls._finish_lift_width_trace = finish

def bind_source_exclusion(module):
    """Unavailable/unstable/open checks cannot be reclassified as P empty grasp."""
    original = module.failure_view

    def view(steps):
        items = copy.deepcopy(steps)
        binding_errors = execution_errors(items)

        def visit(nodes):
            for n in nodes:
                if getattr(n, 'op', None) == 'assert_holding' and (not getattr(n, 'ok', False)):
                    hc = (getattr(n, 'data', None) or {}).get('holding_check', {})
                    if binding_errors or hc.get('execution_binding_status') != 'MATCH' or hc.get('semantics') != 'width_hold_proxy/v1' or (hc.get('p_source_eligible') is not True):
                        n.op = 'holding_verification_unavailable'
                visit(getattr(n, 'children', None) or [])
        visit(items)
        out = original(items)
        out['no_oracle_binding_errors'] = binding_errors
        return out
    module.failure_view = view
    return original

def bind_bridge_recorder_paths(c):
    import carryon.recovery.bridge as bridge
    import carryon.recovery.recorder as recorder
    from carryon.recovery.schema import TrajSample

    def trajectory(state, *, quat_order='xyzw'):
        st = clean(state)
        q = list(st.get('ee_quat', [1.0, 0.0, 0.0, 0.0]))
        if quat_order == 'xyzw' and len(q) == 4:
            q = bridge.quat_xyzw_to_wxyz(q)
        measured = st.get('w_meas')
        measured = float(measured) if finite(measured) and measured >= 0 else None
        ts = st.get('ts')
        ts = float(ts) if finite(ts) else float('nan')
        cmd = st.get('w_cmd')
        cmd = float(cmd) if finite(cmd) else None
        return TrajSample(ts=ts, q=list(st.get('q', [])), ee_pos=list(st.get('ee_pos', [])), ee_quat=q, gripper_width=measured, w_meas=measured, w_cmd=cmd, release_commanded=st.get('release_commanded'))
    bridge.traj_sample_from_state = trajectory
    old_state = bridge.ZmqBridgeClient.get_state

    def state(self):
        return clean(old_state(self))
    bridge.ZmqBridgeClient.get_state = state

    def recent(self, n=240):
        out = []
        seen = set()
        for s in self.buffer.snapshot()[-max(1, int(n)):]:
            ts = getattr(s, 'ts', None)
            w = getattr(s, 'w_meas', None)
            if finite(ts) and finite(w) and (w >= 0) and (0 <= time.time() - ts <= c['measurement_max_age_s']) and (ts not in seen):
                out.append((float(ts), float(w)))
                seen.add(ts)
        return out
    recorder.EpisodeRecorderRegistry.recent_gripper_samples = recent
    return {'bridge_measurement_fallback_removed': True, 'recorder_uses_w_meas': True}

def bind_evidence_source(module):
    old = module.assemble_evidence
    module.widths_only = widths_only

    def assembled(backend, steps, attribution):
        errors = execution_errors(steps)
        if errors:
            return {'status': 'SOURCE_REJECTED', 'reason_codes': errors, 'P_source_eligible': False}
        ev = old(backend, steps, attribution)
        return ev
    module.assemble_evidence = assembled
    return old

def fresh_samples(samples, c, now=None):
    now = time.time() if now is None else now
    out = []
    last = None
    for ts, width in samples:
        if not finite(ts) or not finite(width) or width < 0 or (not 0 <= now - ts <= c['measurement_max_age_s']):
            continue
        if last is not None and ts <= last:
            continue
        last = ts
        out.append((ts, width))
    return out

def judge_B_config(c):
    c.checkpoint_enabled = False
    c.checkpoint_pause = False
    c.checkpoint_triage = False
    c.proprio_drop_enabled = False
    c.stuck_force_failed_polls = 0
    c.checkpoint_release_stuck_count = 0
    c.checkpoint_emptygrasp_count = 0
    c.failure_takeover_wall_cap_s = 0
    return c

def bind_judge_B(candidate_module, c):
    import carryon.judge.factory as factory
    import carryon.judge.supervisor as supervisor
    original = factory._apply_checkpoint_env_overrides

    def config(c):
        original(c)
        judge_B_config(c)
        c.no_oracle_measurement_max_age_s = cal['measurement_max_age_s']
    cal = c
    factory._apply_checkpoint_env_overrides = config
    supervisor.VLMJudgeSupervisor._watch_inner = candidate_module.VLMJudgeSupervisor._watch_inner
    return {'actual_factory_class_preserved': factory.VLMJudgeSupervisor is supervisor.VLMJudgeSupervisor}

def bind_judge_A(candidate_factory, c, *, cpu=False):
    """Reuse existing width transition monitor, never feed oracle contact to it."""
    if not calibration_valid(c, cpu=cpu):
        raise ValueError('CALIBRATION_UNAVAILABLE')
    import carryon.judge.factory as factory
    import carryon.judge.checkpoint as checkpoint
    original_config = candidate_factory._apply_checkpoint_env_overrides
    original_feed = checkpoint.GripperTransitionMonitor.on_sample

    def config(cfg):
        original_config(cfg)
        judge_B_config(cfg)
        cfg.no_oracle_measurement_max_age_s = c['measurement_max_age_s']
        cfg.checkpoint_enabled = True
        cfg.checkpoint_pause = True
        cfg.proprio_empty_max_m = c['empty_width_m'] + c['hold_margin_m']
        cfg.proprio_hold_min_m = cfg.proprio_empty_max_m
        cfg.proprio_hold_max_m = c['max_open_width_m'] - c['max_open_margin_m']

    def sample(self, ts, width, *, grasp_contact=None, release_commanded=None):
        if not finite(ts) or not finite(width) or width < 0 or (not 0 <= time.time() - ts <= c['measurement_max_age_s']):
            return
        last = getattr(self, '_no_oracle_last_producer_ts', None)
        if last is not None and ts <= last:
            return
        self._no_oracle_last_producer_ts = ts
        return original_feed(self, ts, width, grasp_contact=None, release_commanded=release_commanded)

    def trajectory(self, st):
        return self.on_sample(getattr(st, 'ts', None), getattr(st, 'w_meas', None), release_commanded=getattr(st, 'release_commanded', None))
    checkpoint.GripperTransitionMonitor.on_sample = sample
    checkpoint.GripperTransitionMonitor.on_traj_sample = trajectory
    candidate_factory._apply_checkpoint_env_overrides = config
    factory.build_judge_supervisor = candidate_factory.build_judge_supervisor
    return {'actual_factory_module': factory.build_judge_supervisor.__module__, 'existing_monitor': True}
