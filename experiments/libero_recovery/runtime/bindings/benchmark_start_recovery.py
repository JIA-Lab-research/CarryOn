"""Start recovery and recording for the benchmark episode."""
from __future__ import annotations
from runtime_paths import bound_path, resource
import asyncio, functools, json, os, time
from pathlib import Path
STATUS = 'protocol_recovery_request'
SOURCE = 'benchmark_harness'
REASON = 'BENCHMARK_START_RECOVERY'
VERSION = 'carryon-benchmark-protocol-recovery/v1'
STATE = {'factory_bound': False}

def _event(v):
    p = Path(os.environ['CARRYON_RECOVERY_ROW_ROOT']).resolve() / 'PROTOCOL_RECOVERY_EVENTS.jsonl'
    with p.open('a') as f:
        f.write(json.dumps(v, sort_keys=True, default=str) + '\n')
        f.flush()
        os.fsync(f.fileno())

def _matches(p):
    m = p.get('meta') or {}
    r = p.get('result') or {}
    return str(p.get('status') or '').lower() == STATUS and p.get('terminal') is False and (m.get('source') == SOURCE) and (m.get('protocol_version') == VERSION) and (r.get('reason') == REASON)

def _goal(jm, rec):
    try:
        plan = getattr(jm.plans.get(rec.plan_id), 'plan', None)
        for s in list(getattr(plan, 'subtasks', None) or []):
            if getattr(s, 'id', None) == rec.subtask_id:
                x = str(getattr(s, 'instruction', '') or '').strip()
                if x:
                    return x
        return str(getattr(plan, 'goal', '') or '').strip()
    except Exception:
        return ''

def _bind_factory(cls):
    if STATE['factory_bound']:
        return
    import carryon.recovery.factory as F
    old = F.build_recovery

    @functools.wraps(old)
    def build(*a, **k):
        w = old(*a, **k)
        jm = a[1] if len(a) > 1 else k.get('jm')
        rr = getattr(w, 'recorder_registry', None)
        if jm is None or not isinstance(jm, cls):
            raise RuntimeError('real JobManager absent')
        if rr is None or not hasattr(rr, 'ensure_episode'):
            raise RuntimeError('EpisodeRecorderRegistry absent')
        jm._benchmark_protocol_recorder_registry = rr
        return w
    build.__protocol_recorder_capture_v2__ = True
    build.__wrapped_original__ = old
    F.build_recovery = build
    STATE['factory_bound'] = True

def _summary(rr, key):
    b = rr.build(key)
    inv = getattr(b, 'recovery', None) if b is not None else None
    steps = list(getattr(inv, 'steps', None) or []) if inv is not None else []
    names = [str(s.get('name') or s.get('op') or s.get('step') or '' if isinstance(s, dict) else getattr(s, 'name', None) or getattr(s, 'op', None) or '') for s in steps]
    return {'recovery_invocation_present': inv is not None, 'recovery_step_count': len(steps), 'recovery_step_names': names}

def install(cls):
    _bind_factory(cls)
    oldreg = cls.register_job
    oldres = cls.on_result
    if getattr(oldres, '__protocol_recovery_v2__', False):
        raise RuntimeError('protocol recovery v2 already installed')

    @functools.wraps(oldreg)
    async def register_job(self, job_key, vla_job_id, plan_id, subtask_id, endpoint_name, deadline_s, callback_url=''):
        await oldreg(self, job_key, vla_job_id, plan_id, subtask_id, endpoint_name, deadline_s, callback_url)
        if endpoint_name != 'franka_sim':
            return
        rr = getattr(self, '_benchmark_protocol_recorder_registry', None)
        rec = self.jobs.get(job_key)
        if rr is None or rec is None:
            _event({'event': 'EPISODE_RECORD_CREATE_REJECTED', 'job_key': job_key, 'wall_ts': time.time()})
            return
        before = rr.get(job_key) if hasattr(rr, 'get') else None
        ep = rr.ensure_episode(job_key, _goal(self, rec))
        _event({'event': 'EPISODE_RECORD_ENSURED', 'job_key': job_key, 'created': before is None, 'episode_id': getattr(ep, 'episode_id', None), 'goal_nonempty': bool(getattr(ep, 'goal', '')), 'judge_verdict_fabricated': False, 'parameter_source_eligible': False, 'wall_ts': time.time()})

    @functools.wraps(oldres)
    async def on_result(self, job_key, payload):
        if not _matches(payload or {}):
            return await oldres(self, job_key, payload)
        rec = self.jobs.get(job_key)
        if rec is None:
            self._sys_event('protocol recovery rejected: unknown job', None, None)
            return
        if rec.endpoint_name != 'franka_sim' or rec.status != 'running':
            self._sys_event('protocol recovery rejected: inactive/non-sim job', rec.plan_id, rec.subtask_id)
            return
        if self.recovery_hook is None:
            self._sys_event('protocol recovery rejected: hook absent', rec.plan_id, rec.subtask_id)
            return
        rr = getattr(self, '_benchmark_protocol_recorder_registry', None)
        if rr is None:
            raise RuntimeError('protocol recovery registry absent at request')
        l = getattr(self, '_benchmark_protocol_recovery_latches', None)
        if l is None:
            l = self._benchmark_protocol_recovery_latches = {}
        if job_key in l:
            self._sys_event('duplicate protocol recovery ignored', rec.plan_id, rec.subtask_id)
            _event({'event': 'DUPLICATE_PROTOCOL_REQUEST_IGNORED', 'job_key': job_key, 'wall_ts': time.time()})
            return
        r = payload.get('result') or {}
        ts = time.time()
        l[job_key] = {'requested_wall_ts': ts, 'simulation_step_id': r.get('simulation_step_id'), 'disturbance_kind': r.get('disturbance_kind'), 'reason': REASON}
        ep = rr.ensure_episode(job_key, _goal(self, rec))
        anchor_error = None
        try:
            ep.first_fail_ts = ts
            if hasattr(ep, '_pin_prefail_window'):
                ep._pin_prefail_window(ts)
        except Exception as e:
            anchor_error = f'{type(e).__name__}: {e}'
        built = rr.build(job_key)
        n = len(getattr(built, 'trajectory', None) or []) if built is not None else 0
        if built is None or n <= 0:
            raise RuntimeError('protocol EpisodeRecord has no real trajectory window')
        rec.recovery_trigger_source = 'benchmark_protocol_start'
        rec.recovery_trigger_is_judge_failure = False
        rec.recovery_trigger_is_parameter_source = False
        rec.recovery_trigger_payload = dict(l[job_key])
        _event({'event': 'RECOVERY_HOOK_ENTERED', 'job_key': job_key, 'wall_ts': time.time(), 'episode_id': getattr(ep, 'episode_id', None), 'trajectory_samples': n, 'protocol_anchor_ts': ts, 'anchor_is_judge_verdict': False, 'anchor_error': anchor_error, 'parameter_source_eligible': False})
        self._sys_event('protocol recovery accepted', rec.plan_id, rec.subtask_id)
        await self._recover_then_finalize(job_key)
        h = getattr(rec, 'recovery_terminal_handback_event', None)
        _event({'event': 'RECOVERY_HOOK_RETURNED', 'job_key': job_key, 'wall_ts': time.time(), **_summary(rr, job_key), 'takeover_ack': (h or {}).get('takeover_ack') if isinstance(h, dict) else None, 'handback_ack': (h or {}).get('ok') if isinstance(h, dict) else None, 'parameter_source_eligible': False})
    register_job.__protocol_episode_record_v2__ = True
    register_job.__wrapped_original__ = oldreg
    on_result.__protocol_recovery_v2__ = True
    on_result.__wrapped_original__ = oldres
    cls.register_job = register_job
    cls.on_result = on_result
    return (oldres, on_result)
_base_install = install
MIN_SAMPLES = 25
MIN_SPAN_S = 1.5
MAX_WAIT_S = 6.0

async def _start_when_ready(jm, key):
    begin = time.monotonic()
    try:
        while True:
            rec = jm.jobs.get(key)
            if rec is None or rec.status != 'running' or jm._stopped.is_set():
                _event({'event': 'START_CANCELLED_BEFORE_REQUEST', 'job_key': key, 'wall_ts': time.time()})
                return
            rr = getattr(jm, '_benchmark_protocol_recorder_registry', None)
            built = rr.build(key) if rr else None
            samples = list(getattr(built, 'trajectory', None) or [])
            ts = [float(x.ts) for x in samples]
            span = max(ts) - min(ts) if ts else 0.0
            if len(ts) >= MIN_SAMPLES and span >= MIN_SPAN_S:
                _event({'event': 'START_READY', 'job_key': key, 'trajectory_samples': len(ts), 'sample_span_s': span, 'wall_ts': time.time(), 'reason': REASON})
                await jm.on_result(key, {'status': STATUS, 'terminal': False, 'meta': {'source': SOURCE, 'protocol_version': VERSION}, 'result': {'reason': REASON}})
                return
            if time.monotonic() - begin >= MAX_WAIT_S:
                raise RuntimeError('ENGINEERING_FAIL_RECORDER_WINDOW')
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        _event({'event': 'START_TASK_CANCELLED', 'job_key': key, 'wall_ts': time.time()})
        raise
    except Exception as exc:
        _event({'event': 'START_ENGINEERING_FAILED', 'job_key': key, 'error': str(exc), 'wall_ts': time.time()})
        jm._sys_event('start recovery engineering failure: ' + str(exc), None, None)
        await jm.cancel_job(key)

def install(cls):
    _base_install(cls)
    oldreg = cls.register_job
    oldstop = cls.stop

    @functools.wraps(oldreg)
    async def register(self, *args, **kwargs):
        key = args[0] if args else kwargs['job_key']
        if key in self.jobs:
            raise RuntimeError('duplicate start registration')
        await oldreg(self, *args, **kwargs)
        rec = self.jobs.get(key)
        if rec and rec.endpoint_name == 'franka_sim':
            tasks = getattr(self, '_start_recovery_tasks', None)
            if tasks is None:
                tasks = self._start_recovery_tasks = {}
            tasks[key] = asyncio.create_task(_start_when_ready(self, key), name='benchmark-start-recovery')

    @functools.wraps(oldstop)
    async def stop(self):
        await oldstop(self)
        tasks = list(getattr(self, '_start_recovery_tasks', {}).values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    cls.register_job = register
    cls.stop = stop
    return {'status': 'START_PROTOCOL_INSTALLED', 'judge_constructed': False, 'model_calls_max': 0, 'reason': REASON, 'min_samples': MIN_SAMPLES, 'min_span_s': MIN_SPAN_S, 'max_wait_s': MAX_WAIT_S}
