"""Connect judge events, recovery execution and episode recording."""
import asyncio, contextvars, functools, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
TEMP = ROOT / 'TEMPORAL'

def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m

def event(root, name, **fields):
    with (Path(root) / 'TEMPORAL_RECOVERY_EVENTS.jsonl').open('a') as f:
        f.write(json.dumps({'event': name, 'wall_ts': time.time(), **fields}, default=str) + '\n')
        f.flush()

def goal(jm, rec):
    pr = jm.plans.get(rec.plan_id)
    plan = getattr(pr, 'plan', None)
    for s in getattr(plan, 'subtasks', []) or []:
        if getattr(s, 'id', None) == rec.subtask_id:
            return str(s.instruction)
    return str(getattr(plan, 'goal', ''))

def bind_manager(cls, root):
    oldreg = cls.register_job
    oldrecover = cls._recover_then_finalize
    oldresult = cls.on_result

    @functools.wraps(oldreg)
    async def register(self, *a, **kw):
        result = await oldreg(self, *a, **kw)
        key = kw.get('job_key', a[0] if a else None)
        rec = self.jobs[key]
        rr = self._temporal_recorder_registry
        rr.ensure_episode(key, goal(self, rec))
        assert rr.build(key) is not None
        event(root, 'EPISODE_RECORD_ENSURED', job_key=key, judge_verdict_fabricated=False)
        return result

    @functools.wraps(oldrecover)
    async def recover(self, key):
        rec = self.jobs.get(key)
        if getattr(rec, '_temporal_executor_terminal', False):
            event(root, 'EXECUTOR_TERMINAL_NO_RECOVERY', job_key=key)
            return await self._release_then_finalize(key, return_home_override=False)
        rr = self._temporal_recorder_registry
        assert rr.build(key) is not None
        seen = getattr(self, '_temporal_recovery_started', set())
        self._temporal_recovery_started = seen
        assert not seen, 'SECOND_EPISODE_RECOVERY_HOOK_DISALLOWED'
        seen.add(key)
        event(root, 'RECOVERY_HOOK_ENTERED', job_key=key)
        try:
            return await oldrecover(self, key)
        finally:
            event(root, 'RECOVERY_HOOK_RETURNED', job_key=key, status=getattr(self.jobs.get(key), 'status', None))

    @functools.wraps(oldresult)
    async def result(self, key, payload):
        if (payload or {}).get('status') == 'protocol_recovery_request':
            event(root, 'DISABLED_PROTOCOL_REQUEST_REJECTED', job_key=key)
            return
        source = ((payload or {}).get('result') or {}).get('source')
        if (payload or {}).get('status') == 'failed' and source not in ('vlm_temporal_judge', 'temporal_judge'):
            rec = self.jobs.get(key)
            if rec is not None:
                rec._temporal_executor_terminal = True
        if (payload or {}).get('status') == 'failed' and source in ('vlm_temporal_judge', 'temporal_judge') and getattr(self, '_temporal_recovery_started', set()):
            event(root, 'RECOVERY_CAP_REFUSAL', job_key=key)
            return
        return await oldresult(self, key, payload)
    cls.register_job = register
    cls._recover_then_finalize = recover
    cls.on_result = result

def apply_intervention_cap(manager, decision, root):
    if decision.status == 'failed' and getattr(manager, '_temporal_recovery_started', set()):
        decision.result = dict(decision.result or {})
        decision.result.update(model_status_before_intervention_cap='failed', control_disposition='CONTINUE_WATCH_RECOVERY_CAP_REFUSAL', physical_recovery_authorized=False)
        event(root, 'RECOVERY_CAP_REFUSAL', model_status='failed', control_status='running', scope='episode_original_task')
        decision.status = 'running'
    return decision
_JUDGE_OWNER = contextvars.ContextVar('common_temporal_judge_owner', default=None)

def install_shutdown_guard(cls, root):
    if getattr(cls, '_common_shutdown_guard_installed', False):
        return
    oldwatch = cls._watch
    oldstop = cls.stop
    oldstart = cls.start_job

    async def drain(tasks):
        current = asyncio.current_task()
        pending = [t for t in list(tasks) if t is not current and (not t.done())]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    @functools.wraps(oldwatch)
    async def watch(self, *a, **kw):
        owner = {'tasks': set(), 'closing': False}
        token = _JUDGE_OWNER.set(owner)
        try:
            return await oldwatch(self, *a, **kw)
        finally:
            owner['closing'] = True
            await drain(owner['tasks'])
            _JUDGE_OWNER.reset(token)
            event(root, 'JUDGE_WATCH_CHILDREN_JOINED', pending=sum((not t.done() for t in owner['tasks'])))

    @functools.wraps(oldstart)
    def start(self, **kw):
        value = oldstart(self, **kw)
        if not getattr(self.client.judge, '__common_shutdown_guard__', False):
            inner = self.client.judge

            @functools.wraps(inner)
            async def judged(context):
                owner = _JUDGE_OWNER.get()
                task = asyncio.current_task()
                if getattr(self, '_common_judge_stopping', False) or (owner is not None and owner['closing']):
                    raise asyncio.CancelledError()
                all_tasks = getattr(self, '_common_judge_children', None)
                if all_tasks is None:
                    all_tasks = set()
                    self._common_judge_children = all_tasks
                all_tasks.add(task)
                if owner is not None:
                    owner['tasks'].add(task)
                try:
                    return await inner(context)
                finally:
                    all_tasks.discard(task)
                    if owner is not None:
                        owner['tasks'].discard(task)
            judged.__common_shutdown_guard__ = True
            self.client.judge = judged
        return value

    @functools.wraps(oldstop)
    async def stop(self):
        self._common_judge_stopping = True
        await drain(getattr(self, '_tasks', set()))
        await drain(getattr(self, '_common_judge_children', set()))
        event(root, 'JUDGE_CHILDREN_JOINED_BEFORE_CLIENT_CLOSE', pending=sum((not t.done() for t in getattr(self, '_common_judge_children', set()))))
        return await oldstop(self)
    cls._watch = watch
    cls.start_job = start
    cls.stop = stop
    cls._common_shutdown_guard_installed = True

def install(cal, root):
    root = Path(root)
    sys.path.insert(0, str(TEMP))
    import gripper_width as legal
    import carryon.judge.supervisor as actual
    import carryon.judge.factory as jf
    import carryon.jobs.manager as jm
    import carryon.recovery.factory as rf
    candidate = load(TEMP / 'temporal_supervisor.py', 'carryon.judge.common_temporal_candidate')
    legal.bind_judge_B(candidate, cal)
    original_config = jf._apply_checkpoint_env_overrides

    def forced(c):
        original_config(c)
        for k, v in dict(checkpoint_enabled=False, checkpoint_pause=False, checkpoint_triage=False, proprio_drop_enabled=False, stuck_force_failed_polls=0, checkpoint_release_stuck_count=0, checkpoint_emptygrasp_count=0, failure_takeover_wall_cap_s=0, max_polls=0, failure_streak_threshold=1).items():
            setattr(c, k, v)
    jf._apply_checkpoint_env_overrides = forced
    temporal = load(TEMP / 'temporal_install.py', 'temporal_install')
    bind_manager(jm.JobManager, root)
    oldfactory = rf.build_recovery

    @functools.wraps(oldfactory)
    def build(*a, **kw):
        wiring = oldfactory(*a, **kw)
        manager = a[1] if len(a) > 1 else kw['jm']
        manager._temporal_recorder_registry = wiring.recorder_registry
        temporal.attach_registry(wiring.recorder_registry)
        temporal.install(actual.VLMJudgeSupervisor)
        original_start = actual.VLMJudgeSupervisor.start_job

        def start(supervisor, **kwargs):
            value = original_start(supervisor, **kwargs)
            if not getattr(supervisor.client.judge, '__common_cap__', False):
                oldjudge = supervisor.client.judge

                async def judged(context):
                    return apply_intervention_cap(manager, await oldjudge(context), root)
                judged.__common_cap__ = True
                supervisor.client.judge = judged
            return value
        actual.VLMJudgeSupervisor.start_job = start
        install_shutdown_guard(actual.VLMJudgeSupervisor, root)
        event(root, 'TEMPORAL_ACTUAL_FACTORY_ATTACHED', record_stream_present=wiring.recorder_registry._stream is not None, protocol_direct_trigger=False)
        return wiring
    rf.build_recovery = build
    event(root, 'TEMPORAL_INSTALL_COMPLETE', streak=1, protocol_direct_trigger=False, UNSURE_forced_failure=False)
    return {'temporal': True, 'source': str(TEMP), 'protocol_direct_trigger': False, 'UNSURE_forced_failure': False}
