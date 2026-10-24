from __future__ import annotations
import asyncio
import time
from typing import Awaitable, Callable, Optional, cast
from ..endpoints import EndpointRegistry
from ..memory import Memory
from ..transports.base import Transport
from ..types import CALLBACK_TERMINAL_STATUSES, STATUS_EVENT_KEYWORDS, CallbackPayload, JobStatus, Message, OnResult, Plan, RecoveryHookResult, ToolResult, is_terminal_status
from .record import JobRecord, PlanRuntime
FinalCallback = Callable[[str], Awaitable[None]]
RecoveryHook = Callable[['JobRecord'], Awaitable[RecoveryHookResult]]
RecoveryResumedCallback = Callable[['JobRecord'], Awaitable[None]]

class JobManager:
    """Tracks in-flight jobs and drives the Executor via an on_final callback.

    Also routes asynchronous, non-terminal status updates (e.g. an endpoint
    reporting "oven LED turned green") to ``on_status_event`` so the
    Reactor can decide whether they warrant a follow-up dispatch.
    """

    def __init__(self, *, registry: EndpointRegistry, transports: dict[str, Transport], memory: Memory, on_final: FinalCallback, on_status_event: Optional[OnResult]=None, tick_interval_s: float=5.0, clock: Callable[[], float]=time.time, judge_release_timeout_s: float=10.0, recovery_hook: Optional[RecoveryHook]=None, on_recovery_resumed: Optional[RecoveryResumedCallback]=None) -> None:
        self.registry = registry
        self.transports = transports
        self.memory = memory
        self.on_final = on_final
        self.on_status_event = on_status_event
        self.recovery_hook = recovery_hook
        self.on_recovery_resumed = on_recovery_resumed
        self._tick_interval = tick_interval_s
        self._clock = clock
        self._judge_release_timeout_s = float(judge_release_timeout_s)
        self.jobs: dict[str, JobRecord] = {}
        self.plans: dict[str, PlanRuntime] = {}
        self._tick_task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        self._stopped.clear()
        self._tick_task = asyncio.create_task(self._tick_loop(), name='job-tick')

    async def stop(self) -> None:
        self._stopped.set()
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except (asyncio.CancelledError, Exception):
                pass
            self._tick_task = None

    async def register_plan(self, plan: Plan) -> None:
        self.plans[plan.id] = PlanRuntime(plan=plan)

    async def register_job(self, job_key: str, vla_job_id: Optional[str], plan_id: str, subtask_id: str, endpoint_name: str, deadline_s: float, callback_url: str='') -> None:
        now = self._clock()
        rec = JobRecord(job_key=job_key, vla_job_id=vla_job_id, plan_id=plan_id, subtask_id=subtask_id, endpoint_name=endpoint_name, callback_url=callback_url, status='running', submit_ts=now, deadline_ts=now + float(deadline_s))
        self.jobs[job_key] = rec
        pr = self.plans.get(plan_id)
        if pr is not None:
            pr.job_keys.add(job_key)
        self._sys_event(f'job submitted: {job_key} @ {endpoint_name}', plan_id, subtask_id)

    async def on_result(self, job_key: str, payload: CallbackPayload) -> None:
        """Single intake for every endpoint callback (see CallbackPayload)."""
        rec = self.jobs.get(job_key)
        if rec is None:
            self._sys_event(f'late callback (unknown job): {job_key}', None, None)
            return
        raw_status = str((payload or {}).get('status') or '').lower()
        terminal_flag = (payload or {}).get('terminal')
        is_status_event = terminal_flag is False or raw_status in STATUS_EVENT_KEYWORDS or (raw_status not in CALLBACK_TERMINAL_STATUSES and raw_status != '')
        if is_status_event:
            if _is_internal_ignored_executor_terminal_event(payload or {}):
                self._sys_event(f'ignored internal status event for {job_key}: executor terminal suppressed under VLM judge', rec.plan_id, rec.subtask_id)
                return
            await self._handle_status_event(rec, raw_status or 'status_update', payload or {})
            return
        if is_terminal_status(rec.status):
            ev = rec.release_event
            if ev is not None and (not ev.is_set()):
                ev.set()
                self._sys_event(f"executor released after judge close: {job_key} (reported {raw_status or 'unknown'})", rec.plan_id, rec.subtask_id)
            else:
                self._sys_event(f'late callback ignored ({rec.status}): {job_key}', rec.plan_id, rec.subtask_id)
            return
        status = cast(JobStatus, raw_status if raw_status in CALLBACK_TERMINAL_STATUSES else 'failed')
        rec.status = status
        rec.result = ToolResult(ok=status == 'success', data=payload.get('result'), error=payload.get('error'), latency_ms=(payload.get('meta') or {}).get('latency_ms'), raw=payload)
        self.memory.append(Message(role='tool_result', content=_summarize_result(rec), endpoint=rec.endpoint_name, subtask_id=rec.subtask_id, plan_id=rec.plan_id, metadata={'job_key': job_key, 'status': status}))
        meta_source = (payload.get('meta') or {}).get('source')
        if meta_source in {'vlm_judge', 'local_hold_loss_event'}:
            if self.recovery_hook is not None and status == 'failed':
                await self._recover_then_finalize(job_key)
            elif status == 'success' and rec.endpoint_name == 'franka_sim' and (self.registry.get(rec.endpoint_name).extra.get('judge_success_release') == 'complete_terminal_v1'):
                asyncio.create_task(self._complete_then_finalize(job_key))
            else:
                asyncio.create_task(self._release_then_finalize(job_key))
        else:
            asyncio.create_task(self._safe_on_final(job_key))

    async def _recover_then_finalize(self, job_key: str) -> None:
        """Try physical recovery before the default judge-driven release.

        Dispatches on the tri-valued ``RecoveryHookResult`` (see carryon.types):
        ``"resumed"`` reopens the job, ``"handled_failed"`` advances the DAG
        without a second cancel, and ``False`` falls through to the normal
        judge-driven cancel/release path.
        """
        rec = self.jobs.get(job_key)
        if rec is None:
            return
        recovery_result: RecoveryHookResult = False
        try:
            recovery_result = await self.recovery_hook(rec)
        except Exception as e:
            self._sys_event(f'recovery hook failed for {job_key}: {type(e).__name__}: {e}', rec.plan_id, rec.subtask_id)
        if recovery_result == 'resumed':
            paused_s = rec.resume_deadline(self._clock())
            if paused_s is not None:
                self._sys_event(f'job deadline resumed after recovery: {job_key} (excluded {paused_s:.1f}s checkpoint standby)', rec.plan_id, rec.subtask_id)
            rec.status = 'running'
            rec.result = None
            rec.release_event = None
            self._sys_event(f'recovery resumed {job_key}; waiting for continued VLA result', rec.plan_id, rec.subtask_id)
            if self.on_recovery_resumed is not None:
                try:
                    await self.on_recovery_resumed(rec)
                except Exception as e:
                    self._sys_event(f'on_recovery_resumed error for {job_key}: {type(e).__name__}: {e}', rec.plan_id, rec.subtask_id)
            return
        if recovery_result == 'handled_failed':
            self._sys_event(f'recovery handled {job_key}; closing terminal record without motion', rec.plan_id, rec.subtask_id)
            await self._release_then_finalize(job_key, terminal_reason=rec.terminal_reason_hint or 'normal_failure', return_home_override=False)
        else:
            await self._release_then_finalize(job_key, terminal_reason=rec.terminal_reason_hint or 'normal_failure')

    async def _complete_then_finalize(self, job_key: str) -> None:
        """Completion ACK is not an environment-success assertion."""
        rec = self.jobs.get(job_key)
        if rec is None:
            return
        ep = self.registry.get(rec.endpoint_name)
        tr = self.transports.get(ep.transport)
        complete = getattr(tr, 'complete_terminal', None)
        if not callable(complete):
            self._sys_event('complete release unavailable; no cancel fallback', rec.plan_id, rec.subtask_id)
            return
        try:
            acknowledged = await complete(ep, rec.job_key, timeout_s=self._judge_release_timeout_s)
        except Exception as exc:
            self._sys_event(f'complete release error: {type(exc).__name__}', rec.plan_id, rec.subtask_id)
            return
        if not acknowledged:
            self._sys_event('complete release unacknowledged; no DAG advance or cancel fallback', rec.plan_id, rec.subtask_id)
            return
        rec.ensure_release_event().set()
        await self._safe_on_final(job_key)

    async def _release_then_finalize(self, job_key: str, *, terminal_reason: str='cancel', return_home_override: Optional[bool]=None) -> None:
        """Close executor state, wait for its ack, then advance the DAG.

        ``return_home_override=False`` is the record-only recovery-failure path:
        the arm remains in its fail-safe pose while the simulator lands a complete
        terminal record.
        """
        rec = self.jobs.get(job_key)
        if rec is None:
            return
        ev = rec.ensure_release_event()
        return_home = True
        try:
            ep = self.registry.get(rec.endpoint_name)
            endpoint_return_home = bool(getattr(ep, 'return_home_on_cancel', True))
            return_home = endpoint_return_home if return_home_override is None else bool(return_home_override)
            tr = self.transports.get(ep.transport)
            if tr is not None:
                try:
                    try:
                        await tr.cancel(ep, rec.job_key, return_home=return_home, terminal_reason=terminal_reason)
                    except TypeError as e:
                        if 'terminal_reason' not in str(e):
                            raise
                        await tr.cancel(ep, rec.job_key, return_home=return_home)
                except Exception as e:
                    self._sys_event(f'judge-driven cancel failed for {job_key}: {type(e).__name__}: {e}', rec.plan_id, rec.subtask_id)
        except Exception:
            pass
        try:
            await asyncio.wait_for(ev.wait(), timeout=self._judge_release_timeout_s)
        except asyncio.TimeoutError:
            self._sys_event(f'executor release timeout for {job_key} after {self._judge_release_timeout_s}s; advancing anyway (return_home={return_home}, terminal_reason={terminal_reason})', rec.plan_id, rec.subtask_id)
        await self._safe_on_final(job_key)

    async def _handle_status_event(self, rec: JobRecord, status: str, payload: dict) -> None:
        """Route a non-terminal endpoint-pushed update to the Reactor."""
        detail = payload.get('detail') or payload.get('note') or (payload.get('result') if isinstance(payload.get('result'), str) else None) or ''
        if not detail and isinstance(payload.get('result'), dict):
            d = payload['result']
            detail = d.get('summary') or d.get('note') or ''
        self.memory.append(Message(role='system_event', content=f'endpoint status update on {rec.job_key} @ {rec.endpoint_name}: status={status} {str(detail)[:200]}', endpoint=rec.endpoint_name, subtask_id=rec.subtask_id, plan_id=rec.plan_id, metadata={'job_key': rec.job_key, 'kind': 'status_event', 'status': status, 'payload': payload}))
        if self.on_status_event is not None:
            asyncio.create_task(self._safe_on_status_event(rec.job_key, payload))

    async def _safe_on_status_event(self, job_key: str, payload: CallbackPayload) -> None:
        if self.on_status_event is None:
            return
        try:
            await self.on_status_event(job_key, payload)
        except Exception as e:
            self._sys_event(f'on_status_event error for {job_key}: {type(e).__name__}: {e}', None, None)

    async def _safe_on_final(self, job_key: str) -> None:
        try:
            await self.on_final(job_key)
        except Exception as e:
            self._sys_event(f'on_final error for {job_key}: {type(e).__name__}: {e}', None, None)

    async def cancel_plan(self, plan_id: str) -> int:
        pr = self.plans.get(plan_id)
        if pr is None:
            return 0
        n = 0
        for jk in list(pr.job_keys):
            rec = self.jobs.get(jk)
            if rec is None or is_terminal_status(rec.status):
                continue
            await self._cancel_record(rec)
            n += 1
        pr.status = 'cancelled'
        return n

    async def cancel_job(self, job_key: str) -> bool:
        rec = self.jobs.get(job_key)
        if rec is None or is_terminal_status(rec.status):
            return False
        await self._cancel_record(rec)
        return True

    async def _cancel_record(self, rec: JobRecord) -> None:
        return_home = True
        try:
            ep = self.registry.get(rec.endpoint_name)
            endpoint_return_home = bool(getattr(ep, 'return_home_on_cancel', True))
            return_home = endpoint_return_home
            tr = self.transports.get(ep.transport)
            if tr is not None:
                await tr.cancel(ep, rec.job_key, return_home=return_home)
        except Exception:
            pass
        rec.status = 'cancelled'
        ev = rec.release_event
        if ev is not None and (not ev.is_set()):
            ev.set()
        self._sys_event(f'job cancelled: {rec.job_key} (return_home={return_home})', rec.plan_id, rec.subtask_id)
        asyncio.create_task(self._safe_on_final(rec.job_key))

    def get(self, job_key: str) -> Optional[JobRecord]:
        return self.jobs.get(job_key)

    def list_jobs(self) -> list[JobRecord]:
        return list(self.jobs.values())

    def list_plan_jobs(self, plan_id: str) -> list[JobRecord]:
        pr = self.plans.get(plan_id)
        if pr is None:
            return []
        return [self.jobs[jk] for jk in pr.job_keys if jk in self.jobs]

    def plan_all_terminal(self, plan_id: str) -> bool:
        pr = self.plans.get(plan_id)
        if pr is None or not pr.job_keys:
            return False
        return all((is_terminal_status(self.jobs[jk].status) for jk in pr.job_keys if jk in self.jobs))

    async def _tick_loop(self) -> None:
        while not self._stopped.is_set():
            try:
                await self._tick_once()
            except Exception as e:
                self._sys_event(f'tick error: {type(e).__name__}: {e}', None, None)
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._tick_interval)
            except asyncio.TimeoutError:
                pass

    async def _tick_once(self) -> None:
        now = self._clock()
        for rec in list(self.jobs.values()):
            if rec.status in ('pending', 'running') and rec.deadline_pause_ts is None and (now > rec.deadline_ts):
                rec.status = 'timeout'
                self._sys_event(f'job timeout: {rec.job_key}', rec.plan_id, rec.subtask_id)
                asyncio.create_task(self._safe_on_final(rec.job_key))

    def _sys_event(self, content: str, plan_id, subtask_id) -> None:
        self.memory.append(Message(role='system_event', content=content, plan_id=plan_id, subtask_id=subtask_id))

def _summarize_result(rec: JobRecord) -> str:
    r = rec.result
    if r is None:
        return f'{rec.job_key}: no result'
    if r.ok:
        data = r.data
        if isinstance(data, dict):
            brief = data.get('summary') or data.get('note') or str(data)[:120]
        else:
            brief = str(data)[:120]
        return f'{rec.job_key} success: {brief}'
    return f"{rec.job_key} {rec.status}: {r.error or ''}"

def _is_internal_ignored_executor_terminal_event(payload: dict) -> bool:
    meta = payload.get('meta')
    if not isinstance(meta, dict):
        return False
    return meta.get('source') == 'vlm_judge_intake' and meta.get('kind') == 'ignored_executor_terminal'
