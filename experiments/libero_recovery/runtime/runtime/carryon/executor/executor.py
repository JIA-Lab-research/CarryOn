from __future__ import annotations
import asyncio
import json
import time
from typing import Awaitable, Callable, Optional
from ..config import Config
from ..endpoints import EndpointRegistry, RobotEndpoint
from ..errors import TransportError
from ..jobs.channels.base import ResultChannel
from ..jobs.manager import JobManager
from ..memory import Memory
from ..transports.base import Transport
from ..types import Message, Plan, Subtask, is_terminal_status
from ..judge.routing import endpoint_under_judge
from .dag import resolve_placeholders
PlanFinishedCallback = Callable[[Plan], Awaitable[None]]

class Executor:
    """Fire-and-forget event-driven DAG scheduler.

    - submit_plan: register plan, submit all root subtasks, return immediately.
    - on_job_final: advance DAG when a job reaches a terminal state.
    """

    def __init__(self, *, registry: EndpointRegistry, transports: dict[str, Transport], memory: Memory, job_manager: JobManager, result_channel: ResultChannel, config: Config, notifier=None, planner=None, reactor=None, judge_supervisor=None, on_plan_finished: Optional[PlanFinishedCallback]=None) -> None:
        self.registry = registry
        self.transports = transports
        self.memory = memory
        self.jm = job_manager
        self.rc = result_channel
        self.notifier = notifier
        self.planner = planner
        self.reactor = reactor
        self.judge_supervisor = judge_supervisor
        self.config = config
        self.on_plan_finished = on_plan_finished
        self._sem = asyncio.Semaphore(max(1, int(config.executor_max_concurrency)))
        self._plans: dict[str, Plan] = {}
        self.jm.on_recovery_resumed = self.on_recovery_resumed

    async def submit_plan(self, plan: Plan) -> None:
        self._plans[plan.id] = plan
        await self.jm.register_plan(plan)
        roots = [s for s in plan.subtasks if not s.depends_on]
        await asyncio.gather(*(self.submit_subtask(plan, s) for s in roots), return_exceptions=False)

    async def on_job_final(self, job_key: str) -> None:
        rec = self.jm.get(job_key)
        if rec is None:
            return
        plan = self._plans.get(rec.plan_id)
        if plan is None:
            return
        sub = plan.get(rec.subtask_id)
        if sub is None:
            return
        sub.status = rec.status
        sub.result = rec.result
        if rec.status == 'success':
            await self._advance_successors(plan, sub)
        elif rec.status in ('failed', 'timeout'):
            await self._handle_failure(plan, sub)
        await self.maybe_finish_plan(plan)

    async def cancel_plan(self, plan_id: str) -> int:
        return await self.jm.cancel_plan(plan_id)

    async def on_recovery_resumed(self, rec) -> None:
        """Recovery succeeded and gello resumed the same request_id.

        The original VLM judge watcher returned when it emitted the failed
        verdict that triggered recovery. Restart it so the continued VLA task is
        supervised again; executor terminal callbacks for arm endpoints remain
        intentionally ignored while Judge is in charge.
        """
        plan = self._plans.get(rec.plan_id)
        if plan is None:
            return
        sub = plan.get(rec.subtask_id)
        if sub is None:
            return
        sub.status = 'running'
        sub.result = None
        if self.judge_supervisor is None:
            return
        try:
            ep = self.registry.get(sub.endpoint)
            instruction = resolve_placeholders(sub.instruction, plan)
            inputs = resolve_placeholders(sub.inputs, plan)
        except Exception:
            return
        if not self._endpoint_under_judge(ep):
            return
        self.judge_supervisor.start_job(job_key=rec.job_key, plan=plan, subtask=sub, endpoint=ep, instruction=instruction, inputs=inputs, vla_job_id=rec.vla_job_id, deadline_s=max(1.0, rec.deadline_ts - time.time()), post_handback=True)

    def _endpoint_under_judge(self, ep: RobotEndpoint) -> bool:
        """Whether the VLM Judge should supervise this endpoint.

        Driven by ``Config.vlm_judge_applies_to_tags`` (ANY-of whitelist).
        Empty/unset list means "supervise everything" (legacy default).
        """
        return endpoint_under_judge(ep.tags, self.config.vlm_judge_applies_to_tags)

    async def submit_subtask(self, plan: Plan, sub: Subtask) -> None:
        try:
            instruction = resolve_placeholders(sub.instruction, plan)
            inputs = resolve_placeholders(sub.inputs, plan)
        except Exception as e:
            await self._mark_submit_failed(plan, sub, f'placeholder: {e}')
            return
        try:
            ep = self.registry.get(sub.endpoint)
        except Exception as e:
            await self._mark_submit_failed(plan, sub, f'endpoint lookup: {e}')
            return
        tr = self.transports.get(ep.transport)
        if tr is None:
            await self._mark_submit_failed(plan, sub, f'no transport for {ep.transport}')
            return
        job_key = f'{plan.id}:{sub.id}'
        meta = {'request_id': job_key, 'callback_url': self.rc.external_url(job_key), 'deadline_s': self.config.job_default_deadline_s}
        sub.status = 'pending'
        sub.attempts += 1
        async with self._sem:
            try:
                sr = await tr.send(ep, instruction, inputs, meta)
            except TransportError as e:
                await self._mark_submit_failed(plan, sub, f'transport: {e}')
                return
            except Exception as e:
                await self._mark_submit_failed(plan, sub, f'{type(e).__name__}: {e}')
                return
        if not sr.accepted:
            await self._mark_submit_failed(plan, sub, sr.reason or 'not accepted')
            return
        sub.job_id = sr.job_id
        await self.jm.register_job(job_key=job_key, vla_job_id=sr.job_id, plan_id=plan.id, subtask_id=sub.id, endpoint_name=ep.name, deadline_s=self.config.job_default_deadline_s, callback_url=meta['callback_url'])
        if self.notifier is not None:
            await self.notifier.info(f'[{plan.id} {sub.id} @ {ep.name}] dispatched')
        if self.judge_supervisor is not None and self._endpoint_under_judge(ep):
            self.judge_supervisor.start_job(job_key=job_key, plan=plan, subtask=sub, endpoint=ep, instruction=instruction, inputs=inputs, vla_job_id=sr.job_id, deadline_s=self.config.job_default_deadline_s)

    async def _mark_submit_failed(self, plan: Plan, sub: Subtask, reason: str) -> None:
        from ..types import ToolResult
        sub.status = 'failed'
        sub.result = ToolResult(ok=False, error=reason)
        self.memory.append(Message(role='system_event', content=f'submit failed {plan.id}:{sub.id} @ {sub.endpoint}: {reason}', plan_id=plan.id, subtask_id=sub.id, endpoint=sub.endpoint))
        if self.notifier is not None:
            await self.notifier.job_update(plan.id, sub.id, sub.endpoint, 'failed', reason)
        await self._handle_failure(plan, sub)
        await self.maybe_finish_plan(plan)

    async def _advance_successors(self, plan: Plan, sub: Subtask) -> None:
        for nxt in plan.subtasks:
            if nxt.status != 'pending':
                continue
            if sub.id not in nxt.depends_on:
                continue
            if all((plan.get(d) is not None and plan.get(d).status == 'success' for d in nxt.depends_on)):
                await self.submit_subtask(plan, nxt)

    async def _handle_failure(self, plan: Plan, sub: Subtask) -> None:
        """Staged failure ladder: local Retry, then React.

        Re-dispatch the same subtask up to ``executor_max_retries`` times
        (backoff). Once retries are exhausted, fail the downstream subgraph and
        — if ``executor_react_on_failure`` and a Reactor is wired — hand the
        failure to the Reactor so the planner VLM can decide a follow-up (an
        empty reaction Plan means "give up").
        """
        max_retries = int(self.config.executor_max_retries)
        if sub.attempts <= max_retries:
            base = float(self.config.executor_retry_backoff_base_s)
            cap = float(self.config.executor_retry_backoff_cap_s)
            delay = min(cap, base * 2 ** max(0, sub.attempts - 1))
            if delay > 0:
                await asyncio.sleep(delay)
            sub.status = 'pending'
            sub.result = None
            await self.submit_subtask(plan, sub)
            return
        self._fail_downstream(plan, sub)
        if self.config.executor_react_on_failure and self.reactor is not None:
            await self.reactor.on_subtask_failed(plan, sub)

    async def on_status_event(self, job_key: str, payload: dict) -> None:
        """Hook invoked by JobManager for non-terminal endpoint updates."""
        rec = self.jm.get(job_key)
        plan = self._plans.get(rec.plan_id) if rec is not None else None
        sub = plan.get(rec.subtask_id) if plan is not None and rec is not None else None
        if plan is None:
            self.memory.append(Message(role='system_event', content=f'status event for unknown plan/job {job_key}', metadata={'kind': 'status_event_orphan', 'payload': payload}))
            return
        if self.reactor is not None:
            await self.reactor.on_status_update(plan, sub, payload)

    def _fail_downstream(self, plan: Plan, failed: Subtask) -> None:
        from ..types import ToolResult
        changed = True
        failed_set = {failed.id}
        while changed:
            changed = False
            for s in plan.subtasks:
                if s.status != 'pending':
                    continue
                if any((d in failed_set for d in s.depends_on)):
                    s.status = 'failed'
                    s.result = ToolResult(ok=False, error=f'upstream failed: {failed.id}')
                    failed_set.add(s.id)
                    changed = True

    async def maybe_finish_plan(self, plan: Plan, *, force: bool=False) -> None:
        if not all((is_terminal_status(s.status) for s in plan.subtasks)):
            return
        if plan.id not in self._plans:
            return
        if not force and self.reactor is not None and self._reactor_busy(plan.id):
            asyncio.create_task(self._defer_finish_recheck(plan))
            return
        self._plans.pop(plan.id, None)
        pr = self.jm.plans.get(plan.id)
        if pr is not None:
            any_failed = any((s.status != 'success' for s in plan.subtasks))
            any_cancelled = any((s.status == 'cancelled' for s in plan.subtasks))
            pr.status = 'cancelled' if any_cancelled and (not any((s.status == 'success' for s in plan.subtasks))) else 'failed' if any_failed else 'completed'
            self.jm.plans.pop(plan.id, None)
        succeeded = [s for s in plan.subtasks if s.status == 'success']
        failed = [s for s in plan.subtasks if s.status != 'success']
        summary = await self._build_plan_finished_summary(plan, succeeded, failed)
        self.memory.append(Message(role='assistant', content=summary, plan_id=plan.id, metadata={'kind': 'plan_finished'}))
        if self.notifier is not None:
            await self.notifier.plan_finished(plan.id, summary)
        if self.on_plan_finished is not None:
            try:
                await self.on_plan_finished(plan)
            except Exception:
                pass

    async def _build_plan_finished_summary(self, plan: Plan, succeeded: list[Subtask], failed: list[Subtask]) -> str:
        fallback = _plan_finished_summary(plan, succeeded, failed)
        if not _needs_planner_summary(plan):
            return fallback
        if self.planner is None or not hasattr(self.planner, 'summarize_results'):
            return fallback
        if self.config.planner_vlm_provider == 'mock':
            return fallback
        try:
            answer = (await self.planner.summarize_results(plan)).strip()
        except Exception:
            return fallback
        if not answer:
            return fallback
        header = f'goal: {plan.goal} | success: {len(succeeded)}/{len(plan.subtasks)}' + (f' | failures: {[s.id for s in failed]}' if failed else '')
        return header + '\n' + _truncate(answer, 1500)

    def _reactor_busy(self, plan_id: str) -> bool:
        """Whether the Reactor still owes work for this plan (defer finalize).

        Delegates to the Reactor's public :meth:`Reactor.is_busy` rather than
        inspecting its private state (ADR-0006). ``None`` reactor => never busy.
        """
        if self.reactor is None:
            return False
        return self.reactor.is_busy(plan_id)

    async def _defer_finish_recheck(self, plan: Plan) -> None:
        """Poll the reactor settle state and re-attempt finalization."""
        for _ in range(200):
            await asyncio.sleep(0.05)
            if plan.id not in self._plans:
                return
            if not self._reactor_busy(plan.id):
                await self.maybe_finish_plan(plan)
                return
        await self.maybe_finish_plan(plan, force=True)

def _needs_planner_summary(plan: Plan) -> bool:
    for sub in plan.subtasks:
        if sub.status != 'success' or sub.result is None or (not sub.result.ok):
            continue
        data = sub.result.data
        if sub.endpoint == 'web_query':
            return True
        if isinstance(data, dict) and data.get('kind') in {'weather', 'search'}:
            return True
    return False

def _plan_finished_summary(plan: Plan, succeeded: list[Subtask], failed: list[Subtask]) -> str:
    summary = f'goal: {plan.goal} | success: {len(succeeded)}/{len(plan.subtasks)}' + (f' | failures: {[s.id for s in failed]}' if failed else '')
    result_lines = [line for sub in plan.subtasks if sub.status == 'success' for line in [_brief_subtask_result(sub)] if line]
    if result_lines:
        summary += '\n' + '\n'.join(result_lines)
    return summary

def _brief_subtask_result(sub: Subtask) -> str:
    result = sub.result
    if result is None or not result.ok or result.data is None:
        return ''
    data = result.data
    if isinstance(data, dict):
        brief = data.get('answer') or data.get('summary') or data.get('note')
        if not brief:
            brief = _compact_json(data)
    else:
        brief = str(data)
    brief = _truncate(str(brief).strip(), 1000)
    return f'- {sub.id} @ {sub.endpoint}: {brief}' if brief else ''

def _compact_json(data: object) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(data)

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)].rstrip() + '…'
