from __future__ import annotations

import asyncio
import json
import sys
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

from .dag import (
    arm_subtasks_total_order_ok,
    is_single_slot_endpoint,
    resolve_placeholders,
)


PlanFinishedCallback = Callable[[Plan], Awaitable[None]]


def _resolved_objective(sub: Subtask, plan: Plan) -> str:
    """Placeholder-resolved :attr:`Subtask.objective` for the VLM Judge.

    Resolved exactly like ``instruction`` so both describe the same concrete
    objects. Fail-open by design: the objective only sharpens the Judge's
    criterion (it falls back to the instruction), so an unresolvable
    placeholder must never take down a dispatch that is otherwise fine.
    """
    if not (sub.objective or "").strip():
        return ""
    try:
        return str(resolve_placeholders(sub.objective, plan))
    except Exception:  # noqa: BLE001 - see fail-open rationale above
        return ""


class Executor:
    """Fire-and-forget event-driven DAG scheduler.

    - submit_plan: register plan, submit all root subtasks, return immediately.
    - on_job_final: advance DAG when a job reaches a terminal state.
    """

    def __init__(
        self,
        *,
        registry: EndpointRegistry,
        transports: dict[str, Transport],
        memory: Memory,
        job_manager: JobManager,
        result_channel: ResultChannel,
        config: Config,
        notifier=None,      # Notifier | None (kept loose to avoid import cycles)
        planner=None,       # Planner | None, used for plan-finished result summaries
        reactor=None,       # Reactor | None, drives planner VLM-in-the-loop reactions
        judge_supervisor=None,  # VLMJudgeSupervisor | None
        on_plan_finished: Optional[PlanFinishedCallback] = None,
    ) -> None:
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
        # Resource single-flight: a single-
        # slot physical endpoint (an arm) runs one job at a time. The slot is
        # held from ACCEPT until on_job_final (WHOLE job lifetime, not just
        # transport.send); extra ready same-endpoint subtasks wait in an
        # endpoint-keyed FIFO instead of being sent/failed. Different
        # endpoints stay parallel -- this is NOT a global concurrency=1.
        self._ep_active: dict[str, str] = {}      # endpoint name -> job_key holding the slot
        self._ep_queue: dict[str, list] = {}       # endpoint name -> FIFO [(plan_id, subtask_id)]
        self._plans: dict[str, Plan] = {}  # plan_id -> plan (owned by Executor)
        # plan_id -> force-finalize record. Kept AFTER the plan is dropped from
        # _plans, because its whole purpose is to explain work that arrives once
        # the plan is gone.
        self._force_finalized: dict[str, dict] = {}
        # Futures resolved when a plan reaches a terminal finish (see wait_plan).
        self._plan_done: dict[str, asyncio.Future] = {}
        # JobManager owns the recovery state machine; Executor owns enough plan
        # context to restart VLM-judge supervision after recovery successfully
        # hands control back to the same request_id/callback_url.
        self.jm.on_recovery_resumed = self.on_recovery_resumed

    # ---- public ----

    async def submit_plan(self, plan: Plan) -> None:
        loop = asyncio.get_running_loop()
        self._plan_done[plan.id] = loop.create_future()
        self._plans[plan.id] = plan
        await self.jm.register_plan(plan)
        if not arm_subtasks_total_order_ok(plan, self.registry):
            self.memory.append(Message(
                role="system_event",
                content=(f"plan {plan.id}: same-arm subtasks are not a total order "
                         "(fan-out/parallel on a single-slot endpoint); the executor "
                         "will serialize them via the endpoint FIFO"),
                plan_id=plan.id,
            ))
        roots = [s for s in plan.subtasks if not s.depends_on]
        if not roots:
            # Chat-only / reaction-noop: no dispatchable roots. Finalize so
            # on_plan_finished (UserSummarizer) still runs.
            await self.maybe_finish_plan(plan)
            return
        await asyncio.gather(
            *(self.submit_subtask(plan, s) for s in roots),
            return_exceptions=False,
        )

    async def wait_plan(self, plan_id: str) -> None:
        """Block until ``plan_id`` has been finalized (or was never tracked)."""
        fut = self._plan_done.get(plan_id)
        if fut is None:
            return
        await fut

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

        # sync subtask state with job record
        sub.status = rec.status
        sub.result = rec.result

        # Whole-lifetime slot release: only now is the arm actually free.
        self._release_endpoint_slot(rec.endpoint_name, job_key)
        if rec.status == "success":
            await self._advance_successors(plan, sub)
        elif rec.status in ("failed", "timeout"):
            await self._handle_failure(plan, sub)
        # cancelled: slot released above; plan finish check handles the rest.

        # Dispatch the next queued subtask for this endpoint (if slot free).
        await self._pump_endpoint(rec.endpoint_name)
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
        sub.status = "running"
        sub.result = None
        if self.judge_supervisor is None:
            return
        try:
            ep = self.registry.get(resolve_placeholders(sub.endpoint, plan))
            instruction = resolve_placeholders(sub.instruction, plan)
            inputs = resolve_placeholders(sub.inputs, plan)
            objective = _resolved_objective(sub, plan)
        except Exception:
            return
        if not self._endpoint_under_judge(ep):
            return
        self.judge_supervisor.start_job(
            job_key=rec.job_key,
            plan=plan,
            subtask=sub,
            endpoint=ep,
            instruction=instruction,
            inputs=inputs,
            vla_job_id=rec.vla_job_id,
            deadline_s=max(1.0, rec.deadline_ts - time.time()),
            objective=objective,
        )

    def _endpoint_under_judge(self, ep: RobotEndpoint) -> bool:
        """Whether the VLM Judge should supervise this endpoint.

        Driven by ``Config.vlm_judge_applies_to_tags`` (ANY-of whitelist).
        Empty/unset list means "supervise everything" (legacy default).
        """
        return endpoint_under_judge(
            ep.tags,
            self.config.vlm_judge_applies_to_tags,
        )

    # ---- collaboration surface (also used by the Reactor; see ADR-0006) ----

    def _release_endpoint_slot(self, ep_name: str, job_key: str) -> None:
        if self._ep_active.get(ep_name) == job_key:
            del self._ep_active[ep_name]

    def _enqueue_for_endpoint(self, ep_name: str, plan_id: str, sub_id: str) -> None:
        q = self._ep_queue.setdefault(ep_name, [])
        if (plan_id, sub_id) not in q:
            q.append((plan_id, sub_id))

    async def _fail_submit_release(self, plan, sub, ep_name, job_key, reason) -> None:
        # A send genuinely failed (not a resource wait): release the reserved
        # slot, mark failed (which may retry -> re-reserve), then pump the queue.
        self._release_endpoint_slot(ep_name, job_key)
        await self._mark_submit_failed(plan, sub, reason)
        await self._pump_endpoint(ep_name)

    async def _pump_endpoint(self, ep_name: str) -> None:
        # Dispatch the next eligible queued subtask, IFF the slot is free. Skips
        # queued items whose plan is gone/force-finalized or subtask is no longer
        # pending (no ghost dispatch).
        if self._ep_active.get(ep_name) is not None:
            return
        q = self._ep_queue.get(ep_name)
        while q:
            plan_id, sub_id = q.pop(0)
            if plan_id in self._force_finalized:
                continue
            plan = self._plans.get(plan_id)
            if plan is None:
                continue
            sub = plan.get(sub_id)
            if sub is None or sub.status != "pending":
                continue
            await self.submit_subtask(plan, sub)
            return

    async def submit_subtask(self, plan: Plan, sub: Subtask) -> None:
        # Resolve placeholders against in-plan completed results. The endpoint
        # field itself may be a ${tN.result...} placeholder so a later subtask
        # can run on an endpoint *discovered* by an earlier one; plain endpoint
        # names pass through resolve_placeholders unchanged.
        try:
            instruction = resolve_placeholders(sub.instruction, plan)
            inputs = resolve_placeholders(sub.inputs, plan)
            endpoint_name = resolve_placeholders(sub.endpoint, plan)
        except Exception as e:
            await self._mark_submit_failed(plan, sub, f"placeholder: {e}")
            return

        try:
            ep = self.registry.get(endpoint_name)
        except Exception as e:
            await self._mark_submit_failed(plan, sub, f"endpoint lookup: {e}")
            return

        tr = self.transports.get(ep.transport)
        if tr is None:
            await self._mark_submit_failed(plan, sub, f"no transport for {ep.transport}")
            return

        job_key = f"{plan.id}:{sub.id}"
        # Resource single-flight: if this endpoint is single-slot and its slot
        # is busy, hold this subtask pending in the endpoint FIFO -- do NOT
        # send, do NOT count a retry, do NOT fail it. It dispatches when the
        # slot releases (on_job_final). Otherwise reserve the slot now.
        if is_single_slot_endpoint(ep):
            active = self._ep_active.get(ep.name)
            if active is not None and active != job_key:
                self._enqueue_for_endpoint(ep.name, plan.id, sub.id)
                sub.status = "pending"
                return
            self._ep_active[ep.name] = job_key
        meta = {
            "request_id": job_key,
            "callback_url": self.rc.external_url(job_key),
            "deadline_s": self.config.job_default_deadline_s,
        }

        sub.status = "pending"
        sub.attempts += 1
        async with self._sem:
            try:
                sr = await tr.send(ep, instruction, inputs, meta)
            except TransportError as e:
                await self._fail_submit_release(plan, sub, ep.name, job_key, f"transport: {e}")
                return
            except Exception as e:
                await self._fail_submit_release(plan, sub, ep.name, job_key, f"{type(e).__name__}: {e}")
                return

        if not sr.accepted:
            await self._fail_submit_release(plan, sub, ep.name, job_key, sr.reason or "not accepted")
            return

        sub.job_id = sr.job_id
        await self.jm.register_job(
            job_key=job_key,
            vla_job_id=sr.job_id,
            plan_id=plan.id,
            subtask_id=sub.id,
            endpoint_name=ep.name,
            deadline_s=self.config.job_default_deadline_s,
            callback_url=meta["callback_url"],
        )
        if self.notifier is not None:
            await self.notifier.info(
                f"[{plan.id} {sub.id} @ {ep.name}] dispatched"
            )
        if self.judge_supervisor is not None and self._endpoint_under_judge(ep):
            self.judge_supervisor.start_job(
                job_key=job_key,
                plan=plan,
                subtask=sub,
                endpoint=ep,
                instruction=instruction,
                inputs=inputs,
                vla_job_id=sr.job_id,
                deadline_s=self.config.job_default_deadline_s,
                # Judge-only; deliberately NOT part of the transport payload
                # above, which already went out with the untouched instruction.
                objective=_resolved_objective(sub, plan),
            )

    async def _mark_submit_failed(self, plan: Plan, sub: Subtask, reason: str) -> None:
        from ..types import ToolResult
        sub.status = "failed"
        sub.result = ToolResult(ok=False, error=reason)
        self.memory.append(
            Message(
                role="system_event",
                content=f"submit failed {plan.id}:{sub.id} @ {sub.endpoint}: {reason}",
                plan_id=plan.id,
                subtask_id=sub.id,
                endpoint=sub.endpoint,
            )
        )
        if self.notifier is not None:
            await self.notifier.job_update(plan.id, sub.id, sub.endpoint, "failed", reason)
        await self._handle_failure(plan, sub)
        await self.maybe_finish_plan(plan)

    async def _advance_successors(self, plan: Plan, sub: Subtask) -> None:
        # find all subtasks that depend on this one; submit any whose deps are all success
        for nxt in plan.subtasks:
            if nxt.status != "pending":
                continue
            if sub.id not in nxt.depends_on:
                continue
            if all(
                (plan.get(d) is not None and plan.get(d).status == "success")
                for d in nxt.depends_on
            ):
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
            # exponential backoff from config
            base = float(self.config.executor_retry_backoff_base_s)
            cap = float(self.config.executor_retry_backoff_cap_s)
            delay = min(cap, base * (2 ** max(0, sub.attempts - 1)))
            if delay > 0:
                await asyncio.sleep(delay)
            sub.status = "pending"
            sub.result = None
            await self.submit_subtask(plan, sub)
            return

        # retries exhausted: mark downstream first so the plan-finish check
        # doesn't fire prematurely while we wait on the planner VLM.
        self._fail_downstream(plan, sub)

        if self.config.executor_react_on_failure and self.reactor is not None:
            await self.reactor.on_subtask_failed(plan, sub)

    async def on_status_event(self, job_key: str, payload: dict) -> None:
        """Hook invoked by JobManager for non-terminal endpoint updates."""
        # find the owning plan/subtask, if any
        rec = self.jm.get(job_key)
        plan = self._plans.get(rec.plan_id) if rec is not None else None
        sub = plan.get(rec.subtask_id) if (plan is not None and rec is not None) else None
        if plan is None:
            # status event for an unknown / already-finished plan; record but
            # do nothing actionable.
            self.memory.append(
                Message(
                    role="system_event",
                    content=f"status event for unknown plan/job {job_key}",
                    metadata={"kind": "status_event_orphan", "payload": payload},
                )
            )
            return
        if self.reactor is not None:
            await self.reactor.on_status_update(plan, sub, payload)

    def _fail_downstream(self, plan: Plan, failed: Subtask) -> None:
        from ..types import ToolResult
        # propagate failure to anything depending on `failed`
        changed = True
        failed_set = {failed.id}
        while changed:
            changed = False
            for s in plan.subtasks:
                if s.status != "pending":
                    continue
                if any(d in failed_set for d in s.depends_on):
                    s.status = "failed"
                    s.result = ToolResult(
                        ok=False, error=f"upstream failed: {failed.id}"
                    )
                    failed_set.add(s.id)
                    changed = True

    async def maybe_finish_plan(self, plan: Plan, *, force: bool = False) -> None:
        # all subtasks must be in a terminal state (success/failed/cancelled/timeout)
        if not all(is_terminal_status(s.status) for s in plan.subtasks):
            return
        # idempotence: remove plan from tracking
        if plan.id not in self._plans:
            return

        # If the Reactor is still chewing on events for this plan -- either a
        # debounce window is open, a planner VLM call is in flight, or events are
        # buffered -- defer finishing. The reactor may be about to append more
        # subtasks (e.g. status updates that still warrant a follow-up). ``force``
        # bypasses this when a bounded watchdog gives up waiting on the reactor.
        if not force and self.reactor is not None and self._reactor_busy(plan.id):
            # re-check shortly: someone has to drive maybe_finish_plan again
            # once the reactor settles. Schedule a tiny watchdog.
            asyncio.create_task(self._defer_finish_recheck(plan))
            return

        self._plans.pop(plan.id, None)

        # let JobManager reflect the plan is no longer in-flight, so callers
        # like AgentLoop.stop() can wait on jm.plans draining.
        pr = self.jm.plans.get(plan.id)
        if pr is not None:
            any_failed = any(s.status != "success" for s in plan.subtasks)
            any_cancelled = any(s.status == "cancelled" for s in plan.subtasks)
            pr.status = (
                "cancelled" if any_cancelled and not any(s.status == "success" for s in plan.subtasks)
                else "failed" if any_failed
                else "completed"
            )
            self.jm.plans.pop(plan.id, None)

        succeeded = [s for s in plan.subtasks if s.status == "success"]
        failed = [s for s in plan.subtasks if s.status != "success"]
        summary = await self._build_plan_finished_summary(plan, succeeded, failed)
        self.memory.append(
            Message(
                role="assistant",
                content=summary,
                plan_id=plan.id,
                metadata={"kind": "plan_finished"},
            )
        )
        if self.notifier is not None:
            await self.notifier.plan_finished(plan.id, summary)
        if self.on_plan_finished is not None:
            try:
                await self.on_plan_finished(plan)
            except Exception:
                pass
        self._resolve_plan_done(plan.id)

    def _resolve_plan_done(self, plan_id: str) -> None:
        fut = self._plan_done.pop(plan_id, None)
        if fut is not None and not fut.done():
            fut.set_result(None)

    async def _build_plan_finished_summary(
        self,
        plan: Plan,
        succeeded: list[Subtask],
        failed: list[Subtask],
    ) -> str:
        fallback = _plan_finished_summary(plan, succeeded, failed)
        if not _needs_planner_summary(plan):
            return fallback
        if self.planner is None or not hasattr(self.planner, "summarize_results"):
            return fallback
        if self.config.planner_vlm_provider == "mock":
            return fallback
        try:
            answer = (await self.planner.summarize_results(plan)).strip()
        except Exception:
            return fallback
        if not answer:
            return fallback
        header = (
            f"goal: {plan.goal} | "
            f"success: {len(succeeded)}/{len(plan.subtasks)}"
            + (f" | failures: {[s.id for s in failed]}" if failed else "")
        )
        return header + "\n" + _truncate(answer, 1500)

    # ---- reactor coordination helpers ----

    def _reactor_busy(self, plan_id: str) -> bool:
        """Whether the Reactor still owes work for this plan (defer finalize).

        Delegates to the Reactor's public :meth:`Reactor.is_busy` rather than
        inspecting its private state (ADR-0006). ``None`` reactor => never busy.
        """
        if self.reactor is None:
            return False
        return self.reactor.is_busy(plan_id)

    def _finish_watchdog_budget_s(self) -> float:
        """Wait for recovery completion using the configured planner timeout."""
        planner_timeout = float(getattr(self.config, "planner_vlm_timeout_s", 60.0) or 60.0)
        return max(15.0, planner_timeout * self._planner_attempts() + 15.0)

    def _planner_attempts(self) -> int:
        """How many HTTP attempts one ``planner.react()`` may make.

        Read off the live Planner rather than duplicated as a constant here: a
        second copy silently goes stale the day someone changes the retry count,
        and the symptom would be the watchdog quietly under-covering again.
        Falls back to the Planner's own default when unreachable.
        """
        planner = getattr(self.reactor, "planner", None) if self.reactor else None
        retries = getattr(planner, "max_retries", None)
        try:
            return max(1, int(retries) + 1)
        except (TypeError, ValueError):
            return 3

    async def _defer_finish_recheck(self, plan: Plan) -> None:
        """Poll the reactor settle state and re-attempt finalization."""
        budget_s = self._finish_watchdog_budget_s()
        waited = 0.0
        step = 0.05
        while waited < budget_s:
            await asyncio.sleep(step)
            waited += step
            if plan.id not in self._plans:
                return  # someone else finalized
            if not self._reactor_busy(plan.id):
                await self.maybe_finish_plan(plan)
                return
            # Force-finalize an unresponsive plan and record its outstanding planner calls.
            # Identify replies arriving after finalization as orphaned.
        inflight_s = self._planner_inflight_for(plan.id)
        inflight_n = self._planner_inflight_count()
        stamp = time.time()
        abandoning = inflight_s is not None
        self._force_finalized[plan.id] = {
            "ts": stamp,
            "budget_s": budget_s,
            "planner_inflight_s": inflight_s,
            "planner_inflight_count": inflight_n,
        }
        detail = (
            f"planner_inflight={inflight_s:.1f}s ABANDONING_RESPONSE=yes"
            if abandoning else "planner_inflight=none ABANDONING_RESPONSE=no"
        )
        line = (
            f"[reactor force-finalize] {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(stamp))} "
            f"plan={plan.id} budget={budget_s:.0f}s {detail} "
            f"planner_calls_inflight_total={inflight_n}"
        )
        sys.stderr.write(line + "\n")
        self.memory.append(
            Message(
                role="system_event",
                content=(
                    f"reactor did not settle for {plan.id} within {budget_s:.0f}s; "
                    "force-finalizing" + (
                        f" WHILE a planner call has been in flight {inflight_s:.1f}s "
                        "-- its reply may dispatch into a finished plan"
                        if abandoning else ""
                    )
                ),
                plan_id=plan.id,
                metadata={
                    "kind": "reactor_force_finalize",
                    "ts": stamp,
                    "budget_s": budget_s,
                    "planner_inflight_s": inflight_s,
                    "planner_inflight_count": inflight_n,
                    "abandoning_response": abandoning,
                },
            )
        )
        await self.maybe_finish_plan(plan, force=True)

    def was_force_finalized(self, plan_id: str) -> Optional[dict]:
        """Force-finalize a plan record, or return None if it is unknown.

        The reactor uses the finalized state to identify late planner replies."""
        return self._force_finalized.get(plan_id)

    def _planner_inflight_for(self, plan_id: str) -> Optional[float]:
        """Seconds the reactor's planner call has been running, if any."""
        fn = getattr(self.reactor, "planner_inflight_for", None) if self.reactor else None
        if fn is None:
            return None
        try:
            return fn(plan_id)
        except Exception:  # noqa: BLE001 - diagnostics must not break finalize
            return None

    def _planner_inflight_count(self) -> int:
        """Planner calls in flight across all plans; 0 when unavailable."""
        fn = getattr(self.reactor, "planner_inflight_count", None) if self.reactor else None
        if fn is None:
            return 0
        try:
            return int(fn())
        except Exception:  # noqa: BLE001
            return 0


def _needs_planner_summary(plan: Plan) -> bool:
    for sub in plan.subtasks:
        if sub.status != "success" or sub.result is None or not sub.result.ok:
            continue
        data = sub.result.data
        if sub.endpoint == "web_query":
            return True
        if isinstance(data, dict) and data.get("kind") in {"weather", "search"}:
            return True
    return False


def _plan_finished_summary(plan: Plan, succeeded: list[Subtask], failed: list[Subtask]) -> str:
    summary = (
        f"goal: {plan.goal} | "
        f"success: {len(succeeded)}/{len(plan.subtasks)}"
        + (f" | failures: {[s.id for s in failed]}" if failed else "")
    )
    result_lines = [
        line
        for sub in plan.subtasks
        if sub.status == "success"
        for line in [_brief_subtask_result(sub)]
        if line
    ]
    if result_lines:
        summary += "\n" + "\n".join(result_lines)
    return summary


def _brief_subtask_result(sub: Subtask) -> str:
    result = sub.result
    if result is None or not result.ok or result.data is None:
        return ""
    data = result.data
    if isinstance(data, dict):
        brief = data.get("answer") or data.get("summary") or data.get("note")
        if not brief:
            brief = _compact_json(data)
    else:
        brief = str(data)
    brief = _truncate(str(brief).strip(), 1000)
    return f"- {sub.id} @ {sub.endpoint}: {brief}" if brief else ""


def _compact_json(data: object) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(data)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"

