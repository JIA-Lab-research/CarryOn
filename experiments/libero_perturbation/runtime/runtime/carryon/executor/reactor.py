"""Forward endpoint events to the planner and dispatch resulting subtasks.

Failure, timeout and status events supply context to planner.react(). New steps
receive unique round prefixes and are submitted through Executor. Empty plans
record a no-action decision. is_busy and maybe_finish_plan coordinate completion."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
from ..memory import Memory
from ..types import Message, Plan, Subtask
if TYPE_CHECKING:
    from ..planner.planner import Planner
    from ..providers import SceneCapture
    from .executor import Executor

@dataclass
class _PlanReactState:
    """Per-plan reactor bookkeeping."""
    rounds: int = 0
    pending_events: list[dict] = field(default_factory=list)
    inflight: bool = False
    debounce_task: Optional[asyncio.Task] = None

class Reactor:
    """Aggregates events per-plan and asks the planner VLM what to do next.

    Events are debounced (``debounce_s``) so that bursts of callbacks --
    several siblings finishing within milliseconds -- result in a SINGLE planner VLM
    call carrying all of them together, instead of one call per event.
    """

    def __init__(self, *, planner: 'Planner', executor: 'Executor', memory: Memory, notifier=None, debounce_s: float=0.05, scene_capture: 'SceneCapture | None'=None) -> None:
        self.planner = planner
        self.executor = executor
        self.memory = memory
        self.notifier = notifier
        self.debounce_s = debounce_s
        self.scene_capture = scene_capture
        self._state: dict[str, _PlanReactState] = {}
        self._lock = asyncio.Lock()

    def is_busy(self, plan_id: str) -> bool:
        """Whether the Reactor still owes work for ``plan_id``.

        Part of the explicit Reactor<->Executor contract (ADR-0006): the
        Executor consults this before finalizing a plan so it doesn't finish
        while the Reactor is mid-call or has events queued that may append
        more subtasks. True iff a planner-VLM call is in flight OR events are
        buffered waiting for the next debounce tick.

        Intentionally does NOT consult ``debounce_task.done()``: the Reactor
        calls :meth:`Executor.maybe_finish_plan` from inside that very task's
        finally-block, where ``inflight`` is already False and
        ``pending_events`` is empty but the task hasn't returned yet. Treating
        that as "busy" would deadlock the finalize path.
        """
        st = self._state.get(plan_id)
        if st is None:
            return False
        return bool(st.inflight or st.pending_events)

    async def _collect_failure_images(self, plan: Plan, sub: Subtask) -> dict | None:
        """Return ``{"before": [...], "after": [...]}`` or ``None`` if no
        images are available. Errors are swallowed so a flaky camera never
        breaks the failure path.
        """
        before: list[dict] = []
        getter = getattr(self.memory, 'get_subtask_start_scene', None)
        if callable(getter):
            try:
                before = list(getter(plan.id, sub.id) or [])
            except Exception:
                before = []
        after: list[dict] = []
        if self.scene_capture is not None:
            try:
                snapshots = await self.scene_capture.capture()
                after = [s.to_dict() for s in snapshots]
            except Exception:
                after = []
        if not before and (not after):
            return None
        return {'before': before, 'after': after}

    async def on_subtask_failed(self, plan: Plan, sub: Subtask) -> None:
        event = {'kind': 'subtask_failed', 'subtask_id': sub.id, 'endpoint': sub.endpoint, 'attempts': sub.attempts, 'detail': sub.result.error if sub.result is not None and sub.result.error else f'status={sub.status}'}
        images = await self._collect_failure_images(plan, sub)
        if images is not None:
            event['images'] = images
        await self._enqueue_event(plan, event)

    async def on_status_update(self, plan: Plan, sub: Optional[Subtask], payload: dict) -> None:
        endpoint = (sub.endpoint if sub is not None else None) or ((payload or {}).get('endpoint') or '?')
        sid = sub.id if sub is not None else (payload or {}).get('subtask_id') or '-'
        detail = ''
        if isinstance(payload, dict):
            detail = payload.get('detail') or payload.get('note') or ''
            if not detail and isinstance(payload.get('result'), dict):
                r = payload['result']
                detail = r.get('summary') or r.get('note') or str(r)[:200]
            if not detail and payload.get('result') is not None:
                detail = str(payload.get('result'))[:200]
        await self._enqueue_event(plan, {'kind': 'endpoint_status_update', 'subtask_id': sid, 'endpoint': endpoint, 'status': (payload or {}).get('status'), 'detail': detail})

    async def _enqueue_event(self, plan: Plan, event: dict) -> None:
        async with self._lock:
            st = self._state.setdefault(plan.id, _PlanReactState())
            st.pending_events.append(event)
            if st.debounce_task is not None and (not st.debounce_task.done()):
                st.debounce_task.cancel()
            st.debounce_task = asyncio.create_task(self._debounced_react(plan), name=f'reactor:{plan.id}')

    async def _debounced_react(self, plan: Plan) -> None:
        try:
            await asyncio.sleep(self.debounce_s)
        except asyncio.CancelledError:
            return
        await self._run_react(plan)

    async def _run_react(self, plan: Plan) -> None:
        async with self._lock:
            st = self._state.get(plan.id)
            if st is None or not st.pending_events or st.inflight:
                return
            events = list(st.pending_events)
            st.pending_events.clear()
            st.inflight = True
            st.rounds += 1
            round_no = st.rounds
        try:
            try:
                new_plan = await self.planner.react(plan, events)
            except Exception as e:
                self.memory.append(Message(role='system_event', content=f'reactor: planner.react failed for {plan.id}: {type(e).__name__}: {e}', plan_id=plan.id, metadata={'kind': 'reactor_error'}))
                if self.notifier is not None:
                    await self.notifier.info(f'[{plan.id}] reactor error: {type(e).__name__}: {e}')
                return
            await self._merge_and_dispatch(plan, new_plan, round_no)
        finally:
            async with self._lock:
                st = self._state.get(plan.id)
                if st is not None:
                    st.inflight = False
                    if st.pending_events:
                        if st.debounce_task is not None and (not st.debounce_task.done()):
                            st.debounce_task.cancel()
                        st.debounce_task = asyncio.create_task(self._debounced_react(plan), name=f'reactor:{plan.id}')
            try:
                await self.executor.maybe_finish_plan(plan)
            except Exception:
                pass

    async def _merge_and_dispatch(self, plan: Plan, new_plan: Plan, round_no: int) -> None:
        if not new_plan.subtasks:
            self.memory.append(Message(role='assistant', content=f'reactor[{plan.id} r{round_no}]: no action -- {new_plan.goal}', plan_id=plan.id, metadata={'kind': 'reactor_noop'}))
            if self.notifier is not None:
                await self.notifier.info(f'[{plan.id}] reactor: no action ({new_plan.goal})')
            return
        prefix = f'r{round_no}-'
        existing_ids = {s.id for s in plan.subtasks}
        new_subs: list[Subtask] = []
        for s in new_plan.subtasks:
            s.id = self._unique_id(prefix + s.id, existing_ids)
            existing_ids.add(s.id)
            s.depends_on = [prefix + d for d in s.depends_on]
            s.depends_on = [d for d in s.depends_on if d in existing_ids]
            new_subs.append(s)
        plan.subtasks.extend(new_subs)
        self.memory.append(Message(role='assistant', content=f'reactor[{plan.id} r{round_no}]: appended {[s.id for s in new_subs]} -- {new_plan.goal}', plan_id=plan.id, metadata={'kind': 'reactor_appended'}))
        if self.notifier is not None:
            await self.notifier.info(f"[{plan.id}] reactor: dispatched {', '.join((s.endpoint for s in new_subs if not s.depends_on))}")
        roots = [s for s in new_subs if not s.depends_on]
        await asyncio.gather(*(self.executor.submit_subtask(plan, s) for s in roots), return_exceptions=False)

    @staticmethod
    def _unique_id(candidate: str, taken: set[str]) -> str:
        if candidate not in taken:
            return candidate
        i = 2
        while f'{candidate}-{i}' in taken:
            i += 1
        return f'{candidate}-{i}'
