from __future__ import annotations

import asyncio
import os
import re
import sys
from typing import Optional

from ..config import Config
from ..endpoints import EndpointRegistry
from ..errors import PlanError, CarryOnError
from ..executor.executor import Executor
from ..jobs.channels.base import OnResult, ResultChannel
from ..jobs.manager import JobManager
from ..memory import Memory
from ..planner.planner import Planner

from ..types import Message

from .commands import CommandHandler
from .context import ContextBuilder, is_outdoor_weather_gate_plan
from .notifier import Notifier


def _places_named(inputs: dict) -> list:
    """Every place an update_scene subtask names, under any accepted key."""
    out = []
    for key in ("place", "location", "name"):
        v = inputs.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for key in ("places", "locations"):
        for item in inputs.get(key) or []:
            if isinstance(item, dict):
                for k in ("place", "location", "name"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
    return out


class AgentLoop:
    def __init__(
        self,
        *,
        memory: Memory,
        planner: Planner,
        executor: Executor,
        context_builder: ContextBuilder,
        job_manager: JobManager,
        result_channel: ResultChannel,
        notifier: Notifier,
        endpoint_registry: EndpointRegistry,
        command_handler: CommandHandler,
        config: Config,
        result_intake: Optional[OnResult] = None,
    ) -> None:

        self.memory = memory
        self.planner = planner
        self.executor = executor
        self.cb = context_builder
        self.jm = job_manager
        self.rc = result_channel
        self.notifier = notifier
        self.registry = endpoint_registry
        self.commands = command_handler
        self.config = config
        self._result_intake: Optional[OnResult] = result_intake

        self._latest_plan_id: Optional[str] = None

        self._bg_tasks: set[asyncio.Task] = set()
        self._stopping = asyncio.Event()

    # ---- lifecycle ----

    async def start(self) -> None:
        intake = self._result_intake if self._result_intake is not None else self.jm.on_result
        await self.rc.start(intake)
        await self.jm.start()
        # Stop any leftover VLA episode from a previous CarryOn session and
        # park the arm at init before the first user instruction.
        await self._reset_vla_endpoints(reason="startup")

    async def stop(self) -> None:
        self._stopping.set()
        # grace period: let in-flight plans finish naturally before we cancel.
        grace = float(self.config.shutdown_grace_s or 0.0)
        if grace > 0:
            deadline = asyncio.get_running_loop().time() + grace
            while self.jm.plans and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
        # cancel any still-in-flight plans best-effort
        for plan_id in list(self.jm.plans.keys()):
            try:
                await self.jm.cancel_plan(plan_id)
            except Exception:
                pass
        # Extra /reset so gello homes even if cancel raced (unknown request_id)
        # or the arm was idle but still streaming residual motion.
        await self._reset_vla_endpoints(reason="shutdown")
        await self.jm.stop()
        await self.rc.stop()
        # drain background handle tasks
        for t in list(self._bg_tasks):
            t.cancel()
        for t in list(self._bg_tasks):
            try:
                await t
            except BaseException:
                pass

    async def _reset_vla_endpoints(self, *, reason: str) -> None:
        """POST /reset on HTTP arm/VLA endpoints (best-effort, never raises)."""
        reset_tags = {"arm", "vla", "franka"}
        for ep in self.registry.list():
            if ep.transport != "http":
                continue
            tags = set(ep.tags or [])
            if tags and tags.isdisjoint(reset_tags):
                continue
            tr = self.jm.transports.get(ep.transport)
            reset_fn = getattr(tr, "reset", None) if tr is not None else None
            if reset_fn is None:
                continue
            try:
                ok = await reset_fn(ep)
                await self.notifier.info(
                    f"[vla reset] {reason}: {ep.name} ok={ok}"
                )
            except Exception as e:
                await self.notifier.info(
                    f"[vla reset] {reason}: {ep.name} failed: {type(e).__name__}: {e}",
                    important=True,
                )

    # ---- REPL ----

    async def run(self) -> None:
        await self.start()
        # Register the REPL prompt so async stdout writers (Notifier,
        # EventLogger one-liners) can redraw it after their messages
        # instead of leaving the user staring at a blank line.
        from ..logging_util import set_repl_prompt
        set_repl_prompt(self.notifier.prompt)
        try:
            if self.config.verbose:
                sys.stdout.write(
                    f"[callback server] listening on {self.config.callback_host}:{self.config.callback_port}\n"
                    f"[endpoints] loaded {len(self.registry.list())}: "
                    f"{', '.join(self.registry.names()) or '(none)'}\n"
                )
                if self.config.planner_log_path:
                    sys.stdout.write(
                        f"[planner log] {self.config.planner_log_path} "
                        f"(provider={self.config.planner_vlm_provider} model={self.config.planner_vlm_model})\n"
                    )
                cameras = self.config.planner_vlm_cameras or []
                sys.stdout.write(
                    f"[planner vision] provider={self.config.planner_vlm_provider} "
                    f"model={self.config.planner_vlm_model} cameras={len(cameras)}\n"
                )
                if self.config.vlm_judge_enabled:
                    from ..judge.config import judge_host

                    judge_settings = self.config.vlm_judge_settings or {}
                    judge_vlm = judge_settings.get("vlm") or {}
                    judge_provider = judge_vlm.get("provider", "?")
                    # Print the HOST WE WILL ACTUALLY CALL, not just the provider
                    # label. `provider: openrouter` only means "OpenAI-compatible
                    # protocol"; it will happily point at any gateway. An
                    # operator read `provider=openrouter` while every poll went
                    # to a dead Aliyun MaaS workspace and 403'd -- the banner
                    # said nothing was wrong. The host is the fact, the label is
                    # not, so the banner states the fact.
                    sys.stdout.write(
                        f"[vlm judge] enabled provider={judge_provider} "
                        f"model={judge_vlm.get('model', '?')} "
                        f"host={judge_host(judge_vlm.get('base_url'))}\n"
                    )
                    applies = list(self.config.vlm_judge_applies_to_tags)
                    if applies:
                        sys.stdout.write(
                            f"[vlm judge] applies only to endpoints with any of tags={applies}\n"
                        )
                    else:
                        sys.stdout.write(
                            "[vlm judge] applies to all endpoints "
                            "(set vlm_judge.applies_to_tags to restrict)\n"
                        )
                    if self.config.vlm_judge_ignore_executor_terminal:
                        sys.stdout.write(
                            "[vlm judge] executor terminal callbacks ignored "
                            "(only judge closes jobs"
                            + (" for endpoints in scope" if applies else "")
                            + ")\n"
                        )
                sys.stdout.flush()
            while not self._stopping.is_set():
                try:
                    line = await self._read_line()
                except (EOFError, KeyboardInterrupt):
                    break
                if line is None:
                    break
                s = line.strip()
                if not s:
                    continue
                if s in ("exit", "quit"):
                    break
                if CommandHandler.is_command(s):
                    out = await self.commands.dispatch(s)
                    if out:
                        sys.stdout.write(out + "\n")
                        sys.stdout.flush()
                    continue
                # Natural language: run to plan completion before re-prompting.
                # Showing `carryon> ` mid-turn is misleading (user_reply / done
                # would appear under a prompt that looks ready for new input).
                await self._safe_handle(s)
        finally:
            if self.config.verbose:
                sys.stdout.write(
                    f"[shutdown] cancelling {len(self.jm.plans)} in-flight plans...\n"
                )
                sys.stdout.flush()
            # Stop redrawing the prompt during shutdown so trailing log
            # lines don't leave a stray `carryon> ` behind.
            set_repl_prompt("")
            await self.stop()

    async def _read_line(self) -> Optional[str]:
        loop = asyncio.get_running_loop()
        prompt = self.notifier.prompt
        try:
            return await loop.run_in_executor(None, _blocking_input, prompt)
        except (EOFError, KeyboardInterrupt):
            raise

    # ---- handling ----

    async def handle(self, user_input: str) -> None:
        # Inline image attachments: "@image:<path>" (or @image:"path with
        # spaces") tokens anywhere in a normal message attach the image to the
        # user Message (metadata.attachments path refs). The post-plan
        # UserSummarizer sees which user messages carried images and, when it
        # distills a durable entity fact grounded in that message ("this is my
        # lizard" + photo), stores the image as the entity's reference_image
        # -- no explicit command needed (/remember_image stays as a debug
        # backdoor).
        user_text, attachments, missing = extract_image_attachments(user_input)
        for path in missing:
            await self.notifier.info(
                f"[attach] image not found, ignored: {path}", important=True
            )
        # New user instruction: stop any in-flight plan/VLA episode and home
        # the arm before planning, so the next /execute starts from a clean
        # init pose instead of mid-trajectory residual motion.
        for plan_id in list(self.jm.plans.keys()):
            try:
                await self.jm.cancel_plan(plan_id)
            except Exception:
                pass
        await self._reset_vla_endpoints(reason="new_instruction")
        self.memory.append(Message(
            role="user",
            content=user_text,
            metadata={"attachments": attachments} if attachments else {},
        ))
        self.cb.mark_task_start()
        try:
            plan = await self.planner.plan(user_text)

        except PlanError as e:
            await self.notifier.info(f"[plan error] {e}", important=True)
            return
        await self._dispatch_and_wait(plan)
        await self._continue_until_goal(user_text, plan)

    # Continue planning after observation steps until the goal completes, a failure
    # occurs or a stopping guard fires. The round cap prevents unbounded continuation.
    MAX_CONTINUATION_ROUNDS = 10

    async def _continue_until_goal(self, user_text: str, plan) -> None:
        """Keep planning while this turn's goal is not yet reached.

        The planner decides when to stop: the EMPTY-subtasks convention already
        means "nothing to do / goal already achieved". Two further stops guard
        against spinning -- a hard round cap, and a no-progress check (the same
        subtask set proposed twice).

        Every round re-plans with THIS turn's results in context plus a fresh
        camera frame, so the planner knows what it already tried. Without that
        it would reveal the same place forever.
        """
        # Track executed steps. A repeated step gets one corrective re-plan; stop if it repeats again.
        done = set(self._step_signatures(plan))
        # The same ledger, ordered and with status, for the planner to read.
        # Subtask objects are mutated in place by the executor, so appending
        # them at dispatch time still renders current statuses next round.
        executed = list(plan.subtasks)
        # Whether the JUST-EXECUTED plan was produced under a record debt. The
        # memory-only stop exists because a write that follows NO new
        # information cannot change the next round's answer. A write demanded
        # by THIS turn's observation is the opposite case: the turn has an
        # unfinished goal and fresher memory than any turn before it, so
        # stopping there strands the goal -- observed on the rig as a turn that
        # ended with the record written and the target still on the table.
        # False for the initial plan: nothing was owed before anything ran.
        planned_under_debt = False
        for _ in range(self.MAX_CONTINUATION_ROUNDS):
            if self._stopping.is_set() or self._was_cancelled(plan):
                return                      # the user is in charge, not the loop
            if self._had_failure(plan):
                return                      # a failure ends the turn -- see _had_failure
            if self._only_wrote_memory(plan) and not planned_under_debt:
                return                      # nothing new to go on -- see _only_wrote_memory
            owed = self._outstanding_record(
                executed, getattr(getattr(self, "cb", None), "scene_memory", None))
            try:
                nxt = await self._plan_continuation(
                    user_text, plan, self._progress_note(executed), owed,
                    executed=executed)
            except PlanError as e:
                await self.notifier.info(f"[plan error] {e}", important=True)
                return
            if self._stopping.is_set():
                return                      # interrupt landed during the call
            if not nxt.subtasks:
                # The planner is DONE or STUCK -- either way its user_reply is
                # the explanation ("goal reached" / "no library entry covers
                # X"). Dropping it left the operator a bare prompt where the
                # planner had in fact said why it stopped.
                reply = (getattr(nxt, "user_reply", "") or "").strip()
                if reply:
                    await self.notifier.say(f"[{nxt.id}] {reply}")
                return
            repeats = [s for s in nxt.subtasks if self._step_signature(s) in done]
            if repeats:
                remaining = {self._step_signature(s) for s in nxt.subtasks
                             if self._step_signature(s) not in done}
                if not remaining:
                    await self.notifier.info(
                        "[continue] stopping: the plan proposes only steps "
                        "that already ran", important=True)
                    return
                names = "; ".join(s.instruction for s in repeats)
                await self.notifier.info(
                    f"[continue] plan repeats step(s) that already ran "
                    f"({names}) -- asking for only the remaining work",
                    important=True)
                try:
                    nxt = await self.planner.plan(
                        user_text, current_plan_id=plan.id,
                        progress=self._progress_note(executed),
                        executed=executed,
                        correction=(
                            "These step(s) ALREADY EXECUTED in this turn and "
                            f"succeeded: {names}. The world already reflects "
                            "them. Plan ONLY the remaining work; do not "
                            "re-propose completed steps. If nothing remains, "
                            "emit an empty subtasks list."))
                except PlanError as e:
                    await self.notifier.info(f"[plan error] {e}", important=True)
                    return
                if not nxt.subtasks:
                    return
                # The correction's mandate is DROP THE DONE STEPS, nothing
                # more: the corrected plan may only carry the remaining work
                # it was asked to keep. Anything new is not a correction.
                if not all(self._step_signature(s) in remaining
                           for s in nxt.subtasks):
                    await self.notifier.info(
                        "[continue] stopping: the corrected plan is not the "
                        "remaining work it was asked for", important=True)
                    return
            done.update(self._step_signatures(nxt))
            executed.extend(nxt.subtasks)
            planned_under_debt = bool(owed)
            await self._dispatch_and_wait(nxt)
            plan = nxt
        await self.notifier.info(
            f"[continue] stopping after {self.MAX_CONTINUATION_ROUNDS} rounds; "
            "the goal was never reported complete", important=True)

    @staticmethod
    def _outstanding_record(executed, scene=None):
        """The reveal text of an observation nothing has written down yet.

        The judge flags it (result.scene_update_required, set exactly when
        revealed_contents is non-null) and the planner owns the write. Asking
        the planner to remember that write was tried five times across two
        channels -- four policy wordings, then the flag itself riding the very
        result JSON whose revealed_contents it was demonstrably following --
        and it never once dispatched `update_scene`: the write is bookkeeping
        beside the decision in front of it, and prose about bookkeeping does
        not survive contact with that decision. So the loop tracks the debt
        and the parse gate collects it (validate_scene_update_requirement);
        this method only answers "what is still owed".

        Scanned in execution order: a later `update_scene` dispatch clears
        everything before it -- one write settles the ledger, and per-place
        bookkeeping stays with the planner, which is the only party that
        names places.
        """
        # Per place, because a turn can owe two writes at once: the station
        # that turned out to hold the target, and the remembered station that
        # turned out not to. One write that names only the first leaves the
        # second entry standing, and a standing entry is what sends the next
        # request back to the wrong place.
        owed: dict[str, str] = {}
        for s in executed:
            if s.endpoint == "update_scene":
                named = _places_named(getattr(s, "inputs", None) or {})
                if named:
                    for p in named:
                        owed.pop(p, None)
                else:
                    owed.clear()                # a free-form write settles all
                continue
            data = getattr(getattr(s, "result", None), "data", None)
            if not isinstance(data, dict):
                continue
            place = str(data.get("revealed_place") or "").strip()
            if data.get("scene_update_required"):
                text = str(data.get("revealed_contents") or "").strip() \
                    or "the contents this step made visible"
                owed[place or text] = text
            elif (s.endpoint == "check_endpoint" and data.get("present") is False
                  and place and scene is not None):
                # The retraction half. A check at a place scene memory has an
                # entry for came back empty: the entry is now contradicted by
                # this turn's own observation, and it must be rewritten by
                # the party that authored it. No object name is consulted --
                # "you looked where memory said and did not find what you
                # were looking for" is the whole condition.
                try:
                    listed = scene.get_location(place)
                except Exception:
                    listed = None
                if listed:
                    what = str(data.get("target_object") or "the target")
                    owed[place] = (f"{what} was not found at {place}, which "
                                   f"scene memory still lists")
        if not owed:
            return None
        # Judge-flagged reveals carry no place and keep the legacy shape: one
        # string, one write settles it. A check_endpoint result names its
        # station, and then the ledger is per place.
        if all(k == v for k, v in owed.items()):
            return list(owed.values())[-1]
        return owed

    @staticmethod
    def _progress_note(executed) -> str:
        """Execute the current planning round and retain completed-step results for the next round."""
        if not executed:
            return ""
        lines = [f'  {i}. {s.endpoint}: "{s.instruction}" [{s.status}]'
                 for i, s in enumerate(executed, 1)]
        return (
            "STEPS ALREADY EXECUTED THIS TURN -- do not plan any of these "
            "again:\n" + "\n".join(lines) + "\n"
            "Plan ONLY work that is not in that list. The world already "
            "reflects these steps, and the live image may look unchanged even "
            "where one succeeded, so the image is not evidence that a step did "
            "not run. If nothing remains, return an empty subtasks list."
        )

    async def _plan_continuation(self, user_text: str, plan, progress: str = "",
                                 outstanding: str | None = None,
                                 executed: list | None = None):
        """Build the next round.

        ONE specialization survives, and only because the two cases differ in
        WHERE the new information lives. A phase-1 weather lookup returns
        STRUCTURED data that exists only in the subtask result, so it is
        injected explicitly (and the planner is told not to re-query). Every
        other step -- a reveal above all -- puts its new information into the
        WORLD, which the next round's live camera frame carries for free.
        The control flow, the stop conditions and the round budget are shared.

        That second half holds only while a step LEAVES its effect visible. A
        step that restores the scene erases its own trace, and the frame stops
        being a record of what was tried -- so the executed-step ledger rides
        along on every round regardless. See _progress_note.
        """
        if is_outdoor_weather_gate_plan(plan):
            await self.notifier.info(
                f"[{plan.id}] weather ready -- planning pack from result")
            return await self.planner.plan_after_weather(user_text, plan)
        return await self.planner.plan(
            user_text, current_plan_id=plan.id, progress=progress,
            require_scene_update=outstanding, executed=executed)


    def _only_wrote_memory(self, plan) -> bool:
        """Did this plan do nothing but record what was already known?

        Continuation exists because a step can produce NEW information that
        changes what to do next. A memory write produces none: it stores a
        fact we already had. Re-planning after one just re-poses the same
        request against the same world, and the planner obligingly records it
        again -- observed on the rig as five `update_scene` calls for one
        sentence from the user.

        Decided by the `memory` TAG on the endpoint, not by endpoint name, so
        a future bookkeeping skill is covered the day it is tagged. A plan
        with no subtasks is not "memory only" -- that case is the planner's
        own empty-subtasks stop, handled above.
        """
        if not plan.subtasks:
            return False
        tags = {}
        for ep in self.registry.list():
            tags[getattr(ep, "name", "")] = {str(t) for t in (getattr(ep, "tags", None) or [])}
        return all("memory" in tags.get(s.endpoint, set()) for s in plan.subtasks)

    @staticmethod
    def _had_failure(plan) -> bool:
        """Did any of this plan's work FAIL?

        Continuation exists for the plan that SUCCEEDED without reaching the
        goal -- one deliberate step, then decide from what it showed. A failed
        step is a different situation and gets a different answer: stop, and
        report it. Retrying here would silently become a second attempt the
        operator never asked for, and on this rig it would be the ONLY one --
        the executor's Retry/React ladder is switched off in config
        (max_retries 0, react_on_failure false), so nothing else re-dispatches.

        It also keeps the round budget honest: rounds are meant to count
        places examined, and a grasp that missed is not a place examined.
        """
        return any(getattr(s, "status", None) == "failed" for s in plan.subtasks)

    @staticmethod
    def _was_cancelled(plan) -> bool:
        """Did this plan's work get cancelled rather than run to a verdict?

        Ctrl-C and /cancel come back through the normal terminal path, so the
        dispatch call returns without raising. Only the subtask status shows
        it, and continuing past it would be the loop overriding the user.
        """
        return any(getattr(s, "status", None) == "cancelled" for s in plan.subtasks)

    @staticmethod
    def _step_signature(subtask):
        """Identity of ONE piece of work, ignoring id and casing."""
        return (subtask.endpoint, (subtask.instruction or "").strip().lower())

    @classmethod
    def _step_signatures(cls, plan):
        return {cls._step_signature(s) for s in plan.subtasks}

    async def _dispatch_and_wait(self, plan) -> None:
        self._latest_plan_id = plan.id
        reply = (plan.user_reply or "").strip()
        self.memory.append(
            Message(
                role="assistant",
                content=(
                    f"plan {plan.id}: goal={plan.goal}; "
                    f"subtasks={[s.id for s in plan.subtasks]}"
                    + (f"; user_reply={reply}" if reply else "")
                ),
                plan_id=plan.id,
                metadata={"kind": "plan_created"},
            )
        )
        if reply:
            await self.notifier.say(f"[{plan.id}] {reply}")
        if plan.subtasks:
            await self.notifier.say(plan.format_subtasks())
        await self.executor.submit_plan(plan)
        if plan.subtasks:
            roots = [s.id for s in plan.subtasks if not s.depends_on]
            eps = sorted({s.endpoint for s in plan.subtasks if s.id in roots})
            await self.notifier.info(
                f"[{plan.id}] dispatched to: {', '.join(eps)}"
            )
        # Hold the REPL until the plan actually finishes (not just until
        # roots are submitted), so `carryon> ` only returns when idle.
        await self.executor.wait_plan(plan.id)

    async def _safe_handle(self, user_input: str) -> None:
        from ..logging_util import set_repl_prompt

        self.notifier.set_busy(True)
        set_repl_prompt("")
        try:
            await self.handle(user_input)
        except CarryOnError as e:
            await self.notifier.info(f"[error] {e}", important=True)
        except Exception as e:
            await self.notifier.info(
                f"[error] {type(e).__name__}: {e}", important=True
            )
        finally:
            self.notifier.set_busy(False)
            set_repl_prompt(self.notifier.prompt)


def _blocking_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise


# "@image:/path/to/pic.jpg" or "@image:\"/path with spaces/pic.jpg\"".
_IMAGE_TOKEN_RE = re.compile(r'@image:(?:"([^"]+)"|(\S+))')


def extract_image_attachments(
    text: str,
) -> tuple[str, list[dict], list[str]]:
    """Split ``@image:<path>`` tokens out of a user message.

    Returns ``(cleaned_text, attachments, missing)`` where attachments are
    ``{"type": "image", "path": <abs path>}`` dicts destined for
    ``Message.metadata["attachments"]`` and ``missing`` lists tokens whose
    path did not resolve to a readable file (dropped, reported to the user).
    Each token is replaced inline with a VLM-readable marker so the
    planner and summarizer still see WHERE the image sat in the sentence.
    """
    attachments: list[dict] = []
    missing: list[str] = []

    def _sub(m: "re.Match[str]") -> str:
        raw = m.group(1) or m.group(2)
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.isfile(path):
            attachments.append({"type": "image", "path": path})
            return f"[attached image: {os.path.basename(path)}]"
        missing.append(raw)
        return f"[missing image: {raw}]"

    cleaned = _IMAGE_TOKEN_RE.sub(_sub, text or "").strip()
    return cleaned, attachments, missing
