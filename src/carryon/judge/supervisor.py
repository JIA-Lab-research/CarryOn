from __future__ import annotations

import asyncio
import itertools
import sys
import time
from typing import Any, Awaitable, Callable, Optional

from ..endpoints import RobotEndpoint
from ..types import OnResult, Plan, Subtask
from ..providers import SceneCapture
from .client import VLMJudgeClient, build_vlm_judge_client
from .config import VLMJudgeConfig


SceneCallback = Callable[[str, Plan, Subtask, list[dict]], Awaitable[None]]
# (job_key, decision, poll_context) for every poll -- used by the recovery
# recorder to build a per-episode judge timeline. The context is the one the
# judge was actually asked with, including the per-poll fields. Fired
# best-effort; must not raise.
TickCallback = Callable[[str, Any, dict], None]


def judge_evidence(decision: Any) -> tuple[Optional[float], Optional[str]]:
    """Pull (confidence, visual_evidence) out of a judge decision, if present."""
    result = getattr(decision, "result", None)
    if not isinstance(result, dict):
        return None, None
    conf = result.get("confidence")
    try:
        score = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        score = None
    evidence = result.get("visual_evidence")
    note = str(evidence) if evidence is not None else None
    return score, note


def judge_recovery_object(decision: Any) -> Optional[str]:
    """Pull ``result.recovery_object`` (the object the Judge says recovery should
    re-grasp) out of a judge decision. Returns a trimmed noun phrase or ``None``
    when the field is absent/empty/null -- the recovery layer then falls back to
    the configured hint / subtask instruction."""
    result = getattr(decision, "result", None)
    if not isinstance(result, dict):
        return None
    val = result.get("recovery_object")
    if val is None:
        return None
    name = str(val).strip()
    return name or None


def judge_placed_match(decision: Any) -> tuple[Optional[str], Optional[bool]]:
    """Extract (placed_object, placed_matches_instruction) from a decision.

    Success identity gate inputs: placed_object is the judge's free visual
    label of whatever it saw inside the destination; placed_matches_instruction
    is its explicit attestation that this is the instruction's target object.
    Missing/malformed fields return (None, None) so the gate fails closed."""
    result = getattr(decision, "result", None)
    if not isinstance(result, dict):
        return None, None
    obj = result.get("placed_object")
    if not isinstance(obj, str) or not obj.strip():
        obj = None
    match = result.get("placed_matches_instruction")
    if not isinstance(match, bool):
        match = None
    return obj, match


def judge_recovery_point(decision: Any) -> Optional[tuple[float, float]]:
    """Pull ``result.recovery_point`` -- the Judge's coarse *normalized* (x, y)
    location of the recovery target in the most recent image -- out of a judge
    decision.

    Returns ``(x, y)`` with each component clamped to ``[0, 1]`` (fraction of
    image width/height, origin top-left), or ``None`` when the field is
    absent/malformed. This is a semantic *anchor*, not a precise grasp point:
    perception enumerates instance masks geometrically and picks the one nearest
    this point, so the Judge only has to be roughly right.
    """
    result = getattr(decision, "result", None)
    if not isinstance(result, dict):
        return None
    val = result.get("recovery_point")
    if val is None:
        return None
    try:
        x, y = float(val[0]), float(val[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    # Accept either [0,1] fractions or 0-1000 / pixel-ish integers: if either
    # component is >1, treat the pair as 0-1000 normalized (Qwen convention) and
    # rescale. Final clamp keeps it on-image regardless.
    if x > 1.0 or y > 1.0:
        x, y = x / 1000.0, y / 1000.0
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    return (x, y)


def scan_drop_state(samples, held_samples: int, last_hold_ts, cfg):
    """Fold a (ts, width) sample series into the drop-lane state.

    Returns ``(held_samples, last_hold_ts, fire)``. The judge polls every
    ~3.3s while the hold->drop transition lasts well under a second, so the
    lane scans the 50Hz ring series and keeps its own memory across polls:

    - every in-band sample (hold_min..hold_max) bumps ``held_samples`` and
      advances ``last_hold_ts`` -- re-holding therefore RESETS the countdown
      (self-heal: the VLA re-picked the object, do not intervene);
    - FIRE when the gripper accumulated >= proprio_hold_min_samples of
      holding, the latest sample is out of the holding band, and
      ``proprio_grace_s`` have passed since the last in-band sample (the
      disturbance drops the object in EVERY episode; self-heal median is
      ~5.5s post-drop, so an immediate fire would re-create the 97%
      over-intervention this lane replaces).
    """
    if not samples:
        return held_samples, last_hold_ts, False
    for ts, w in samples:
        if cfg.proprio_hold_min_m <= w <= cfg.proprio_hold_max_m:
            held_samples += 1
            if last_hold_ts is None or ts > last_hold_ts:
                last_hold_ts = ts
    now_ts, now_w = samples[-1]
    holding_now = cfg.proprio_hold_min_m <= now_w <= cfg.proprio_hold_max_m
    fire = (
        not holding_now
        and held_samples >= cfg.proprio_hold_min_samples
        and last_hold_ts is not None
        and (now_ts - last_hold_ts) >= cfg.proprio_grace_s
    )
    return held_samples, last_hold_ts, fire


# Bound on the per-subtask text copied into every poll's prompt. The progress
# block is re-sent on each poll, so an unbounded instruction/result would be
# paid for once per poll for the whole watch.
_PROGRESS_TEXT_LIMIT = 160


def _clip(text: Any, limit: int = _PROGRESS_TEXT_LIMIT) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def subtask_result_digest(sub: Subtask) -> Optional[str]:
    """One-line outcome of a finished subtask, for the Judge's progress block.

    Mirrors the digest the Reactor already shows the planner
    (``ContextBuilder.build_for_reaction``) so both VLMs read plan history in
    the same shape. ``None`` for a subtask that never produced a result.
    """
    res = sub.result
    if res is None:
        return None
    if not res.ok:
        return _clip(f"{sub.status}: {res.error or 'no error detail'}")
    data = res.data
    if isinstance(data, dict) and data.get("summary"):
        brief = str(data["summary"])
    elif data is not None:
        brief = str(data)
    else:
        brief = ""
    return _clip(f"success: {brief}" if brief else "success")


def plan_progress(plan: Plan, subtask: Subtask) -> dict[str, Any]:
    """Snapshot where ``subtask`` sits inside ``plan``, for the Judge context.

    The Judge watches ONE subtask but sees a scene that earlier subtasks already
    changed. Without this block it cannot tell "the destination already held a
    slice of bread when my watch started" from "my subtask just delivered it",
    and rules a stale scene success on poll 1. So it gets:

    - ``completed_subtasks``: every subtask that already reached a terminal
      status, each with its outcome. Whatever they changed is somebody else's
      achievement, not this subtask's -- for the common sequential plan they
      finished before this watch and their work is already in the baseline
      frame. Failed/cancelled ones are listed too, with their outcome: they may
      have changed the scene partway, and hiding them would leave the Judge
      crediting this subtask for a half-finished earlier one.
    - ``pending_subtasks``: not yet terminal (queued or still running), so the
      Judge neither waits for nor takes credit for undispatched work.
    - ``subtask_index``: position/total, the cheap "am I the 3rd of 3" signal.

    Called per poll rather than snapshotted at dispatch, so a sibling that
    completes mid-watch moves from pending to completed while the watch is
    still open. That matters for parallel DAG layers: such a sibling's change
    is absent from the baseline but present in the current frame, and only a
    live block tells the Judge whose it is.

    Note ``instruction`` here is the planner's authored text; unlike the watched
    subtask's own instruction it is NOT placeholder-resolved, since the values
    would be resolved against results this block already summarises.
    """
    entries = list(plan.subtasks)
    # Identity, not ==: Subtask is a plain dataclass, so == compares every
    # field. Unique ids make a false match impossible today, but the loop below
    # matches on identity and these two must not be able to disagree.
    # position 0 = watched outside its plan's list: a coherent (if positionless)
    # block beats crashing dispatch.
    position = next((i + 1 for i, s in enumerate(entries) if s is subtask), 0)

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for other in entries:
        if other is subtask:
            continue
        item: dict[str, Any] = {
            "id": other.id,
            "instruction": _clip(other.instruction),
        }
        if other.status == "success":
            digest = subtask_result_digest(other)
            item["result"] = digest if digest is not None else "success"
            completed.append(item)
        elif other.status in ("failed", "cancelled", "timeout"):
            digest = subtask_result_digest(other)
            item["result"] = digest if digest is not None else other.status
            completed.append(item)
        else:
            pending.append(item)
    return {
        "subtask_index": {"position": position, "total": len(entries)},
        "completed_subtasks": completed,
        "pending_subtasks": pending,
    }


def cameras_for_endpoint(
    cameras: list[dict[str, Any]],
    endpoint: RobotEndpoint,
) -> list[dict[str, Any]]:
    """Scope the judge's configured camera set to one endpoint.

    An endpoint entry may declare an optional ``cameras: [name, ...]`` list in
    the endpoints YAML (kept in ``RobotEndpoint.extra``) naming which of the
    judge's configured cameras actually cover that robot's workspace. When
    present, only those cameras are fetched/judged for jobs on this endpoint,
    so evidence from an unrelated robot's view can never decide the verdict.
    Absent / malformed / matching nothing => the full configured set (legacy
    behaviour, and fail-open on a name typo rather than judging blind).
    """
    wanted = (endpoint.extra or {}).get("cameras")
    if not isinstance(wanted, list) or not wanted:
        return list(cameras)
    names = {str(n) for n in wanted}
    scoped = [
        c for c in cameras
        if isinstance(c, dict) and str(c.get("name") or "") in names
    ]
    return scoped or list(cameras)


class VLMJudgeSupervisor:
    """Poll a VLM judge and publish only terminal decisions to JobManager.

    Optionally captures a multi-camera snapshot when each job starts and
    invokes ``on_scene_captured`` with the snapshots; the wiring in
    :mod:`carryon.main` routes these into :meth:`Memory.record_subtask_start_scene`
    so the Reactor can build before/after image context for failure replan.
    """

    def __init__(
        self,
        *,
        config: VLMJudgeConfig,
        on_result: OnResult,
        client: VLMJudgeClient | None = None,
        scene_capture: SceneCapture | None = None,
        on_scene_captured: Optional[SceneCallback] = None,
        on_tick: Optional[TickCallback] = None,
        state_provider=None,
    ) -> None:
        self.config = config
        self.on_result = on_result
        self.client = client or build_vlm_judge_client(config)
        self.scene_capture = scene_capture
        self.on_scene_captured = on_scene_captured
        self.on_tick = on_tick
        # Optional live proprioception source: () -> list of recent measured
        # gripper widths. Feeds the VLM context and the deterministic drop lane.
        self.state_provider = state_provider
        self._tasks: set[asyncio.Task] = set()
        # Mints watch_id. A retry re-dispatches the same subtask under the same
        # job_key, so job_key cannot identify a WATCH -- only this can.
        self._watch_seq = itertools.count(1)

    def start_job(
        self,
        *,
        job_key: str,
        plan: Plan,
        subtask: Subtask,
        endpoint: RobotEndpoint,
        instruction: str,
        inputs: dict,
        vla_job_id: str | None,
        deadline_s: float,
        objective: str | None = None,
    ) -> None:
        """Start watching one dispatched subtask.

        ``objective`` is the planner's judge-only completion criterion
        (:attr:`Subtask.objective`), already placeholder-resolved by the caller
        the same way ``instruction`` is. It is passed separately rather than
        read off ``subtask`` precisely to keep that resolution in one place.
        Falls back to ``instruction`` when the planner set no objective.
        """
        context = {
            "job_key": job_key,
            # Identifies THIS watch. job_key repeats across retries of the same
            # subtask; watch_id never does, so the judge client can scope one
            # baseline per attempt.
            "watch_id": f"{job_key}#{next(self._watch_seq)}",
            "vla_job_id": vla_job_id,
            "plan_id": plan.id,
            "plan_goal": plan.goal,
            # Read-only background provenance. The current instruction and
            # subtask_objective below remain the Judge's decision target.
            "original_user_request": plan.original_user_request,
            "subtask_id": subtask.id,
            "endpoint": {
                "name": endpoint.name,
                "transport": endpoint.transport,
                "address": endpoint.address,
                "description": endpoint.description,
                "tags": list(endpoint.tags),
                "skill_library": list(endpoint.skill_library or []),
            },
            "instruction": instruction,
            # What "done" looks like, for the Judge only. The VLA never sees it
            # (see the Subtask.objective dual-field contract).
            "subtask_objective": (objective or "").strip() or instruction,
            # Why the Planner emitted this step. "observe" means its point is to
            # find something out, which is what turns on the reveal report in
            # DEFAULT_SYSTEM_PROMPT: the judge is the only party looking at the
            # pixels when an observe step finishes, so if it does not say what
            # appeared, nothing downstream ever learns it.
            "subtask_purpose": (getattr(subtask, "purpose", "") or "").strip(),
            "inputs": inputs or {},
            "deadline_s": deadline_s,
            "cameras": cameras_for_endpoint(self.config.cameras, endpoint),
            "system_prompt": self.config.system_prompt,
        }
        task = asyncio.create_task(
            self._watch(job_key, context, plan=plan, subtask=subtask),
            name=f"vlm-judge:{job_key}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.client.aclose()
        # scene_capture is owned by the caller (main.py shares it with the
        # Reactor); the caller closes it on shutdown.

    async def _capture_start_scene(self, plan: Plan, subtask: Subtask) -> None:
        if self.scene_capture is None or self.on_scene_captured is None:
            return
        try:
            snapshots = await self.scene_capture.capture()
        except Exception:
            return
        if not snapshots:
            return
        try:
            await self.on_scene_captured(
                subtask.id, plan, subtask, [s.to_dict() for s in snapshots]
            )
        except Exception:
            pass

    async def _watch(
        self,
        job_key: str,
        context: dict[str, Any],
        *,
        plan: Plan,
        subtask: Subtask,
    ) -> None:
        """Run one watch and release its frame memory however it ends.

        The release runs on the terminal verdict, on max_polls, on cancellation
        at shutdown, and on an unexpected raise -- the client's own LRU bound is
        only a backstop for the paths that somehow skip this.
        """
        try:
            await self._watch_loop(job_key, context, plan=plan, subtask=subtask)
        finally:
            release = getattr(self.client, "release_scope", None)
            if release is not None:
                try:
                    release(context.get("watch_id") or job_key)
                except Exception:  # noqa: BLE001 - cleanup must not mask the verdict
                    pass

    async def _watch_loop(
        self,
        job_key: str,
        context: dict[str, Any],
        *,
        plan: Plan,
        subtask: Subtask,
    ) -> None:
        await self._capture_start_scene(plan, subtask)
        polls = 0
        # Polls this watch that were transport/API failures rather than model
        # verdicts. Those are reported as `running` (we learned nothing), which
        # is right for control and dangerous for the record: an episode whose
        # judge was unreachable throughout now looks like a quiet, healthy watch.
        # Counting them is what lets such an episode be found and quarantined
        # afterwards instead of being read as evidence.
        judge_api_errors = 0
        judge_parse_errors = 0
        consecutive_api_errors = 0
        success_streak = 0
        failure_streak = 0
        no_progress_polls = 0
        held_samples = 0
        last_hold_ts = None
        proprio_drop_fired = False
        start_ts = time.monotonic()
        recent: list[dict[str, Any]] = []
        while True:
            if self.config.max_polls and polls >= self.config.max_polls:
                # Carries the same quarantine counters as the normal terminal.
                # This is the path a total judge outage exits through, so it is
                # the LAST place that may omit them: without the marker, "the
                # judge never reached the gateway" is recorded as an ordinary
                # judge-driven failure.
                await self.on_result(
                    job_key,
                    {
                        "status": "failed",
                        "error": "VLM judge reached max_polls without terminal decision",
                        "result": {"source": "vlm_judge"},
                        "meta": {
                            "source": "vlm_judge",
                            "polls": polls,
                            "judge_api_errors": judge_api_errors,
                            "judge_parse_errors": judge_parse_errors,
                            "verdict_from_model": False,
                            "reason": "max_polls",
                        },
                    },
                )
                return
            polls += 1
            elapsed_s = round(max(0.0, time.monotonic() - start_ts), 2)
            poll_context: dict[str, Any] = {
                **context,
                "poll_index": polls,
                "elapsed_s": elapsed_s,
                # Recomputed per poll, not snapshotted at dispatch: a sibling
                # running in parallel can finish mid-watch, and its change IS
                # in the current frame. Snapshotting would leave it listed as
                # pending forever, i.e. as nobody's work -- and the judge might
                # then credit it to this subtask, the exact mis-attribution the
                # progress block exists to prevent.
                **plan_progress(plan, subtask),
            }
            if self.config.stuck_timeout_s > 0:
                poll_context["stuck_timeout_s"] = self.config.stuck_timeout_s
            if recent:
                poll_context["recent_observations"] = recent
            proprio_drop = False
            if self.state_provider is not None:
                try:
                    samples = self.state_provider() or []
                except Exception:  # noqa: BLE001 - proprio is best-effort
                    samples = []
                if samples:
                    widths = [w for _, w in samples]
                    poll_context["proprioception"] = {
                        "gripper_width_m": round(float(widths[-1]), 4),
                        "recent_widths_m": [round(float(w), 4) for w in widths[-6:]],
                        "note": (
                            "measured jaw opening: >=0.008 = holding an object; "
                            "<0.005 = jaws closed on air (object lost); "
                            "~0.08 = fully open"
                        ),
                    }
                    if self.config.proprio_drop_enabled and not proprio_drop_fired:
                        held_samples, last_hold_ts, proprio_drop = scan_drop_state(
                            samples, held_samples, last_hold_ts, self.config
                        )
            decision = await self.client.judge(poll_context)
            is_api_error = bool(getattr(decision, "api_error", False))
            is_parse_error = bool(getattr(decision, "parse_error", False))
            # Both mean the same thing for control: this poll produced no
            # information about the robot. Counted apart because the remedies
            # differ -- fix the gateway vs fix the model/prompt.
            is_inconclusive = is_api_error or is_parse_error
            if is_parse_error:
                judge_parse_errors += 1
            if is_inconclusive:
                if is_api_error:
                    judge_api_errors += 1
                consecutive_api_errors += 1
                limit = int(getattr(self.config, "api_error_terminal_polls", 0) or 0)
                if limit > 0 and consecutive_api_errors >= limit:
                    # Bound the outage. Publishing here routes through the normal
                    # judge-terminal path, which cancels the arm and waits for
                    # release -- unlike waiting forever, which leaves the VLA
                    # running uncancelled to the job deadline and then advances
                    # the DAG over a live policy. The payload says plainly that
                    # this is OUR infrastructure failing, not the robot.
                    await self.on_result(
                        job_key,
                        {
                            "status": "failed",
                            "error": (
                                f"VLM judge unreachable for {consecutive_api_errors} "
                                f"consecutive polls; giving up so the arm is not left "
                                f"running unsupervised. Last error: {decision.error}"
                            ),
                            "result": {
                                "source": "vlm_judge",
                                "planner_action": "invoke_correction",
                                "infrastructure_failure": True,
                            },
                            "meta": {
                                "source": "vlm_judge",
                                "polls": polls,
                                "judge_api_errors": judge_api_errors,
                                "judge_parse_errors": judge_parse_errors,
                                "verdict_from_model": False,
                                "reason": "judge_unreachable",
                            },
                        },
                    )
                    return
                if judge_api_errors == 1:
                    # Loud on the FIRST one. A judge that is down degrades this
                    # watch to unsupervised, and the operator needs to know that
                    # while the run is still cheap to abort -- not afterwards
                    # from a counter in an artifact.
                    print(
                        f"[vlm judge] API ERROR on {job_key} -- reporting "
                        f"'running' (inconclusive, not intervening). The judge "
                        f"is NOT supervising this job: {decision.error}",
                        file=sys.stderr,
                    )
            else:
                # A real verdict got through: the link is alive again, so the
                # give-up countdown starts over. Only a SUSTAINED outage should
                # terminate a watch.
                consecutive_api_errors = 0
            if proprio_drop:
                # Deterministic fast lane: the held->closed-on-air signature IS
                # the dropped-object failure (post grace window); do not wait
                # for the visual judge (which watched a dropped ketchup and
                # voted ok x3) or the min_fail floor (which guards imprecise
                # triggers, not this one).
                proprio_drop_fired = True
                decision.status = "failed"
                decision.error = None
                decision.result = {
                    "planner_action": "invoke_correction",
                    "confidence": 1.0,
                    "visual_evidence": (
                        "proprioceptive drop signature: gripper was holding, "
                        "then stayed empty past the self-heal grace window "
                        f"(width={poll_context['proprioception']['gripper_width_m']} m, "
                        f"grace={self.config.proprio_grace_s}s)"
                    ),
                }
            score, evidence = judge_evidence(decision)
            if decision.status == "success" and getattr(
                self.config, "success_requires_placed_match", False
            ):
                placed_obj, placed_match = judge_placed_match(decision)
                if not (placed_obj is not None and placed_match is True):
                    # Identity gate: a success verdict without a named
                    # result.placed_object and an explicit
                    # placed_matches_instruction=true attestation is demoted to
                    # running BEFORE streak/stuck handling -- a box inside the
                    # basket is not necessarily THE instructed object, and the
                    # demoted poll must still count toward the stuck horizon.
                    decision.status = "running"
                    note = (
                        " [demoted: success lacked placed_object/"
                        "placed_matches_instruction identity attestation]"
                    )
                    evidence = (evidence or "") + note
                    if isinstance(decision.result, dict):
                        decision.result["visual_evidence"] = (
                            str(decision.result.get("visual_evidence") or "") + note
                        )
            forced_stuck = False
            if (
                decision.status == "running"
                and not is_inconclusive
                and self.config.stuck_timeout_s > 0
                and self.config.stuck_force_failed_polls > 0
                and elapsed_s >= self.config.stuck_timeout_s
            ):
                no_progress_polls += 1
                if no_progress_polls >= self.config.stuck_force_failed_polls:
                    forced_stuck = True
                    decision.status = "failed"
                    decision.result = {
                        "planner_action": "invoke_correction",
                        "confidence": 1.0,
                        "visual_evidence": evidence
                        or "no visible task progress after stuck timeout",
                        "stuck_evidence": (
                            f"forced stuck after {no_progress_polls} running polls "
                            f"at elapsed_s={elapsed_s} >= stuck_timeout_s="
                            f"{self.config.stuck_timeout_s}"
                        ),
                    }
                    decision.error = None
            elif not is_inconclusive:
                # An inconclusive poll leaves the no-progress count alone: we did
                # not observe "no progress", we observed nothing. Resetting it
                # would let an intermittent gateway hide a genuinely stuck arm.
                no_progress_polls = 0
            if self.config.recent_observations_window > 0:
                recent.append(
                    {
                        "poll": polls,
                        "elapsed_s": elapsed_s,
                        "status": decision.status,
                        "visual_evidence": evidence,
                    }
                )
                if len(recent) > self.config.recent_observations_window:
                    del recent[: -self.config.recent_observations_window]
            if self.on_tick is not None:
                try:
                    # poll_context, not the base context: a superset, so every
                    # key an existing consumer reads is still there, plus the
                    # per-poll ones (poll_index, elapsed_s, progress block,
                    # proprioception) a timeline actually wants to record.
                    #
                    # Diagnostics are added HERE, on a copy, and never to the
                    # dict handed to the client: poll_context is serialised
                    # verbatim into the judge's user turn, so putting an
                    # operational counter in it would change the model's input
                    # and break comparability with pre-fix runs.
                    self.on_tick(job_key, decision, {
                        **poll_context,
                        "judge_api_errors": judge_api_errors,
                        "judge_parse_errors": judge_parse_errors,
                        "api_error": is_api_error,
                        "parse_error": is_parse_error,
                    })
                except Exception:
                    pass
            should_publish = False
            if is_inconclusive:
                # A poll that learned nothing must not erase what earlier polls
                # DID learn. Resetting here means a gateway that 429s every other
                # poll permanently disarms the judge: failure_streak can never
                # reach its threshold, no terminal is ever published, and the
                # episode runs to the job deadline unsupervised. Leave the
                # streaks exactly as they were.
                pass
            elif decision.status == "running":
                success_streak = 0
                failure_streak = 0
            elif decision.status == "success":
                success_streak += 1
                failure_streak = 0
                should_publish = (
                    success_streak >= self.config.success_streak_threshold
                )
            elif decision.status == "failed":
                failure_streak += 1
                success_streak = 0
                should_publish = (
                    proprio_drop
                    or
                    forced_stuck
                    or
                    failure_streak >= self.config.failure_streak_threshold
                )
                # Conservative floor: a failed verdict (forced-stuck included)
                # only publishes after min_fail_elapsed_s -- the VLA gets a
                # real chance to recover on its own before the closed loop
                # intervenes (see JudgeConfig.min_fail_elapsed_s rationale).
                if (
                    should_publish
                    and not proprio_drop
                    and self.config.min_fail_elapsed_s > 0
                    and elapsed_s < self.config.min_fail_elapsed_s
                ):
                    should_publish = False
            elif decision.status == "cancelled":
                success_streak = 0
                failure_streak = 0
                should_publish = True

            if should_publish:
                payload: dict[str, Any] = {
                    "status": decision.status,
                    "result": decision.result or {"source": "vlm_judge"},
                    "error": decision.error,
                    "meta": {
                        "source": "vlm_judge",
                        "polls": polls,
                        "success_streak": success_streak,
                        "failure_streak": failure_streak,
                        "forced_stuck": forced_stuck,
                        # Quarantine handles. `judge_api_errors` vs `polls` is
                        # the honest pair -- deliberately NOT a "was it
                        # supervised" boolean, which would report True for a
                        # watch that got one verdict out of fifty and read as
                        # healthy in a sweep filter. `verdict_from_model` is the
                        # narrow claim that CAN be made: whether the poll that
                        # actually decided this job came from the model rather
                        # than from a transport fault or a deterministic lane.
                        "judge_api_errors": judge_api_errors,
                        "judge_parse_errors": judge_parse_errors,
                        "verdict_from_model": not (
                            is_inconclusive or proprio_drop or forced_stuck
                        ),
                    },
                }
                if payload["error"] is None:
                    payload.pop("error")
                await self.on_result(job_key, payload)
                return
            # Keep the supervisor loop cancellable even when no polling limit is set.
            await asyncio.sleep(max(0.0, self.config.poll_interval_s))
