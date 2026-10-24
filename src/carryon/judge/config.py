from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlparse


def judge_host(base_url: Any) -> str:
    """Host of the gateway the judge will actually call.

    For banners and diagnostics. ``provider`` names a wire protocol, not a
    destination -- ``provider: openrouter`` pointed at a dead Aliyun MaaS
    workspace for a whole session while the banner reported nothing amiss.
    Printing the host makes the destination checkable at a glance.

    Returns ``"(default)"`` when unset (the client falls back to OpenRouter) and
    ``"(unparseable: ...)"`` rather than raising, since a banner must never be
    the thing that stops the agent starting.
    """
    raw = str(base_url or "").strip()
    if not raw:
        return "(default)"
    try:
        parsed = urlparse(raw if "//" in raw else f"//{raw}")
        # hostname, NOT netloc: netloc keeps any embedded credentials, and this
        # string goes to stdout at startup and into every captured log. Some
        # OpenAI-compatible gateways accept inline keys in the URL.
        host = parsed.hostname or ""
        if host and parsed.port:
            host = f"{host}:{parsed.port}"
        return host or parsed.path.split("/", 1)[0] or f"(unparseable: {raw[:40]})"
    except Exception:  # noqa: BLE001
        return f"(unparseable: {raw[:40]})"


# Visual judge prompt.
DEFAULT_SYSTEM_PROMPT = """You are CarryOn VLM Judge, a visual progress supervisor for a robot manipulation task.
You receive the currently executing task, endpoint metadata, job ids, timing context, previous observations, and camera inputs.

Your job is to decide whether the planner should keep waiting, advance to the next task, or invoke a correction module.
Return STRICT JSON only. Do not include markdown or prose.

The context may include original_user_request. It is the user's original input
and is provided as read-only background reference. The Planner has already
decomposed that request: instruction and subtask_objective are the Planner's
output for the ONE current subtask you must judge. Use the original request for
background context only; judge completion against the current subtask.

Use exactly one of these statuses:
- running: the task is still progressing or the visual evidence is inconclusive.
- success: the requested task is visibly complete; planner_action must be advance_to_next_task.
- failed: the task is visibly impossible, unsafe, clearly failed, or stuck beyond stuck_timeout_s; planner_action must be invoke_correction.
- cancelled: the task was externally cancelled.

YOU JUDGE ONE SUBTASK OF A LONGER PLAN, AND YOU JUDGE TRANSITIONS, NOT STATES.
The context gives you subtask_index (which subtask of how many you are watching), completed_subtasks (already finished before your watch began, with their outcomes), and pending_subtasks (not dispatched yet). The baseline frame is the scene AT THE MOMENT YOUR WATCH STARTED -- not the start of the episode. Therefore:
- Success means a NEW state change that happened DURING your watch: a difference between the baseline frame and the current frame that achieves this subtask's goal. Only that change is yours to judge.
- Anything ALREADY IN PLACE in the baseline frame was put there by an earlier subtask (see completed_subtasks). It does NOT satisfy your subtask, no matter how well it matches your instruction's words. An object that is present at baseline and simply still present now is NOT success.
- completed_subtasks is the list of work that is NOT yours, whether it finished before your watch (so it is visible in the baseline) or finished during it (a subtask running in parallel with yours -- its change appears mid-watch and is still not yours). If a change matches something in completed_subtasks, it is that subtask's, not yours.
- If the baseline frame already shows what looks like your goal state and nothing new has arrived, return running -- and failed once elapsed_s passes stuck_timeout_s. Never success.
- When your destination already holds objects at baseline (a stack, a pile, a partly built assembly), success requires a further, newly-arrived object matching THIS subtask. Count and compare: the destination must contain something at the end that it did not contain at the baseline.
- Never report success for work described in completed_subtasks. That is done and it is not yours.
- Judge against subtask_objective when it is present: it is the planner's explicit completion criterion for THIS subtask, written for you. instruction is the raw command sent to the robot and may be phrased for the policy rather than for judging; when the two differ, subtask_objective wins.

You are given a baseline frame captured at the start of THIS subtask's watch and the most recent current frame(s). Judge by comparing the current scene against the baseline scene and the goal:
- Report success ONLY when the current frame shows the goal state achieved: the target object is clearly AT / INSIDE the specified target location. For "place X in/into Y", X must be visibly inside Y, not merely near, beside, or on the table by it.
- Use the baseline frame to confirm a real change occurred toward the goal. If the current scene is essentially unchanged from the baseline, the task is still running (or stuck), not success.
- If the object ended up outside the target (on the table/floor/beside it) after a place attempt, that is failed (mis-placed / dropped), not success.
- Match the object to the instruction: confirm it is the correct object in the correct destination before declaring success.
- IDENTITY CHECK BEFORE SUCCESS (mandatory): before returning success you MUST first describe, from the pixels, WHICH object is now inside the destination (result.placed_object) and then decide whether it is the instruction's target object (result.placed_matches_instruction). "Some object is in the destination" is NOT success. If the placed object is a different object than instructed, return failed. If you cannot confidently identify it, return running, not success.
- TARGET-STILL-OUTSIDE CHECK: if the instruction's target object is still visible OUTSIDE the destination (on the table/floor or in the gripper), the task CANNOT be success -- whatever is inside the destination must be a wrong object; return failed. Perform this check FIRST; it is usually easier than identifying the object inside the destination. EXCEPTION -- interchangeable duplicates: when the scene holds several equivalent instances of the target kind (two slices of bread, a stack of identical plates) and the plan moves them one per subtask, this check applies only to the ONE instance your subtask moved. A leftover spare of the same kind sitting outside the destination is expected and is NOT failure; decide from the baseline-vs-current difference at the destination instead.

REVEAL REPORT (mandatory when subtask_purpose is "observe"): the point of an observe step is to find something out, and you are the only party that looks at the pixels at the moment it finishes. Describe, from the pixels, what is now visible in the area this step uncovered that the baseline frame did not show (result.revealed_contents). Describe it the way you would to someone who has not been told what is being looked for -- shape, colour, material, count. Do NOT phrase it as confirming or denying the instruction's target, and do NOT fall back on what the instruction, the goal, or the original request expects to be there: the planner has no other source for this, so an expectation echoed back reaches it as an observation and is acted on. Naming something you cannot actually make out is worse than saying so; if the area is still occluded, blurred, or empty, say that instead. This report is independent of the status verdict -- an observe step can succeed at uncovering a place and reveal something nobody wanted. Whenever you fill result.revealed_contents, also set result.scene_update_required to true. What you describe exists only in this reply: the planner keeps a written record of what is where, and that record stays stale until it writes down what you saw. You are not deciding what the record should say, only flagging that this observation has not reached it yet. Set it to false when revealed_contents is null.

Progress must be judged from the TARGET OBJECT and GOAL STATE, not from robot motion. Arm/gripper motion alone is NOT progress. A gripper hovering above the object, repeatedly approaching it, pushing on the table/object, opening/closing near it, or moving without the object changing task-relevant state is still no progress. For "pick X from A and place in B", progress means X visibly leaves A, is securely carried toward B, or is placed inside B.

STUCK = FAILURE (MANDATORY OVERRIDE): If elapsed_s exceeds stuck_timeout_s AND the recent observations plus the current frame show no clear progress toward the goal versus the baseline/earlier frames (the target object has not moved toward or into the target location), you MUST return status="failed" with planner_action="invoke_correction" and a concrete stuck_evidence string. Treat repeated attempts that press, scrape, or collide near the table without task-relevant object progress as STUCK/unsafe. An arm that keeps moving or hovering without advancing the goal is STUCK -- do NOT keep returning "running". Prefer running unless there is clear visual evidence for success or failure; when in doubt after the timeout with no object-level progress, choose "failed".

ALREADY-PLACED GUARD (mandatory -- check before EVERY failed or stuck verdict; it OVERRIDES the STUCK=FAILURE rule above): if the current frame shows the instruction's target object already INSIDE the specified destination, you MUST NOT return failed. A failed verdict here invokes a correction module that pulls the placed object back out and destroys a completed task. When the target is already in its destination, decide only between success and running -- success if that placement is the NEW change during your watch, running if it was already there at your baseline (an earlier subtask placed it); failed is never correct while the target sits in its destination.

Required JSON fields:
- status: one of running, success, failed, cancelled.
- result.planner_action: one of continue_observing, advance_to_next_task, invoke_correction.
- result.confidence: number between 0 and 1.
- result.visual_evidence: short string grounded in the image.
- result.stuck_evidence: short string or null.
- result.placed_object: free-form visual label of the object currently inside/at the target location, or null if none. Required (non-null) whenever status is success.
- result.placed_matches_instruction: boolean -- true only when placed_object is confidently the SAME physical object the instruction asks to place. Required whenever status is success.
- result.revealed_contents: free-form description of what this step made newly visible, or null when it made nothing visible. Required (non-null) whenever subtask_purpose is "observe" and status is success.
- result.scene_update_required: boolean -- true whenever revealed_contents is non-null, false otherwise. It says this observation has not reached the planner's written record yet; it does not say what the record should contain.
- error: string or null."""


@dataclass
class VLMJudgeConfig:
    provider: str = "openrouter"  # openrouter | mock
    model: str = "qwen/qwen2.5-vl-72b-instruct"
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 30.0
    poll_interval_s: float = 1.0
    max_polls: int = 0  # Zero waits until the job deadline or cancellation.
    # For OpenAI-compatible / Gemini judge clients, each poll includes up to
    # N most recent frames per camera (oldest -> newest).
    history_window_frames: int = 3
    # Terminal gating: a single success/failed decision is not enough to close
    # a job; supervisor requires N consecutive terminal decisions before
    # publishing to JobManager.
    success_streak_threshold: int = 3
    failure_streak_threshold: int = 2
    # Informational stuck horizon (seconds) fed to the judge each poll. The
    # prompt asks the model to treat "no visible progress after elapsed_s >
    # stuck_timeout_s" as stuck/failed. 0 = omit from context (no stuck hint).
    # Consecutive transport/API failures before the judge gives up and publishes
    # an INFRASTRUCTURE failure terminal.
    #
    # An API error is reported as `running` -- we learned nothing about the
    # robot, so a blip must not cancel a healthy episode. But "keep waiting"
    # cannot be unbounded: with max_polls=0 (production) and the forced-stuck
    # lane off, a sustained outage would leave the watch polling forever while
    # the VLA runs UNCANCELLED to the job deadline, after which the DAG advances
    # over a still-live policy. That is worse on a real arm than the false
    # failure this whole change removes.
    #
    # So: transient outage -> running (do not intervene); sustained outage ->
    # terminate through the normal judge path, which cancels the arm and waits
    # for release, with the payload plainly labelled as OUR failure rather than
    # the robot's. 0 disables the bound (unbounded waiting -- do not use on a
    # real arm).
    api_error_terminal_polls: int = 5
    stuck_timeout_s: float = 0.0
    # Deterministic stuck fallback. Some judge VLMs report "running" with their
    # own evidence noting "no visible change" well past stuck_timeout_s instead of
    # escalating to failed, so recovery never engages. When > 0, after this many
    # consecutive "running" polls taken at/after stuck_timeout_s (i.e. no terminal
    # decision and no progress within the stuck window), the supervisor forces a
    # stuck "failed" verdict. 0 = disabled (trust the VLM only).
    stuck_force_failed_polls: int = 0
    # Minimum wall-clock time before publishing a failed verdict, including
    # forced-stuck failures. Zero disables this floor.
    min_fail_elapsed_s: float = 0.0
    # Proprioceptive drop trigger: a held->closed-on-air gripper transition is
    # the DEFINING signature of a dropped object and needs no VLM. It fires on
    # the fast lane and bypasses min_fail_elapsed_s. Widths: holding ~
    # object width (>= hold_min); closed on
    # air < empty_max; fully open ~0.08 (a deliberate place/release opens the
    # jaws and does NOT match).
    proprio_drop_enabled: bool = False
    proprio_hold_min_m: float = 0.008
    proprio_hold_max_m: float = 0.075
    proprio_empty_max_m: float = 0.005
    # Minimum number of in-band ring samples (50Hz) that count as "was
    # holding" -- 8 samples ~= 0.16s of stable grasp.
    proprio_hold_min_samples: int = 8
    # Grace window (s) after the drop signature before firing: the disturbance
    # drops the object in EVERY episode and the VLA often re-picks it on its
    # own (self-heal median ~5.5s post-drop). The lane fires only if the
    # gripper has NOT returned to the holding band within this window --
    # "dropped AND failed to self-recover", precise and patient.
    proprio_grace_s: float = 10.0
    # How many of the judge's own recent observations (status + evidence) to
    # feed back each poll so it has temporal memory instead of judging blind.
    recent_observations_window: int = 3
    # Identity gate for success. VLM judges routinely see "an object inside the
    # destination" and declare success even when it is the WRONG object (layout
    # distractors: a box in the basket is not necessarily THE cream cheese). When
    # True, a success decision only counts toward success_streak_threshold if the
    # judge also returned a non-null result.placed_object AND
    # result.placed_matches_instruction == true (chain-of-verification: it must
    # first NAME what it sees in the destination, then attest the match).
    # Decisions failing the gate are demoted to running (streak reset).
    success_requires_placed_match: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    cameras: list[dict[str, Any]] = field(default_factory=list)

    # OpenAI-compatible chat-completions knobs.
    temperature: float = 0.0
    max_tokens: int = 512
    default_headers: dict[str, str] = field(default_factory=dict)
    http_referer: str | None = None
    x_title: str | None = None

    # Per-camera HTTP GET timeout when fetching snapshots for multimodal calls.
    image_fetch_timeout_s: float = 5.0
    # Debug mode: dump each judge query's full request/response and image inputs.
    debug_enabled: bool = False
    debug_output_dir: str = "./logs/vlm_judge_debug"

    # Mock provider only. Each item can be a status string or a dict shaped like
    # {status, result?, error?, terminal?}. Decisions are consumed in order.
    mock_decisions: list[Any] = field(default_factory=lambda: [{"status": "success"}])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLMJudgeConfig":
        flat = _flatten(data or {})
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in flat.items() if k in allowed and v is not None}
        cfg = cls(**kwargs)
        _validate(cfg)
        return cfg

    @classmethod
    def from_file(cls, path: str) -> "VLMJudgeConfig":
        import yaml

        from carryon.config import expand_env_vars

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"VLM judge config {path}: top-level must be a mapping")
        data = expand_env_vars(data)
        return cls.from_dict(data)


_GROUP_ALIASES: dict[tuple[str, str], str] = {
    ("vlm", "provider"): "provider",
    ("vlm", "model"): "model",
    ("vlm", "api_key"): "api_key",
    ("vlm", "base_url"): "base_url",
    ("vlm", "timeout_s"): "timeout_s",
    ("vlm", "temperature"): "temperature",
    ("vlm", "max_tokens"): "max_tokens",
    ("vlm", "default_headers"): "default_headers",
    ("vlm", "http_referer"): "http_referer",
    ("vlm", "x_title"): "x_title",
    ("judge", "poll_interval_s"): "poll_interval_s",
    ("judge", "max_polls"): "max_polls",
    ("judge", "history_window_frames"): "history_window_frames",
    ("judge", "success_streak_threshold"): "success_streak_threshold",
    ("judge", "failure_streak_threshold"): "failure_streak_threshold",
    ("judge", "stuck_timeout_s"): "stuck_timeout_s",
    ("judge", "min_fail_elapsed_s"): "min_fail_elapsed_s",
    ("judge", "proprio_drop_enabled"): "proprio_drop_enabled",
    ("judge", "proprio_hold_min_m"): "proprio_hold_min_m",
    ("judge", "proprio_hold_max_m"): "proprio_hold_max_m",
    ("judge", "proprio_empty_max_m"): "proprio_empty_max_m",
    ("judge", "proprio_hold_min_samples"): "proprio_hold_min_samples",
    ("judge", "proprio_grace_s"): "proprio_grace_s",
    ("judge", "stuck_force_failed_polls"): "stuck_force_failed_polls",
    ("judge", "recent_observations_window"): "recent_observations_window",
    ("judge", "success_requires_placed_match"): "success_requires_placed_match",
    ("judge", "system_prompt"): "system_prompt",
    ("camera", "cameras"): "cameras",
    ("camera", "fetch_timeout_s"): "image_fetch_timeout_s",
    ("debug", "enabled"): "debug_enabled",
    ("debug", "output_dir"): "debug_output_dir",
    ("cameras", "items"): "cameras",
    ("mock", "decisions"): "mock_decisions",
}


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                mapped = _GROUP_ALIASES.get((k, sk))
                out[mapped or f"{k}_{sk}"] = sv
        else:
            out[k] = v
    return out


def _validate(cfg: VLMJudgeConfig) -> None:
    if cfg.provider not in ("openrouter", "mock"):
        raise ValueError(f"unsupported VLM judge provider: {cfg.provider!r}")
    if cfg.timeout_s <= 0:
        raise ValueError("VLM judge timeout_s must be > 0")
    if cfg.poll_interval_s < 0:
        raise ValueError("VLM judge poll_interval_s must be >= 0")
    if cfg.max_polls < 0:
        raise ValueError("VLM judge max_polls must be >= 0")
    if cfg.history_window_frames < 1:
        raise ValueError("VLM judge history_window_frames must be >= 1")
    if cfg.success_streak_threshold < 1:
        raise ValueError("VLM judge success_streak_threshold must be >= 1")
    if cfg.failure_streak_threshold < 1:
        raise ValueError("VLM judge failure_streak_threshold must be >= 1")
    if cfg.stuck_timeout_s < 0:
        raise ValueError("VLM judge stuck_timeout_s must be >= 0")
    if cfg.stuck_force_failed_polls < 0:
        raise ValueError("VLM judge stuck_force_failed_polls must be >= 0")
    if cfg.recent_observations_window < 0:
        raise ValueError("VLM judge recent_observations_window must be >= 0")
    if not isinstance(cfg.cameras, list):
        raise ValueError("VLM judge cameras must be a list")
    if cfg.image_fetch_timeout_s <= 0:
        raise ValueError("VLM judge image_fetch_timeout_s must be > 0")
    if not isinstance(cfg.debug_output_dir, str) or not cfg.debug_output_dir.strip():
        raise ValueError("VLM judge debug_output_dir must be a non-empty string")
    if cfg.max_tokens <= 0:
        raise ValueError("VLM judge max_tokens must be > 0")