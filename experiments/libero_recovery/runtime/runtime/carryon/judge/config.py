from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any
DEFAULT_SYSTEM_PROMPT = 'You are CarryOn VLM Judge, a visual progress supervisor for a robot manipulation task.\nYou receive the currently executing task, endpoint metadata, job ids, timing context, previous observations, and camera inputs.\n\nYour job is to decide whether the planner should keep waiting, advance to the next task, or invoke a correction module.\nReturn STRICT JSON only. Do not include markdown or prose.\n\nUse exactly one of these statuses:\n- running: the task is still progressing or the visual evidence is inconclusive.\n- success: the requested task is visibly complete; planner_action must be advance_to_next_task.\n- failed: the task is visibly impossible, unsafe, clearly failed, or stuck beyond stuck_timeout_s; planner_action must be invoke_correction.\n- cancelled: the task was externally cancelled.\n\nYou are given a reference frame captured at the start of the episode and the most recent current frame(s). Judge by comparing the current scene against the reference scene and the instruction\'s goal:\n- Report success ONLY when the current frame shows the instruction\'s goal state achieved: the target object is clearly AT / INSIDE the specified target location. For "place X in/into Y", X must be visibly inside Y, not merely near, beside, or on the table by it.\n- Use the reference (start) frame to confirm a real change occurred toward the goal. If the current scene is essentially unchanged from the reference, the task is still running (or stuck), not success.\n- If the object ended up outside the target (on the table/floor/beside it) after a place attempt, that is failed (mis-placed / dropped), not success.\n- Match the object to the instruction: confirm it is the correct object in the correct destination before declaring success.\n- IDENTITY CHECK BEFORE SUCCESS (mandatory): before returning success you MUST first describe, from the pixels, WHICH object is now inside the destination (result.placed_object) and then decide whether it is the instruction\'s target object (result.placed_matches_instruction). "Some object is in the destination" is NOT success. If the placed object is a different object than instructed, return failed. If you cannot confidently identify it, return running, not success.\n- TARGET-STILL-OUTSIDE CHECK: if the instruction\'s target object is still visible OUTSIDE the destination (on the table/floor or in the gripper), the task CANNOT be success -- whatever is inside the destination must be a wrong object; return failed. Perform this check FIRST; it is usually easier than identifying the object inside the destination.\n- POST-DROP IDENTITY SKEPTICISM: when proprioception carries a drop_alert (the gripper went from holding to empty earlier), the object the gripper holds NOW may be a look-alike picked after the drop. Do NOT report running/success on the basis of "the arm is carrying the object" -- first verify from the pixels that the held object IS the instructed one; if it is a different object, return failed (wrong object in gripper after a drop).\n- ALREADY-PLACED GUARD (mandatory before any failed verdict): if the current frame shows the instructed target object already INSIDE the destination, the task is COMPLETE -- return success (with placed_object / placed_matches_instruction filled), NEVER failed. A failed verdict here would trigger a correction that pulls the placed object back out and destroys a completed task.\n\nPrefer running unless there is clear visual evidence for success or failure. Treat lack of visible progress across the provided observation history, after elapsed_s exceeds stuck_timeout_s, as stuck.\n\nRequired JSON fields:\n- status: one of running, success, failed, cancelled.\n- result.planner_action: one of continue_observing, advance_to_next_task, invoke_correction.\n- result.confidence: number between 0 and 1.\n- result.visual_evidence: short string grounded in the image.\n- result.stuck_evidence: short string or null.\n- result.placed_object: free-form visual label of the object currently inside/at the target location, or null if none. Required (non-null) whenever status is success.\n- result.placed_matches_instruction: boolean -- true only when placed_object is confidently the SAME physical object the instruction asks to place. Required whenever status is success.\n- result.target_released: boolean -- true only when pixels show the target is no longer held by the gripper. Required whenever status is success.\n- result.target_in_destination: boolean -- true only when pixels show the target inside/on the instructed destination. Required whenever status is success.\n- result.placement_stable: boolean -- true only when the released placement is visibly settled. Required whenever status is success.\n- error: string or null.'

@dataclass
class VLMJudgeConfig:
    provider: str = 'openrouter'
    model: str = 'qwen/qwen2.5-vl-72b-instruct'
    api_key: str | None = None
    base_url: str = 'https://openrouter.ai/api/v1'
    timeout_s: float = 30.0
    poll_interval_s: float = 1.0
    max_polls: int = 0
    history_window_frames: int = 3
    success_streak_threshold: int = 3
    success_settle_interval_s: float = 1.0
    failure_streak_threshold: int = 2
    stuck_timeout_s: float = 0.0
    stuck_force_failed_polls: int = 0
    min_fail_elapsed_s: float = 0.0
    failure_takeover_wall_cap_s: float = 50.0
    proprio_drop_enabled: bool = False
    proprio_hold_min_m: float = 0.008
    proprio_hold_max_m: float = 0.075
    proprio_empty_max_m: float = 0.005
    proprio_hold_min_samples: int = 8
    proprio_grace_s: float = 10.0
    recent_observations_window: int = 3
    success_requires_placed_match: bool = False
    checkpoint_enabled: bool = False
    checkpoint_pause: bool = True
    checkpoint_triage: bool = True
    checkpoint_release_stuck_count: int = 3
    checkpoint_max_pauses: int = 6
    checkpoint_emptygrasp_count: int = 0
    checkpoint_stable_s: float = 0.06
    checkpoint_handback_grace_samples: int = 20
    checkpoint_stable_band_m: float = 0.003
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    cameras: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 512
    default_headers: dict[str, str] = field(default_factory=dict)
    http_referer: str | None = None
    x_title: str | None = None
    image_fetch_timeout_s: float = 5.0
    debug_enabled: bool = False
    debug_output_dir: str = './logs/vlm_judge_debug'
    mock_decisions: list[Any] = field(default_factory=lambda: [{'status': 'success'}])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'VLMJudgeConfig':
        flat = _flatten(data or {})
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in flat.items() if k in allowed and v is not None}
        cfg = cls(**kwargs)
        _validate(cfg)
        return cfg

    @classmethod
    def from_file(cls, path: str) -> 'VLMJudgeConfig':
        import yaml
        from carryon.config import expand_env_vars
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f'VLM judge config {path}: top-level must be a mapping')
        data = expand_env_vars(data)
        return cls.from_dict(data)
_GROUP_ALIASES: dict[tuple[str, str], str] = {('vlm', 'provider'): 'provider', ('vlm', 'model'): 'model', ('vlm', 'api_key'): 'api_key', ('vlm', 'base_url'): 'base_url', ('vlm', 'timeout_s'): 'timeout_s', ('vlm', 'temperature'): 'temperature', ('vlm', 'max_tokens'): 'max_tokens', ('vlm', 'default_headers'): 'default_headers', ('vlm', 'http_referer'): 'http_referer', ('vlm', 'x_title'): 'x_title', ('judge', 'poll_interval_s'): 'poll_interval_s', ('judge', 'max_polls'): 'max_polls', ('judge', 'history_window_frames'): 'history_window_frames', ('judge', 'success_streak_threshold'): 'success_streak_threshold', ('judge', 'success_settle_interval_s'): 'success_settle_interval_s', ('judge', 'failure_streak_threshold'): 'failure_streak_threshold', ('judge', 'stuck_timeout_s'): 'stuck_timeout_s', ('judge', 'min_fail_elapsed_s'): 'min_fail_elapsed_s', ('judge', 'failure_takeover_wall_cap_s'): 'failure_takeover_wall_cap_s', ('judge', 'proprio_drop_enabled'): 'proprio_drop_enabled', ('judge', 'proprio_hold_min_m'): 'proprio_hold_min_m', ('judge', 'proprio_hold_max_m'): 'proprio_hold_max_m', ('judge', 'proprio_empty_max_m'): 'proprio_empty_max_m', ('judge', 'proprio_hold_min_samples'): 'proprio_hold_min_samples', ('judge', 'proprio_grace_s'): 'proprio_grace_s', ('judge', 'stuck_force_failed_polls'): 'stuck_force_failed_polls', ('judge', 'recent_observations_window'): 'recent_observations_window', ('judge', 'success_requires_placed_match'): 'success_requires_placed_match', ('judge', 'checkpoint_enabled'): 'checkpoint_enabled', ('judge', 'checkpoint_pause'): 'checkpoint_pause', ('judge', 'checkpoint_triage'): 'checkpoint_triage', ('judge', 'checkpoint_release_stuck_count'): 'checkpoint_release_stuck_count', ('judge', 'checkpoint_max_pauses'): 'checkpoint_max_pauses', ('judge', 'checkpoint_emptygrasp_count'): 'checkpoint_emptygrasp_count', ('judge', 'checkpoint_stable_s'): 'checkpoint_stable_s', ('judge', 'checkpoint_handback_grace_samples'): 'checkpoint_handback_grace_samples', ('judge', 'checkpoint_stable_band_m'): 'checkpoint_stable_band_m', ('judge', 'system_prompt'): 'system_prompt', ('camera', 'cameras'): 'cameras', ('camera', 'fetch_timeout_s'): 'image_fetch_timeout_s', ('debug', 'enabled'): 'debug_enabled', ('debug', 'output_dir'): 'debug_output_dir', ('cameras', 'items'): 'cameras', ('mock', 'decisions'): 'mock_decisions'}

def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                mapped = _GROUP_ALIASES.get((k, sk))
                out[mapped or f'{k}_{sk}'] = sv
        else:
            out[k] = v
    return out

def _validate(cfg: VLMJudgeConfig) -> None:
    if cfg.provider not in ('openrouter', 'mock'):
        raise ValueError(f'unsupported VLM judge provider: {cfg.provider!r}')
    if cfg.timeout_s <= 0:
        raise ValueError('VLM judge timeout_s must be > 0')
    if cfg.poll_interval_s < 0:
        raise ValueError('VLM judge poll_interval_s must be >= 0')
    if cfg.max_polls < 0:
        raise ValueError('VLM judge max_polls must be >= 0')
    if cfg.history_window_frames < 1:
        raise ValueError('VLM judge history_window_frames must be >= 1')
    if cfg.success_streak_threshold < 1:
        raise ValueError('VLM judge success_streak_threshold must be >= 1')
    if cfg.success_settle_interval_s < 0:
        raise ValueError('VLM judge success_settle_interval_s must be >= 0')
    if cfg.failure_streak_threshold < 1:
        raise ValueError('VLM judge failure_streak_threshold must be >= 1')
    if cfg.stuck_timeout_s < 0:
        raise ValueError('VLM judge stuck_timeout_s must be >= 0')
    if cfg.failure_takeover_wall_cap_s < 0:
        raise ValueError('VLM judge failure_takeover_wall_cap_s must be >= 0')
    if cfg.stuck_force_failed_polls < 0:
        raise ValueError('VLM judge stuck_force_failed_polls must be >= 0')
    if cfg.recent_observations_window < 0:
        raise ValueError('VLM judge recent_observations_window must be >= 0')
    if cfg.checkpoint_release_stuck_count < 0:
        raise ValueError('VLM judge checkpoint_release_stuck_count must be >= 0')
    if cfg.checkpoint_max_pauses < 0:
        raise ValueError('VLM judge checkpoint_max_pauses must be >= 0')
    if cfg.checkpoint_emptygrasp_count < 0:
        raise ValueError('VLM judge checkpoint_emptygrasp_count must be >= 0')
    if cfg.checkpoint_stable_s <= 0:
        raise ValueError('VLM judge checkpoint_stable_s must be > 0')
    if cfg.checkpoint_stable_band_m <= 0:
        raise ValueError('VLM judge checkpoint_stable_band_m must be > 0')
    if not isinstance(cfg.cameras, list):
        raise ValueError('VLM judge cameras must be a list')
    if cfg.image_fetch_timeout_s <= 0:
        raise ValueError('VLM judge image_fetch_timeout_s must be > 0')
    if not isinstance(cfg.debug_output_dir, str) or not cfg.debug_output_dir.strip():
        raise ValueError('VLM judge debug_output_dir must be a non-empty string')
    if cfg.max_tokens <= 0:
        raise ValueError('VLM judge max_tokens must be > 0')
