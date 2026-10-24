"""Construction of the optional VLM Judge supervisor.

Host-side wiring lifted out of ``main._amain``: parse the inline ``vlm_judge:``
settings, decide the scene-capture source, and assemble the
:class:`VLMJudgeSupervisor` plus its two host callbacks:

- ``on_scene_captured`` -> records subtask-start snapshots into short-term memory;
- ``on_tick`` -> feeds the judge timeline into the recovery episode recorder
  (only when the recovery loop is recording).

The caller passes in the result intake (``on_result``) and the components the
supervisor must observe/mutate, keeping this builder decoupled from ``main``'s
intake closures. Config errors are raised (``main`` maps them to exit code 2).
"""
from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING, Any
from ..config import Config
from ..providers import SceneCapture
from ..types import OnResult
from .config import VLMJudgeConfig
from .supervisor import VLMJudgeSupervisor, judge_evidence, judge_recovery_object, judge_recovery_point
if TYPE_CHECKING:
    from ..executor.executor import Executor
    from ..executor.reactor import Reactor
    from ..memory.short_term import ShortTermMemory

def _apply_checkpoint_env_overrides(judge_config: VLMJudgeConfig) -> None:
    """Apply checkpoint settings from environment variables.

    CARRYON_CHECKPOINT enables checkpoints; CARRYON_CHECKPOINT_TRIAGE enables
    judging paused frames; CARRYON_CHECKPOINT_PAUSE enables pausing.
    CARRYON_CHECKPOINT_EMPTYGRASP sets the empty-grasp count: 0 disables, 1 selects
    the default count of 3, and values greater than 1 set the count directly."""
    import os

    def _flag(name: str) -> bool | None:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == '':
            return None
        return str(raw).strip().lower() not in ('0', 'false', 'no', 'off')
    for env, attr in (('CARRYON_CHECKPOINT', 'checkpoint_enabled'), ('CARRYON_CHECKPOINT_TRIAGE', 'checkpoint_triage'), ('CARRYON_CHECKPOINT_PAUSE', 'checkpoint_pause')):
        val = _flag(env)
        if val is not None:
            setattr(judge_config, attr, val)
    raw_eg = os.environ.get('CARRYON_CHECKPOINT_EMPTYGRASP')
    if raw_eg is not None and str(raw_eg).strip() != '':
        try:
            n = int(str(raw_eg).strip())
        except ValueError:
            n = 3 if _flag('CARRYON_CHECKPOINT_EMPTYGRASP') else 0
        judge_config.checkpoint_emptygrasp_count = 3 if n == 1 else max(0, n)

def build_judge_supervisor(config: Config, *, scene_capture: SceneCapture | None, reactor: 'Reactor', memory: 'ShortTermMemory', executor: 'Executor', recovery_recorder_registry: Any | None, on_result: OnResult, gello: Any | None=None, job_lookup: Any | None=None) -> tuple[VLMJudgeSupervisor, SceneCapture | None]:
    """Build and wire the VLM Judge supervisor.

    Returns ``(judge_supervisor, scene_capture)``. ``scene_capture`` may be
    created here from the judge's own cameras when none was configured up-front;
    the (possibly new) value is both pushed onto ``reactor`` and returned so the
    caller can reassign its local handle (used at teardown).

    ``gello`` (the recovery control-plane client) and ``job_lookup``
    (``JobManager.get``) enable the D4 gripper-transition checkpoint layer:
    without them ``checkpoint_enabled`` is inert (no pause primitive).
    """
    judge_config = VLMJudgeConfig.from_dict(config.vlm_judge_settings)
    _apply_checkpoint_env_overrides(judge_config)
    if scene_capture is None and judge_config.cameras:
        scene_capture = SceneCapture(list(judge_config.cameras))
        reactor.scene_capture = scene_capture

    async def _record_start_scene(_key: str, plan, subtask, snapshots: list[dict]) -> None:
        memory.record_subtask_start_scene(plan.id, subtask.id, snapshots)
    _judge_on_tick = None
    if recovery_recorder_registry is not None:
        from ..recovery.factory import dump_recovery_episode

        def _judge_on_tick(job_key, decision, context):
            score, note = judge_evidence(decision)
            recovery_recorder_registry.add_judge_tick(job_key, getattr(decision, 'status', 'running'), goal=context.get('plan_goal', ''), score=score, note=note)
            recovery_recorder_registry.set_recovery_object(job_key, judge_recovery_object(decision))
            recovery_recorder_registry.set_recovery_point(job_key, judge_recovery_point(decision))
            record = recovery_recorder_registry.build(job_key)
            if record is not None:
                dump_recovery_episode(record, job_key)
    _state_provider = None
    if recovery_recorder_registry is not None and hasattr(recovery_recorder_registry, 'recent_gripper_samples'):
        _state_provider = recovery_recorder_registry.recent_gripper_samples
    _checkpoint = None
    _pause_control = None

    def _publish_local_hold_loss(event) -> None:
        """Route a contact/width hold loss directly into recovery intake."""
        note = f'local hold-loss event: stable holding latch cleared without a commanded release (width={event.width:.4f}, contact={event.grasp_contact!r})'
        if recovery_recorder_registry is not None:
            recovery_recorder_registry.add_judge_tick(event.job_key, 'failed', score=1.0, note=note, ts=event.ts)
            try:
                from ..recovery.factory import dump_recovery_episode
                record = recovery_recorder_registry.build(event.job_key)
                if record is not None:
                    dump_recovery_episode(record, event.job_key)
            except Exception:
                pass
        payload = {'status': 'failed', 'error': None, 'result': {'source': 'local_hold_loss_event', 'planner_action': 'invoke_correction', 'confidence': 1.0, 'visual_evidence': note, 'recovery_object': None}, 'meta': {'source': 'local_hold_loss_event', 'remote_judge_called': False, 'checkpoint_event': event.to_meta()}}

        def _schedule() -> None:
            task = asyncio.create_task(on_result(event.job_key, payload), name=f'local-hold-loss:{event.job_key}')
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        _event_loop.call_soon_threadsafe(_schedule)
    if judge_config.checkpoint_enabled and gello is not None and (recovery_recorder_registry is not None) and hasattr(recovery_recorder_registry, 'set_gripper_sample_hook'):
        _event_loop = asyncio.get_running_loop()
        from .checkpoint import CheckpointPauseControl, GripperTransitionMonitor
        _pause_control = CheckpointPauseControl(gello, job_lookup=job_lookup)
        _checkpoint = GripperTransitionMonitor(hold_min_m=judge_config.proprio_hold_min_m, hold_max_m=judge_config.proprio_hold_max_m, empty_max_m=judge_config.proprio_empty_max_m, stable_s=judge_config.checkpoint_stable_s, stable_band_m=judge_config.checkpoint_stable_band_m, pause_fn=_pause_control.pause if judge_config.checkpoint_pause else None, hold_loss_fn=_publish_local_hold_loss, arm_fn=lambda job_key: recovery_recorder_registry.ensure_episode(job_key), max_pauses=judge_config.checkpoint_max_pauses, emptygrasp_count=judge_config.checkpoint_emptygrasp_count)
        recovery_recorder_registry.set_gripper_sample_hook(_checkpoint.on_traj_sample)
        print(f'[checkpoint] enabled: pause={judge_config.checkpoint_pause} triage={judge_config.checkpoint_triage} release_stuck_count={judge_config.checkpoint_release_stuck_count} emptygrasp_count={judge_config.checkpoint_emptygrasp_count}', flush=True)
    judge_supervisor = VLMJudgeSupervisor(config=judge_config, on_result=on_result, scene_capture=scene_capture, on_scene_captured=_record_start_scene, on_tick=_judge_on_tick, state_provider=_state_provider, checkpoint=_checkpoint, pause_control=_pause_control)
    executor.judge_supervisor = judge_supervisor
    return (judge_supervisor, scene_capture)
