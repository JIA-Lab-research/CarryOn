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

from typing import TYPE_CHECKING, Any

from ..config import Config
from ..providers import SceneCapture
from ..types import OnResult
from .config import VLMJudgeConfig
from .supervisor import (
    VLMJudgeSupervisor,
    judge_evidence,
    judge_recovery_object,
    judge_recovery_point,
)

if TYPE_CHECKING:
    from ..executor.executor import Executor
    from ..executor.reactor import Reactor
    from ..memory.short_term import ShortTermMemory


def build_judge_supervisor(
    config: Config,
    *,
    scene_capture: SceneCapture | None,
    reactor: "Reactor",
    memory: "ShortTermMemory",
    executor: "Executor",
    recovery_recorder_registry: Any | None,
    on_result: OnResult,
) -> tuple[VLMJudgeSupervisor, SceneCapture | None]:
    """Build and wire the VLM Judge supervisor.

    Returns ``(judge_supervisor, scene_capture)``. ``scene_capture`` may be
    created here from the judge's own cameras when none was configured up-front;
    the (possibly new) value is both pushed onto ``reactor`` and returned so the
    caller can reassign its local handle (used at teardown).
    """
    judge_config = VLMJudgeConfig.from_dict(config.vlm_judge_settings)

    # If no top-level planner cameras were configured, fall back to the
    # judge's cameras so subtask-start scene capture still works.
    if scene_capture is None and judge_config.cameras:
        scene_capture = SceneCapture(list(judge_config.cameras))
        reactor.scene_capture = scene_capture

    async def _record_start_scene(_key: str, plan, subtask, snapshots: list[dict]) -> None:
        memory.record_subtask_start_scene(plan.id, subtask.id, snapshots)

    # Feed the recovery recorder a judge timeline (best-effort, sync).
    _judge_on_tick = None
    if recovery_recorder_registry is not None:
        from ..recovery.factory import dump_recovery_episode

        def _judge_on_tick(job_key, decision, context):
            score, note = judge_evidence(decision)
            recovery_recorder_registry.add_judge_tick(
                job_key,
                getattr(decision, "status", "running"),
                goal=context.get("plan_goal", ""),
                score=score,
                note=note,
            )
            # B: remember the Judge's named recovery target so the recovery layer
            # grounds SAM3 on that object instead of the whole task instruction.
            recovery_recorder_registry.set_recovery_object(
                job_key, judge_recovery_object(decision)
            )
            # ...and the coarse anchor for that object, so perception can pick the
            # correct instance among look-alikes geometrically (point-disambig).
            recovery_recorder_registry.set_recovery_point(
                job_key, judge_recovery_point(decision)
            )
            record = recovery_recorder_registry.build(job_key)
            if record is not None:
                dump_recovery_episode(record, job_key)

    _state_provider = None
    if recovery_recorder_registry is not None and hasattr(
        recovery_recorder_registry, "recent_gripper_samples"
    ):
        _state_provider = recovery_recorder_registry.recent_gripper_samples
    judge_supervisor = VLMJudgeSupervisor(
        config=judge_config,
        on_result=on_result,
        scene_capture=scene_capture,
        on_scene_captured=_record_start_scene,
        on_tick=_judge_on_tick,
        state_provider=_state_provider,
    )
    executor.judge_supervisor = judge_supervisor
    return judge_supervisor, scene_capture
