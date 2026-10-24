from __future__ import annotations
import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Optional
from ..endpoints import RobotEndpoint
from ..types import OnResult, Plan, Subtask
from ..providers import SceneCapture
from .client import JudgeDecision, VLMJudgeClient, build_vlm_judge_client
from .config import VLMJudgeConfig
from .success_confirmation import PlacementSuccessGate, PlacementSuccessObservation, is_placement_instruction
SceneCallback = Callable[[str, Plan, Subtask, list[dict]], Awaitable[None]]
TickCallback = Callable[[str, Any, dict], None]

def judge_evidence(decision: Any) -> tuple[Optional[float], Optional[str]]:
    """Pull (confidence, visual_evidence) out of a judge decision, if present."""
    result = getattr(decision, 'result', None)
    if not isinstance(result, dict):
        return (None, None)
    conf = result.get('confidence')
    try:
        score = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        score = None
    evidence = result.get('visual_evidence')
    note = str(evidence) if evidence is not None else None
    return (score, note)

def judge_recovery_object(decision: Any) -> Optional[str]:
    """Pull ``result.recovery_object`` (the object the Judge says recovery should
    re-grasp) out of a judge decision. Returns a trimmed noun phrase or ``None``
    when the field is absent/empty/null -- the recovery layer then falls back to
    the configured hint / subtask instruction."""
    result = getattr(decision, 'result', None)
    if not isinstance(result, dict):
        return None
    val = result.get('recovery_object')
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
    result = getattr(decision, 'result', None)
    if not isinstance(result, dict):
        return (None, None)
    obj = result.get('placed_object')
    if not isinstance(obj, str) or not obj.strip():
        obj = None
    match = result.get('placed_matches_instruction')
    if not isinstance(match, bool):
        match = None
    return (obj, match)

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
    result = getattr(decision, 'result', None)
    if not isinstance(result, dict):
        return None
    val = result.get('recovery_point')
    if val is None:
        return None
    try:
        x, y = (float(val[0]), float(val[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if x > 1.0 or y > 1.0:
        x, y = (x / 1000.0, y / 1000.0)
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    return (x, y)

def scan_drop_state(samples, held_samples: int, last_hold_ts, cfg, pause_overlap_fn=None):
    """Fold timestamped gripper widths into the drop-lane state.

    Return (held_samples, last_hold_ts, fire). In-band samples advance the holding
    count and reset the grace countdown. Fire after enough holding samples, an
    out-of-band latest width and the configured grace interval. The optional
    pause_overlap_fn excludes checkpoint pauses from elapsed time."""
    if not samples:
        return (held_samples, last_hold_ts, False)
    for ts, w in samples:
        if cfg.proprio_hold_min_m <= w <= cfg.proprio_hold_max_m:
            held_samples += 1
            if last_hold_ts is None or ts > last_hold_ts:
                last_hold_ts = ts
    now_ts, now_w = samples[-1]
    holding_now = cfg.proprio_hold_min_m <= now_w <= cfg.proprio_hold_max_m
    paused_s = 0.0
    if pause_overlap_fn is not None and last_hold_ts is not None:
        try:
            paused_s = float(pause_overlap_fn(last_hold_ts, now_ts) or 0.0)
        except Exception:
            paused_s = 0.0
    closed_on_air = now_w <= cfg.proprio_empty_max_m
    fire = not holding_now and closed_on_air and (held_samples >= cfg.proprio_hold_min_samples) and (last_hold_ts is not None) and (now_ts - last_hold_ts - paused_s >= cfg.proprio_grace_s)
    return (held_samples, last_hold_ts, fire)
_CHECKPOINT_GUIDANCE = {'drop': 'EVENT CHECKPOINT (drop): the gripper transitioned HOLDING -> CLOSED-ON-AIR {age:.1f}s ago -- the held object was dropped. The robot is {paused}. Triage this exact snapshot. (1) If the dropped target object is visible, upright and reachable on the work surface and the scene is otherwise intact, return running -- the policy will resume and may recover by itself. (2) If the situation is clearly unrecoverable or about to degrade (target out of reach or out of view, wedged/fallen badly, or the arm is moving toward a WRONG look-alike object), return failed to invoke correction NOW while the trajectory is uncontaminated. Do not return success at a drop.', 'release': 'EVENT CHECKPOINT (release): the gripper transitioned HOLDING -> FULLY-OPEN {age:.1f}s ago -- a deliberate release. The robot is {paused}. Give a three-way verdict on this exact snapshot. (1) If the release completed the instruction (the target object visibly inside the destination), return success with the placed_object identity attestation. (2) If the release MIS-placed the object (outside/beside the destination, dropped back on the table), return failed NOW -- correction is cheapest at this instant. (3) If this is an in-progress adjustment (repositioning for a re-grasp), return running.', 'empty_grasp': 'EVENT CHECKPOINT (empty grasp): the gripper has now closed onto AIR repeatedly without ever holding an object ({age:.1f}s since the latest empty close). The robot is {paused}. The policy is failing to pick the target up (mis-aimed grasps). Triage this exact snapshot: (1) if the target object is visible and reachable and the attempts look like they are converging (grasps landing closer), return running to let the policy keep trying; (2) if the attempts are clearly not converging, the target is out of reach/view, or the arm keeps pinching the same empty spot, return failed to invoke correction NOW. Do not return success at an empty grasp.'}
_REHOLD_GUIDANCE = "RE-HOLD IDENTITY CHECK: after an earlier drop the gripper is HOLDING again. Robots often re-grasp a look-alike after a drop, and a real held object masks the failure. Verify from the pixels WHICH object the gripper holds now: if it is NOT the instruction's target object, return failed immediately (wrong-object re-grasp). If it IS the target, return running (self-heal in progress)."

class VLMJudgeSupervisor:
    """Poll a VLM judge and publish only terminal decisions to JobManager.

    Optionally captures a multi-camera snapshot when each job starts and
    invokes ``on_scene_captured`` with the snapshots; the wiring in
    :mod:`carryon.main` routes these into :meth:`Memory.record_subtask_start_scene`
    so the Reactor can build before/after image context for failure replan.
    """

    def __init__(self, *, config: VLMJudgeConfig, on_result: OnResult, client: VLMJudgeClient | None=None, scene_capture: SceneCapture | None=None, on_scene_captured: Optional[SceneCallback]=None, on_tick: Optional[TickCallback]=None, state_provider=None, checkpoint=None, pause_control=None) -> None:
        self.config = config
        self.on_result = on_result
        self.client = client or build_vlm_judge_client(config)
        self.scene_capture = scene_capture
        self.on_scene_captured = on_scene_captured
        self.on_tick = on_tick
        self.state_provider = state_provider
        self.checkpoint = checkpoint
        self.pause_control = pause_control
        self._tasks: set[asyncio.Task] = set()

    def start_job(self, *, job_key: str, plan: Plan, subtask: Subtask, endpoint: RobotEndpoint, instruction: str, inputs: dict, vla_job_id: str | None, deadline_s: float, post_handback: bool=False) -> None:
        context = {'job_key': job_key, 'vla_job_id': vla_job_id, 'plan_id': plan.id, 'plan_goal': plan.goal, 'subtask_id': subtask.id, 'endpoint': {'name': endpoint.name, 'transport': endpoint.transport, 'address': endpoint.address, 'description': endpoint.description, 'tags': list(endpoint.tags)}, 'instruction': instruction, 'inputs': inputs or {}, 'deadline_s': deadline_s, 'cameras': self.config.cameras, 'system_prompt': self.config.system_prompt}
        task = asyncio.create_task(self._watch(job_key, context, plan=plan, subtask=subtask, post_handback=post_handback), name=f'vlm-judge:{job_key}')
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.client.aclose()

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
            await self.on_scene_captured(subtask.id, plan, subtask, [s.to_dict() for s in snapshots])
        except Exception:
            pass

    def _checkpoint_resume(self, job_key: str, context: dict[str, Any]) -> dict[str, Any]:
        """Un-freeze the VLA after a checkpoint pause (best-effort, sync HTTP).

        Resumes under the same request_id with skip_reset, mirroring the
        recovery handback, so the episode continues from the current pose.
        Never raises; a failed resume leaves the world paused and the failed
        state visible in the checkpoint meta (the recovery takeover that
        follows a failed verdict is idempotent on a paused world).
        """
        ckpt = self.checkpoint
        if ckpt is None or not ckpt.is_paused() or self.pause_control is None:
            return {}
        ok = False
        try:
            ok = bool(self.pause_control.resume(job_key, str(context.get('instruction') or '')))
        except Exception:
            ok = False
        if ok:
            ckpt.note_resumed()
        else:
            ckpt.note_resume_failed()
        timing = getattr(self.pause_control, 'resume_timing', None)
        return timing(job_key) if callable(timing) else {}

    async def _watch(self, job_key: str, context: dict[str, Any], *, plan: Plan, subtask: Subtask, post_handback: bool=False) -> None:
        ckpt = self.checkpoint if self.checkpoint is not None and getattr(self.config, 'checkpoint_enabled', False) else None
        if ckpt is not None:
            _grace = int(getattr(self.config, 'checkpoint_handback_grace_samples', 0)) if post_handback else 0
            ckpt.arm(job_key, grace_samples=_grace)
        try:
            await self._watch_inner(job_key, context, plan=plan, subtask=subtask, ckpt=ckpt)
        finally:
            if ckpt is not None:
                ckpt.disarm()

    async def _watch_inner(self, job_key: str, context: dict[str, Any], *, plan: Plan, subtask: Subtask, ckpt=None) -> None:
        await self._capture_start_scene(plan, subtask)
        polls = 0
        success_streak = 0
        failure_streak = 0
        no_progress_polls = 0
        held_samples = 0
        last_hold_ts = None
        proprio_drop_fired = False
        drop_sig_held = False
        drop_sig_ts = None
        placement_success_gate = PlacementSuccessGate(min_settle_interval_s=self.config.success_settle_interval_s) if is_placement_instruction(str(context.get('instruction') or '')) else None
        start_ts = time.monotonic()
        start_wall = time.time()
        first_credible_failure_wall: Optional[float] = None
        recent: list[dict[str, Any]] = []
        while True:
            resume_timing: dict[str, Any] = {}
            placement_gate_decision = None
            placement_observation_trace = None
            if self.config.max_polls and polls >= self.config.max_polls:
                await self.on_result(job_key, {'status': 'failed', 'error': 'VLM judge reached max_polls without terminal decision', 'result': {'source': 'vlm_judge'}})
                return
            polls += 1
            _paused_s = ckpt.paused_overlap(start_wall, time.time()) if ckpt is not None else 0.0
            elapsed_s = round(max(0.0, time.monotonic() - start_ts - _paused_s), 2)
            poll_context: dict[str, Any] = {**context, 'poll_index': polls, 'elapsed_s': elapsed_s}
            if self.config.stuck_timeout_s > 0:
                poll_context['stuck_timeout_s'] = self.config.stuck_timeout_s
            if recent:
                poll_context['recent_observations'] = recent
            ckpt_event = None
            ckpt_rehold = False
            if ckpt is not None:
                _ev = ckpt.pending_event()
                if _ev is not None and (not self.config.checkpoint_triage):
                    ckpt.consume(_ev)
                    resume_timing = self._checkpoint_resume(job_key, context)
                    _ev = None
                if _ev is not None:
                    ckpt_event = _ev
                    _age = max(0.0, min(999.0, time.time() - _ev.ts))
                    _paused_note = 'PAUSED on this exact frame' if ckpt.is_paused() else 'still moving (pause unavailable)'
                    poll_context['checkpoint_event'] = {'kind': _ev.kind, 'age_s': round(_age, 2), 'paused': ckpt.is_paused(), 'release_count': ckpt.release_count(), 'guidance': _CHECKPOINT_GUIDANCE[_ev.kind].format(age=_age, paused=_paused_note)}
                elif ckpt.rehold_pending():
                    ckpt_rehold = True
                    poll_context['checkpoint_rehold'] = {'guidance': _REHOLD_GUIDANCE}
            proprio_drop = False
            if self.state_provider is not None:
                try:
                    samples = self.state_provider() or []
                except Exception:
                    samples = []
                if samples:
                    widths = [w for _, w in samples]
                    poll_context['proprioception'] = {'gripper_width_m': round(float(widths[-1]), 4), 'recent_widths_m': [round(float(w), 4) for w in widths[-6:]], 'note': 'measured jaw opening: >=0.008 = holding an object; <0.005 = jaws closed on air (object lost); ~0.08 = fully open'}
                    for _ts, _w in samples:
                        if self.config.proprio_hold_min_m <= _w <= self.config.proprio_hold_max_m:
                            drop_sig_held = True
                        elif drop_sig_held and drop_sig_ts is None and (_w < 0.005):
                            drop_sig_ts = _ts
                    if drop_sig_ts is not None:
                        poll_context['proprioception']['drop_alert'] = "ALERT: the gripper transitioned from HOLDING to CLOSED-ON-AIR earlier in this episode (the held object slipped or was dropped). Even if the gripper is holding an object NOW, do NOT assume it is the instructed object: robots often pick a look-alike after a drop. Verify the identity of the held object and the dropped object's location before returning running or success."
                    if self.config.proprio_drop_enabled and (not proprio_drop_fired):
                        held_samples, last_hold_ts, proprio_drop = scan_drop_state(samples, held_samples, last_hold_ts, self.config, pause_overlap_fn=ckpt.paused_overlap if ckpt is not None else None)
            if ckpt is not None and callable(getattr(ckpt, 'local_terminal_published', None)) and ckpt.local_terminal_published(job_key):
                return
            failure_wall_cap_forced = False
            judge_task = asyncio.create_task(self.client.judge(poll_context))
            while True:
                wait_s = 0.05
                if first_credible_failure_wall is not None and self.config.failure_takeover_wall_cap_s > 0:
                    remaining = self.config.failure_takeover_wall_cap_s - (time.monotonic() - first_credible_failure_wall)
                    if remaining <= 0:
                        judge_task.cancel()
                        await asyncio.gather(judge_task, return_exceptions=True)
                        failure_wall_cap_forced = True
                        decision = JudgeDecision(status='failed', result={'planner_action': 'invoke_correction', 'confidence': 1.0, 'visual_evidence': 'forced takeover wall cap reached after the first credible failed/stuck observation', 'stuck_evidence': 'failure_takeover_wall_cap'})
                        break
                    wait_s = min(wait_s, remaining)
                done, _pending = await asyncio.wait({judge_task}, timeout=wait_s)
                if done:
                    decision = judge_task.result()
                    break
                if ckpt is not None and callable(getattr(ckpt, 'local_terminal_published', None)) and ckpt.local_terminal_published(job_key):
                    judge_task.cancel()
                    await asyncio.gather(judge_task, return_exceptions=True)
                    return
            _verdict_paused_s = ckpt.paused_overlap(start_wall, time.time()) if ckpt is not None else 0.0
            verdict_elapsed_s = round(max(0.0, time.monotonic() - start_ts - _verdict_paused_s), 2)
            if proprio_drop:
                proprio_drop_fired = True
                decision.status = 'failed'
                decision.error = None
                decision.result = {'planner_action': 'invoke_correction', 'confidence': 1.0, 'visual_evidence': f"proprioceptive drop signature: gripper was holding, then stayed empty past the self-heal grace window (width={poll_context['proprioception']['gripper_width_m']} m, grace={self.config.proprio_grace_s}s)"}
            score, evidence = judge_evidence(decision)
            if decision.status == 'success' and getattr(self.config, 'success_requires_placed_match', False):
                placed_obj, placed_match = judge_placed_match(decision)
                if not (placed_obj is not None and placed_match is True):
                    decision.status = 'running'
                    note = ' [demoted: success lacked placed_object/placed_matches_instruction identity attestation]'
                    evidence = (evidence or '') + note
                    if isinstance(decision.result, dict):
                        decision.result['visual_evidence'] = str(decision.result.get('visual_evidence') or '') + note
            endpoint_gate_t0 = time.perf_counter()
            if placement_success_gate is not None:
                result_obj = decision.result if isinstance(decision.result, dict) else {}
                observation_meta = result_obj.get('_observation')
                if not isinstance(observation_meta, dict):
                    observation_meta = {}
                placement_gate_decision = placement_success_gate.observe(PlacementSuccessObservation(frame_identity=str(observation_meta.get('frame_identity') or ''), captured_at_s=float(observation_meta.get('captured_at_s') or 0.0), producer_frame_seq=observation_meta.get('producer_frame_seq'), producer_frame_ts=observation_meta.get('producer_frame_ts'), simulation_step_id=observation_meta.get('simulation_step_id'), simulation_time_s=observation_meta.get('simulation_time_s'), camera_role=str(observation_meta.get('camera_role') or ''), camera_source=str(observation_meta.get('camera_source') or ''), judge_success=decision.status == 'success', target_released=result_obj.get('target_released') is True, target_in_destination=result_obj.get('target_in_destination') is True, stable=result_obj.get('placement_stable') is True, job_terminal=False))
                result_obj['endpoint_confirmation'] = {'provisional_success': placement_gate_decision.provisional, 'finalize': placement_gate_decision.finalize, 'clear_provisional': placement_gate_decision.clear_provisional, 'reason': placement_gate_decision.reason}
                placement_observation_trace = {'job_key': job_key, 'poll': polls, 'frame_identity': str(observation_meta.get('frame_identity') or ''), 'captured_at_s': float(observation_meta.get('captured_at_s') or 0.0), 'pixel_sha256': str(observation_meta.get('pixel_sha256') or ''), 'producer_frame_seq': observation_meta.get('producer_frame_seq'), 'producer_frame_ts': observation_meta.get('producer_frame_ts'), 'simulation_step_id': observation_meta.get('simulation_step_id'), 'simulation_time_s': observation_meta.get('simulation_time_s'), 'camera_role': str(observation_meta.get('camera_role') or ''), 'camera_source': str(observation_meta.get('camera_source') or ''), 'target_released': result_obj.get('target_released') is True, 'target_in_destination': result_obj.get('target_in_destination') is True, 'placement_stable': result_obj.get('placement_stable') is True, 'provisional_success': placement_gate_decision.provisional, 'clear_provisional': placement_gate_decision.clear_provisional, 'finalize': placement_gate_decision.finalize, 'reason': placement_gate_decision.reason}
                decision.result = result_obj
                if decision.status == 'success' and (not (placement_gate_decision.provisional or placement_gate_decision.finalize)):
                    decision.status = 'running'
                elif placement_gate_decision.clear_provisional and decision.status != 'success' and (not proprio_drop):
                    decision.status = 'running'
                    decision.error = None
            endpoint_gate_ms = (time.perf_counter() - endpoint_gate_t0) * 1000.0
            ckpt_fastlane = False
            if ckpt is not None and decision.status == 'running' and (self.config.checkpoint_release_stuck_count > 0) and (ckpt.release_count() >= self.config.checkpoint_release_stuck_count):
                decision.status = 'failed'
                decision.error = None
                decision.result = {'planner_action': 'invoke_correction', 'confidence': 1.0, 'visual_evidence': evidence or '', 'stuck_evidence': f'checkpoint release-oscillation: {ckpt.release_count()} release events without task completion'}
                ckpt_fastlane = True
            if decision.status == 'failed' and (ckpt_event is not None or ckpt_rehold):
                ckpt_fastlane = True
            forced_stuck = False
            if decision.status == 'running' and decision.error is not None:
                pass
            elif decision.status == 'running' and self.config.stuck_timeout_s > 0 and (self.config.stuck_force_failed_polls > 0) and (elapsed_s >= self.config.stuck_timeout_s):
                no_progress_polls += 1
                if no_progress_polls >= self.config.stuck_force_failed_polls:
                    forced_stuck = True
                    decision.status = 'failed'
                    decision.result = {'planner_action': 'invoke_correction', 'confidence': 1.0, 'visual_evidence': evidence or 'no visible task progress after stuck timeout', 'stuck_evidence': f'forced stuck after {no_progress_polls} running polls at elapsed_s={elapsed_s} >= stuck_timeout_s={self.config.stuck_timeout_s}'}
                    decision.error = None
            else:
                no_progress_polls = 0
            if first_credible_failure_wall is None and decision.status == 'failed' and (decision.error is None):
                first_credible_failure_wall = time.monotonic()
            if first_credible_failure_wall is not None and self.config.failure_takeover_wall_cap_s > 0 and (time.monotonic() - first_credible_failure_wall >= self.config.failure_takeover_wall_cap_s):
                failure_wall_cap_forced = True
                decision.status = 'failed'
                decision.error = None
                decision.result = {'planner_action': 'invoke_correction', 'confidence': 1.0, 'visual_evidence': 'forced takeover wall cap reached after the first credible failed/stuck observation', 'stuck_evidence': 'failure_takeover_wall_cap'}
            if self.config.recent_observations_window > 0:
                recent.append({'poll': polls, 'elapsed_s': elapsed_s, 'status': decision.status, 'visual_evidence': evidence})
                if len(recent) > self.config.recent_observations_window:
                    del recent[:-self.config.recent_observations_window]
            if self.on_tick is not None:
                try:
                    self.on_tick(job_key, decision, context)
                except Exception:
                    pass
            should_publish = False
            if decision.status == 'running':
                success_streak = 0
                failure_streak = 0
            elif decision.status == 'success':
                success_streak += 1
                failure_streak = 0
                if placement_success_gate is not None:
                    should_publish = bool(placement_gate_decision is not None and placement_gate_decision.finalize)
                else:
                    should_publish = success_streak >= self.config.success_streak_threshold
            elif decision.status == 'failed':
                failure_streak += 1
                success_streak = 0
                should_publish = proprio_drop or forced_stuck or failure_wall_cap_forced or (failure_streak >= self.config.failure_streak_threshold)
                if should_publish and (not proprio_drop) and (not ckpt_fastlane) and (not failure_wall_cap_forced) and (self.config.min_fail_elapsed_s > 0) and (verdict_elapsed_s < self.config.min_fail_elapsed_s):
                    should_publish = False
            elif decision.status == 'cancelled':
                success_streak = 0
                failure_streak = 0
                should_publish = True
            if ckpt is not None and decision.status == 'running':
                if ckpt_event is not None:
                    ckpt.consume(ckpt_event)
                    resume_timing = self._checkpoint_resume(job_key, context)
                if ckpt_rehold:
                    ckpt.clear_rehold()
            wall_telemetry: dict[str, Any] = {'image_fetch_start_wall_ts': None, 'image_fetch_end_wall_ts': None, 'image_fetch_ms': None, 'judge_api_request_wall_ts': None, 'judge_api_response_wall_ts': None, 'judge_api_ms': None, 'parse_ms': None}
            if isinstance(decision.result, dict):
                source_timing = decision.result.get('_wall_clock_telemetry')
                if isinstance(source_timing, dict):
                    wall_telemetry.update(source_timing)
            wall_telemetry['endpoint_gate_ms'] = endpoint_gate_ms
            wall_telemetry['parse_endpoint_gate_ms'] = float(wall_telemetry.get('parse_ms') or 0.0) + endpoint_gate_ms
            if ckpt_event is not None:
                wall_telemetry.update({'checkpoint_event_wall_ts': ckpt_event.checkpoint_event_wall_ts, 'pause_request_wall_ts': ckpt_event.pause_request_wall_ts, 'pause_ack_wall_ts': ckpt_event.pause_ack_wall_ts})
            else:
                wall_telemetry.update({'checkpoint_event_wall_ts': None, 'pause_request_wall_ts': None, 'pause_ack_wall_ts': None})
            wall_telemetry.update(resume_timing)
            first_action = wall_telemetry.get('first_resumed_action_wall_ts')
            checkpoint_wall = wall_telemetry.get('checkpoint_event_wall_ts')
            wall_telemetry['total_checkpoint_pause_ms'] = max(0.0, (float(first_action) - float(checkpoint_wall)) * 1000.0) if first_action is not None and checkpoint_wall is not None else None
            recorder = getattr(self.client, 'record_poll_telemetry', None)
            if callable(recorder):
                recorder(poll_context, wall_telemetry)
            print('[judge-wall-clock] ' + json.dumps(wall_telemetry, sort_keys=True), flush=True)
            if placement_observation_trace is not None:
                print('[endpoint-confirmation] ' + json.dumps(placement_observation_trace, sort_keys=True), flush=True)
            print('[judge] %s poll=%d observed=%.1fs verdict=%.1fs status=%s%s streaks(s=%d,f=%d)%s%s%s%s%s' % (job_key, polls, elapsed_s, verdict_elapsed_s, decision.status, ' error=%r' % (decision.error,) if decision.error else '', success_streak, failure_streak, ' FORCED_STUCK' if forced_stuck else '', ' WALL_CAP' if failure_wall_cap_forced else '', ' PROPRIO_DROP' if proprio_drop else '', ' CKPT_FASTLANE' if ckpt_fastlane else '', ' publish' if should_publish else ''), flush=True)
            if should_publish:
                if ckpt is not None:
                    ckpt.disarm()
                    if decision.status == 'success':
                        resume_timing = self._checkpoint_resume(job_key, context)
                        if resume_timing:
                            wall_telemetry.update(resume_timing)
                            first_action = wall_telemetry.get('first_resumed_action_wall_ts')
                            checkpoint_wall = wall_telemetry.get('checkpoint_event_wall_ts')
                            wall_telemetry['total_checkpoint_pause_ms'] = max(0.0, (float(first_action) - float(checkpoint_wall)) * 1000.0) if first_action is not None and checkpoint_wall is not None else None
                            if callable(recorder):
                                recorder(poll_context, wall_telemetry)
                payload: dict[str, Any] = {'status': decision.status, 'result': decision.result or {'source': 'vlm_judge'}, 'error': decision.error, 'meta': {'source': 'vlm_judge', 'polls': polls, 'success_streak': success_streak, 'failure_streak': failure_streak, 'forced_stuck': forced_stuck, 'failure_wall_cap_forced': failure_wall_cap_forced, 'failure_takeover_wall_cap_s': self.config.failure_takeover_wall_cap_s, 'observation_elapsed_s': elapsed_s, 'verdict_elapsed_s': verdict_elapsed_s}}
                if placement_gate_decision is not None:
                    payload['meta']['endpoint_confirmation'] = {'finalize': placement_gate_decision.finalize, 'reason': placement_gate_decision.reason}
                if ckpt is not None:
                    payload['meta']['checkpoint'] = {'events': ckpt.event_summary(), 'paused_at_publish': ckpt.is_paused(), 'fastlane': ckpt_fastlane}
                if payload['error'] is None:
                    payload.pop('error')
                await self.on_result(job_key, payload)
                return
            if self.config.poll_interval_s > 0:
                await asyncio.sleep(self.config.poll_interval_s)
