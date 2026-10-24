"""Coordinate recovery takeover, policy execution and control handback.

The arbiter transfers control and the registered policy invokes backend
primitives. Control is returned after recovery or failure handling."""
from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import Optional, Protocol
from .policy import RecoveryConfig, RecoveryContext, RecoveryPolicyRegistry, RecoveryStep, SequentialPolicy, StepResult
from .primitives import select_rewind_target, select_rewind_targets
from .schema import EpisodeRecord, HandbackMode, RecoveryInvocation, RecoveryOutcome

class ControlArbiter(Protocol):
    """Hands robot control between the VLA endpoint and the recovery backend.

    A real impl wraps :class:`GelloControlClient`: ``takeover`` = ``cancel(...,
    standby=True)`` (brake + self-hold + yield the command channel); ``handback``
    = ``execute(..., skip_reset=True)`` (VLA resumes from the current pose)."""

    def takeover(self) -> bool:
        ...

    def handback(self) -> bool:
        ...

@dataclass
class RecoveryResult:
    invocation: RecoveryInvocation
    steps: list[StepResult]

    @property
    def ok(self) -> bool:
        return self.invocation.outcome == 'success'

class RecoveryOrchestrator:

    def __init__(self, backend, registry: RecoveryPolicyRegistry, *, config: Optional[RecoveryConfig]=None, arbiter: Optional[ControlArbiter]=None, log=None) -> None:
        self.backend = backend
        self.registry = registry
        self.cfg = config or RecoveryConfig()
        self.arbiter = arbiter
        self._log = log

    @staticmethod
    def _append_terminal_step(result: RecoveryResult, step: StepResult) -> None:
        """Append a post-policy control event to runtime and invocation records."""
        result.steps.append(step)
        result.invocation.steps = list(result.invocation.steps or [])
        result.invocation.steps.append(step.to_dict())

    def _park_at_failure_handback_standoff(self, ctx, result) -> None:
        """Release the object at the final destination and hand control back."""
        destination = 'candidate_pregrasp' if ctx.grasp is not None else 'pre_failure'
        width = getattr(self.backend, '_last_gripper_cmd_width', None)
        detail = {'destination': destination, 'command_width_m': width}
        if width is None or not math.isfinite(float(width)):
            self._append_terminal_step(result, StepResult('failure_terminal_destination', False, 'last commanded gripper width unavailable', data=detail))
            return
        if ctx.grasp is not None:
            position = list(ctx.grasp.position)
            position[2] += float(ctx.config.pre_grasp_dis)
            quaternion = list(ctx.grasp.quaternion)
            detail.update(position=position, quaternion_wxyz=quaternion)
            move = lambda: self.backend.goto_pose(position, quaternion, z_approach=0.0, verify=True, gripper_width=float(width))
        else:
            if ctx.rewind_target_q is None:
                self._append_terminal_step(result, StepResult('failure_terminal_destination', False, 'recorded pre-failure pose unavailable', data=detail))
                return
            detail['q_target'] = list(ctx.rewind_target_q)
            move = lambda: self.backend.rewind_to_pose(ctx.rewind_target_q, preserve_last_gripper_command=True)
        start = time.monotonic()
        motion = move()
        self._append_terminal_step(result, StepResult('failure_terminal_destination', bool(motion.ok), getattr(motion, 'reason', None), dt_s=time.monotonic() - start, data=detail))
        if not motion.ok:
            return
        start = time.monotonic()
        opened = self.backend.open_gripper()
        self._append_terminal_step(result, StepResult('failure_handback_open_gripper', bool(opened.ok), getattr(opened, 'reason', None), dt_s=time.monotonic() - start))

    def _handback_failed_recovery(self, ctx: RecoveryContext, result: RecoveryResult) -> bool:
        """Return control after a recovery failure without ending the episode."""
        try:
            self._park_at_failure_handback_standoff(ctx, result)
        except Exception as exc:
            self._append_terminal_step(result, StepResult('failure_handback_parking_exception', False, f'{type(exc).__name__}: {exc}', dt_s=0.0))
        self._emit('recovery terminal failure -> handing control back to the VLA with the remaining horizon')
        ok = False
        reason = None
        try:
            ok = bool(self.arbiter.handback())
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            self._emit(f'HANDBACK RAISED: {reason}')
        if reason is None:
            reason = getattr(self.arbiter, 'last_handback_reason', None)
        self._append_terminal_step(result, StepResult('failure_handback_ack', ok, 'VLA resume acknowledged' if ok else str(reason or 'refused'), dt_s=0.0))
        if not ok:
            result.invocation.notes = (result.invocation.notes + ' | ' if result.invocation.notes else '') + 'handback_failed: ' + str(reason or 'refused')
            self._emit(f'HANDBACK FAILED after recovery failure; VLA was not resumed (reason={reason})')
        return ok

    def _load_scene_reference(self, record: EpisodeRecord) -> None:
        setattr(self.backend, 'scene_reference', _load_reference_frame(record))

    def _emit(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:
                pass

    def recover(self, record: EpisodeRecord, *, object_hint: str, target_px: Optional[tuple]=None, policy_name: str='recover_hold_lost', policy_version: Optional[str]=None, failure_type: str='hold_lost', triggered_ts: Optional[float]=None, request_id: Optional[str]=None, reset_identity: bool=True) -> RecoveryResult:
        ts = triggered_ts if triggered_ts is not None else time.time()
        if reset_identity:
            try:
                self.backend.identity_rejected = []
            except Exception:
                pass

        def _abort(name: str, version: str, notes: str) -> RecoveryResult:
            return RecoveryResult(_invocation(ts, failure_type, name, version, None, outcome='aborted', notes=notes, request_id=request_id), [])
        policy = self.registry.get(policy_name, policy_version)
        if policy is None:
            return _abort(policy_name, policy_version or '?', f"policy not found: {policy_name}@{policy_version or 'latest'}")
        target = select_rewind_target(record, lead_s=self.cfg.rewind_lead_s)
        if target is None:
            return _abort(policy.name, policy.version, 'no rewind target (missing first_fail_ts or empty trajectory)')
        ctx = RecoveryContext(object_hint=object_hint, config=self.cfg, rewind_target_q=list(target.q), rewind_target_ts=target.ts, rewind_fallback_qs=[list(t.q) for t in select_rewind_targets(record, lead_s=self.cfg.rewind_lead_s)[1:]], target_xy_world=_prefail_ee_xy(record), target_px=tuple(target_px) if target_px is not None else None)
        try:
            self._load_scene_reference(record)
            ref = getattr(self.backend, 'scene_reference', None)
            n_frames = len(getattr(record, 'frames', None) or [])
            self._emit(f"scene reference: {('loaded' if ref is not None else 'UNAVAILABLE')} (record has {n_frames} frame refs)")
        except Exception as e:
            setattr(self.backend, 'scene_reference', None)
            self._emit(f'scene reference load raised: {type(e).__name__}: {e}')
        took_over = False
        if self.arbiter is not None:
            try:
                took_over = bool(self.arbiter.takeover())
            except Exception as e:
                return _abort(policy.name, policy.version, f'takeover raised: {e}')
            if not took_over:
                self._emit('takeover REFUSED (VLA did not yield); not moving the arm')
                return _abort(policy.name, policy.version, 'arbiter refused takeover')
            self._emit('takeover ok (VLA in standby); recovery now drives the arm')
        result: Optional[RecoveryResult] = None
        steps: list[StepResult] = []
        try:
            self._emit(f"running {policy.name}@{policy.version} ({len(getattr(policy, 'steps', []))} steps): the arm WILL move (retreat to init -> regrasp -> return to pre-failure pose). Do not interrupt.")
            steps = policy.run(self.backend, ctx, on_step=lambda r: self._emit(f"  {r.op}={('ok' if r.ok else 'FAIL')} ({r.dt_s:.1f}s)" + (f' {r.reason}' if r.reason else '')))
            policy_ran_all = len(steps) == len(getattr(policy, 'steps', steps)) and all((s.ok for s in steps))
            outcome: RecoveryOutcome = 'success' if policy_ran_all else 'failed'
            notes = _summarize(steps)
            inv = _invocation(ts, policy.failure_type, policy.name, policy.version, ctx, outcome=outcome, notes=notes, handback_mode=policy.handback_mode, request_id=request_id)
            inv.steps = [s.to_dict() for s in steps]
            result = RecoveryResult(inv, steps)
        except Exception as exc:
            step = StepResult('recovery_exception', False, f'{type(exc).__name__}: {exc}', dt_s=0.0)
            steps.append(step)
            inv = _invocation(ts, policy.failure_type, policy.name, policy.version, ctx, outcome='failed', notes=_summarize(steps), handback_mode=policy.handback_mode, request_id=request_id)
            inv.steps = [s.to_dict() for s in steps]
            result = RecoveryResult(inv, steps)
            self._emit(f'recovery policy raised; converting to terminal failure before mandatory handback: {type(exc).__name__}: {exc}')
        finally:
            if took_over and self.arbiter is not None:
                if result is not None and result.ok:
                    if result.invocation.handback_mode == 'complete':
                        self._emit('complete-mode succeeded; keeping VLA in standby until terminal verification')
                    else:
                        self._emit('handing control back to the VLA (resume from current pose)')
                        _hb_ok = False
                        try:
                            _hb_ok = bool(self.arbiter.handback())
                        except Exception as _hb_e:
                            self._emit('HANDBACK RAISED: %s: %s' % (type(_hb_e).__name__, _hb_e))
                        inv = result.invocation
                        inv.steps = list(inv.steps or [])
                        _hb_reason = getattr(self.arbiter, 'last_handback_reason', None)
                        inv.steps.append({'op': 'handback_ack', 'ok': _hb_ok, 'reason': 'VLA resume acknowledged' if _hb_ok else str(_hb_reason or 'refused'), 'dt_s': 0.0})
                        if not _hb_ok:
                            self._emit('HANDBACK FAILED after successful recovery (reason=%s); VLA NOT resumed, arm in standby -- demoting recovery outcome to failed' % (_hb_reason,))
                            inv.outcome = 'failed'
                            inv.notes = (inv.notes + ' | ' if inv.notes else '') + 'handback_failed: ' + str(_hb_reason or 'refused')
                else:
                    assert result is not None
                    self._handback_failed_recovery(ctx, result)
        postrow_steps = []
        for op, hook_name, schema in (('fixv2_postrow_parameter_memory', 'reflect_parameter_memory_at_recovery_end', 'carryon-fixv2-parameter-postrow-evidence/v1'), ('fixv2_postrow_semantic_memory', 'reflect_semantic_memory_at_recovery_end', 'carryon-fixv2-semantic-postrow-evidence/v1')):
            try:
                debug = getattr(self.backend, hook_name)(steps)
            except Exception as exc:
                debug = {'schema': schema, 'triggered': False, 'write_succeeded': False, 'vlm_call_attempted': False, 'current_action_unchanged': True, 'fail_open_error': f'{type(exc).__name__}: {exc}'}
            postrow_steps.append(StepResult(op=op, ok=True, reason='post-row mechanism record; current action/outcome unchanged', data=debug))
        steps.extend(postrow_steps)
        result.steps = steps
        result.invocation.steps = list(result.invocation.steps or []) + [item.to_dict() for item in postrow_steps]
        record.recovery = result.invocation
        return result

def _load_reference_frame(record: EpisodeRecord):
    """(rgb, depth_m) of the episode's first recorded external frame, or None."""
    frames = getattr(record, 'frames', None) or []
    if not frames:
        return None
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    first = min(frames, key=lambda f: getattr(f, 'ts', 0.0))
    rgb_path = getattr(first, 'rgb_path', None)
    depth_path = getattr(first, 'depth_path', None)
    if not rgb_path or not depth_path:
        return None
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_raw is None:
        return None
    scale = 0.001
    for cam in getattr(record, 'cameras', None) or []:
        ds = cam.get('depth_scale') if isinstance(cam, dict) else getattr(cam, 'depth_scale', None)
        if ds:
            scale = float(ds)
            break
    depth_m = depth_raw.astype(np.float32) * scale
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return (rgb, depth_m)

def _prefail_ee_xy(record: EpisodeRecord) -> Optional[tuple]:
    """The VLA's end-effector XY (world) at/just-before the failure -- the
    spatial-disambiguation prior. The VLA had driven the gripper toward the
    target object before the disturbance froze it, so this XY points at the
    object the operator/policy actually wanted, letting perception resolve a
    generic mask ("can") to the right instance among look-alikes.

    Returns ``None`` when the trajectory ring buffer or EE samples are missing.
    """
    traj = getattr(record, 'trajectory', None) or []
    if not traj:
        return None
    ff = getattr(record, 'first_fail_ts', None)
    pre = [t for t in traj if ff is None or t.ts <= ff] or list(traj)
    ee = getattr(pre[-1], 'ee_pos', None)
    if ee is None or len(ee) < 2:
        return None
    try:
        return (float(ee[0]), float(ee[1]))
    except (TypeError, ValueError):
        return None

def _invocation(ts: float, failure_type: str, name: str, version: str, ctx: Optional[RecoveryContext], *, outcome: RecoveryOutcome, notes: str, handback_mode: HandbackMode='full_hold', request_id: Optional[str]=None) -> RecoveryInvocation:
    return RecoveryInvocation(triggered_ts=ts, failure_type=failure_type, policy_name=name, policy_version=version, request_id=request_id, handback_mode=handback_mode, rewind_target_ts=ctx.rewind_target_ts if ctx else None, rewind_target_q=ctx.rewind_target_q if ctx else None, perceived_grasp=ctx.grasp if ctx else None, grounding=ctx.grounding if ctx else None, outcome=outcome, notes=notes or None)

def _summarize(steps: list[StepResult]) -> str:
    parts = [f"{s.op}={('ok' if s.ok else 'FAIL')}" for s in steps]
    tail = next((s.reason for s in steps if not s.ok and s.reason), '')
    line = ' -> '.join(parts)
    return f'{line} | {tail}' if tail else line

def _rewind_succeeded(steps: list[StepResult]) -> bool:
    return any((s.op in ('rewind', 'fail_safe_rewind') and s.ok for s in steps))
