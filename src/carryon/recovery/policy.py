"""Declarative recovery policies and a versioned policy registry.

Following AgentChord-style composition, each RecoveryStep names an allowed
RecoveryBackend primitive. Following cap-x-style registration, policies are
keyed by (name, version), and invocations record that identity. The interpreter
executes declared steps through the backend interface."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .primitives import (
    DEFAULT_GRIPPER_CLOSED_WIDTH,
    DEFAULT_GRIPPER_EMPTY_WIDTH,
    DEFAULT_JOINT_TOL_RAD,
    PredicateContext,
    fingertip_to_hand_pose,
    gripper_holding,
    joints_close,
)
from .schema import GraspProposal, GroundingResult, HandbackMode

# Fixed op vocabulary the LLM may compose (one op == one primitive call), plus the
# one control-flow construct: ``retry`` (a bounded loop over a nested body).
OPS = (
    "open_gripper",
    "close_gripper",
    "ground_target",
    "perceive_grasp",
    "perceive_grasps",
    "verify_object_identity",
    "goto_grasp",
    "goto_grasp_goalset",
    "goto_grasp_topdown_family",
    "goto_pose",
    "ground_destination",
    "goto_place",
    "verify_place",
    "goto_init",
    "lift",
    "vertical_lift_to_clearance",
    "rewind",
    "assert_holding",
    "assert_holding_pose_retry",
    "assert_at_rewind",
    "retry",
)
# Declarative steps name individual primitives. Express regrasping by composing
# open_gripper, perceive_grasp, goto_grasp and close_gripper.
MAX_RETRY_ATTEMPTS = 5

# Franka "ready" joint config (rad) -- the default arm initialization vantage the
# regrasp policy retreats to before re-perceiving. Overridden per-rig via
# ``RecoveryConfig.init_q`` (wired from ``config.recovery.init_q``).
FRANKA_READY_Q = (0.0, 0.0, 0.0, -1.566, 0.0, 1.579, 0.0)


@dataclass
class RecoveryConfig:
    """Tolerances the orchestrator + policy interpreter share (override per-task)."""

    # Retained for deserializing policy records; failure handling always returns control.
    handback_on_failure: bool = True

    # Pre-grasp identity check: after perceive, crop the grounded mask and ask
    # the reflection VLM whether it is the expected object; a mismatch aborts
    # the grasp into the existing retry machinery. When no VLM is configured,
    # this optional identity check is skipped.
    verify_identity: bool = True

    # New-matrix semantics (see config.recovery): library one-shot compiles
    # seeds without retry bodies; reflect_turns bounds inline reflection.
    library_oneshot: bool = False
    reflect_turns: int = 1

    rewind_lead_s: float = 1.0
    pre_grasp_dis: float = 0.10
    # Default ADDITIONAL world-+Z nudge applied to the perceived grasp before the
    # descent (overridable per-step via goto_grasp's ``z_offset``). 0 = trust the
    # perceived fingertip as-is. Negative LOWERS the fingertip (descend deeper, e.g.
    # onto a tall cup whose graspable wall sits well below its rim); positive raises
    # it. This is the per-object descent-depth knob -- perception gives the visible
    # top surface, but how far to sink onto the graspable geometry is object-specific.
    grasp_z_offset: float = 0.0
    # Default ADDITIONAL world-frame (x, y) nudge applied to the perceived grasp
    # before the descent (overridable per-step via goto_grasp's ``x_offset`` /
    # ``y_offset``). 0 = trust the perceived xy. Corrects a systematic lateral bias
    # for an object (e.g. perception centroid sits off the graspable part); symmetric
    # to ``grasp_z_offset`` and, like it, applied on a COPY of the perceived position
    # so perception is never mutated. Does NOT change the approach axis (orientation
    # is left to perception), so it carries none of the table-strike risk of a tilt.
    grasp_x_offset: float = 0.0
    grasp_y_offset: float = 0.0
    # Optional fixed world-Z yaw nudge applied on top of the perceived grasp
    # orientation. 0 preserves the detector's PCA/topdown yaw; nonzero is a narrow
    # ablation/tuning knob for objects whose gripper opening axis is misaligned.
    grasp_yaw_offset_deg: float = 0.0
    # default world-+Z distance for the ``lift`` grasp-confirm step (overridable
    # per-step via the op's ``dz`` param).
    lift_confirm_dz: float = 0.05
    # arm initialization joint config the ``goto_init`` op retreats to before a
    # regrasp (clean perception/planning vantage). Defaults to the franka ready
    # pose; the host overrides it from ``config.recovery.init_q``.
    init_q: list[float] = field(default_factory=lambda: list(FRANKA_READY_Q))
    gripper_empty_width: float = DEFAULT_GRIPPER_EMPTY_WIDTH
    gripper_open_width: float = DEFAULT_GRIPPER_CLOSED_WIDTH
    joint_tol_rad: float = DEFAULT_JOINT_TOL_RAD
    # Tolerance for the post-rewind ``assert_at_rewind`` joint check. Must match the
    # backend's looser ``rewind_joint_tol_rad`` (~12 deg): rewind / fail_safe_rewind
    # return the arm to a stored joint config for handback, not a precision grasp, and
    # the sim JOINT_POSITION PD leaves a steady-state residual on the WRIST joints
    # under a grasped load (~0.17 rad on joint 4). The tight grasp tolerance wrongly
    # fails this assertion even when the rewind motion itself converged.
    rewind_assert_tol_rad: float = math.radians(12)  # ~0.209 rad
    # assert_holding robustness: the gripper close *command* target and a margin
    # above it. A real object stalls the jaws OPEN of the close target, so holding
    # requires width > close_cmd + margin -- this catches a miss when the close
    # target (e.g. 12 mm) sits above the fixed empty_width floor (which would then
    # false-positive). None disables the command-relative check (legacy floor only).
    gripper_close_width: Optional[float] = None
    gripper_hold_margin_m: float = 0.005
    # Which lower bound assert_holding uses (see primitives.gripper_holding):
    # - "command_relative" (default, real-rig semantics): a missed close STALLS at
    #   the commanded width, so holding requires width > close_cmd + margin.
    # - "empty_relative" (binarising rigs, e.g. the LIBERO sim shim): any close
    #   command binarises to a FULL close, so a miss lands at ~empty_width
    #   (1-6 mm observed) and a held object stalls at its own thickness. The
    #   command-relative bound then rejects genuinely-held THIN objects (a
    #   19 mm cream-cheese box vs a 25 mm cmd+margin threshold); use
    #   width > empty_width + margin instead.
    gripper_hold_check: str = "command_relative"
    # When True, do not run the orchestrator's safety ``fail_safe_rewind`` unwind
    # after a partial policy failure. Used when the takeover/freeze pose is already
    # flagged cuRobo-infeasible -- planning back to the rewind target cannot succeed
    # and only adds noise to the trace.
    skip_fail_safe_rewind: bool = False


@dataclass
class RecoveryStep:
    """One declarative step: an op name from :data:`OPS` plus params.

    For the control-flow op ``retry`` the params carry a *nested* sub-sequence::

        RecoveryStep("retry", {"max_attempts": 3, "body": [RecoveryStep(...), ...]})

    The body is itself a list of :class:`RecoveryStep`s; the interpreter re-runs it
    (bounded by ``max_attempts``) until one attempt completes with every step ok.
    """

    op: str
    params: dict = field(default_factory=dict)


@dataclass
class StepResult:
    op: str
    ok: bool
    reason: str = ""
    dt_s: float = 0.0  # wall-clock duration of the step (for latency triage)
    data: dict = field(default_factory=dict)
    # For ``retry``: the inner step results across all attempts (for the trace).
    children: list["StepResult"] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize a step and its retry children for the recovery trace and reflection."""
        d: dict = {"op": self.op, "ok": self.ok, "dt_s": round(self.dt_s, 4)}
        if self.reason:
            d["reason"] = self.reason
        if self.data:
            d["data"] = self.data
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


def coerce_steps(raw) -> list[RecoveryStep]:
    """Normalize a heterogeneous step list (str | dict | RecoveryStep) into
    :class:`RecoveryStep`s, recursing into ``retry`` bodies. Shared by the policy
    constructors and the reflection gate so both see the same nested shape."""
    out: list[RecoveryStep] = []
    for s in raw or []:
        if isinstance(s, RecoveryStep):
            step = s
        elif isinstance(s, str):
            step = RecoveryStep(s)
        elif isinstance(s, dict):
            step = RecoveryStep(s.get("op", ""), dict(s.get("params", {})))
        else:
            step = RecoveryStep(str(s))
        if step.op == "retry" and "body" in step.params:
            step.params = dict(step.params)
            step.params["body"] = coerce_steps(step.params.get("body"))
        out.append(step)
    return out


@dataclass
class RecoveryContext:
    """Mutable scratch the interpreter threads through a policy run."""

    object_hint: str
    config: RecoveryConfig
    rewind_target_q: Optional[list[float]] = None
    rewind_target_ts: Optional[float] = None
    # Progressively earlier pre-failure joint states, tried in order when the
    # primary rewind target is planner-infeasible (see select_rewind_targets).
    rewind_fallback_qs: list[list[float]] = field(default_factory=list)
    grasp: Optional[GraspProposal] = None
    grasp_candidates: list[GraspProposal] = field(default_factory=list)
    # One immutable-per-perception pose table plus its monotone
    # execution cursor. The pose-step retry advances this cursor; it never calls
    # perception or the semantic selector again.
    grasp_candidate_index: Optional[int] = None
    grasp_candidate_index_history: list[int] = field(default_factory=list)
    # Spatial-disambiguation prior: the VLA's pre-failure end-effector XY (world
    # frame). When set, perception enumerates every instance matching the prompt
    # and picks the one nearest this point, so a generic mask ("can") resolves to
    # the object the VLA was actually reaching for rather than SAM3's argmax.
    target_xy_world: Optional[tuple] = None
    # Judge-proposed coarse normalized (x, y) anchor for the recovery target, in
    # [0, 1] image fractions. When set, perception enumerates instance masks with
    # a generic geometric prompt and picks the one nearest this point (in pixel
    # space) -- decoupling "which object" (Judge semantics) from "what shape"
    # (SAM3 geometry), so a food-name SAM3 can't ground still resolves correctly.
    # Takes precedence over ``target_xy_world`` for disambiguation when present.
    target_px: Optional[tuple] = None
    # Explicit grounding evidence produced by ``ground_target`` and consumed by
    # ``perceive_grasp``. If absent, legacy target_px/target_xy behavior remains.
    grounding: Optional[GroundingResult] = None
    # The most recent ``close_gripper`` command width (per-step override, if any).
    # ``assert_holding`` checks holding *relative to the close command*, so a step
    # that closed to a non-default width must be verified against THAT width, not
    # the rig default -- otherwise a tight close on a thin object false-fails.
    last_close_width: Optional[float] = None
    # Reflector-proposed SAM3 text fallbacks compiled from StrategySpec.params.
    sam_text_prompts: Optional[list[str]] = None
    # Verified semantic mask for the most recent perceive call. Overwritten on
    # every re-perceive so a physical retry cannot consume a stale mask witness.
    semantic_selection_witness: Optional[dict] = None


def _grasp_tilt_deg(g: GraspProposal) -> Optional[float]:
    """Tilt of the grasp approach axis away from world-down, matching backend."""
    try:
        q = [float(v) for v in g.quaternion]
        if len(q) != 4:
            return None
        w, x, y, z = q
        n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        w, x, y, z = w / n, x / n, y / n, z / n
        az = 1.0 - 2.0 * (x * x + y * y)
        cos_t = max(-1.0, min(1.0, -az))
        return math.degrees(math.acos(cos_t))
    except Exception:  # noqa: BLE001 - diagnostics must not break recovery
        return None


def _grasp_candidate_data(grasps: list[GraspProposal]) -> list[dict]:
    out = []
    tilt_ranks: dict[int, int] = {}
    sortable: list[tuple[float, float, int]] = []
    for i, g in enumerate(grasps):
        tilt = _grasp_tilt_deg(g)
        if tilt is not None:
            sortable.append((float(tilt), -float(g.score or 0.0), i))
    for rank, (_, _, idx) in enumerate(sorted(sortable)):
        tilt_ranks[idx] = rank
    for i, g in enumerate(grasps):
        tilt = _grasp_tilt_deg(g)
        approach_z = None
        try:
            q = [float(v) for v in g.quaternion]
            w, x, y, z = q
            n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
            x, y = x / n, y / n
            approach_z = 1.0 - 2.0 * (x * x + y * y)
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        out.append({
            "index": i,
            "tilt_rank": tilt_ranks.get(i),
            "position": [round(float(v), 4) for v in g.position],
            "quaternion": [round(float(v), 6) for v in g.quaternion],
            "score": None if g.score is None else round(float(g.score), 6),
            "source": g.source,
            "predicted_width_m": (
                None if g.predicted_width_m is None
                else round(float(g.predicted_width_m), 6)
            ),
            "tilt_deg": None if tilt is None else round(float(tilt), 3),
            "approach_z": None if approach_z is None else round(float(approach_z), 6),
        })
    return out


class _HoldingPredicate:
    name = "gripper_holding"

    def __init__(self, empty: float, open_: float, *,
                 closed_cmd: Optional[float] = None, hold_margin: float = 0.0) -> None:
        self._empty, self._open = empty, open_
        self._closed_cmd, self._margin = closed_cmd, hold_margin
        self.last_gripper_width: Optional[float] = None

    def __call__(self, ctx: PredicateContext) -> bool:
        self.last_gripper_width = ctx.gripper_width
        if ctx.gripper_width is None:
            return False
        return gripper_holding(
            ctx.gripper_width,
            empty_width=self._empty,
            open_width=self._open,
            closed_cmd_width=self._closed_cmd,
            hold_margin=self._margin,
        )


def _state_debug(backend) -> dict:
    """Best-effort bridge-state snapshot for contact/close/lift diagnostics."""
    read = getattr(backend, "_read_state", None)
    if not callable(read):
        return {}
    try:
        st = read()
    except Exception as exc:  # noqa: BLE001 - diagnostics must not alter recovery
        return {"state_error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(st, dict):
        return {}
    out = {}
    for key in ("w_meas", "w_cmd"):
        if key in st and st[key] is not None:
            try:
                out[key] = round(float(st[key]), 6)
            except (TypeError, ValueError):
                out[key] = st[key]
    if "ee_pos" in st and st["ee_pos"] is not None:
        try:
            out["ee_pos"] = [round(float(v), 4) for v in st["ee_pos"]]
        except (TypeError, ValueError):
            pass
    if "grasp_contact" in st:
        out["grasp_contact"] = st["grasp_contact"]
    return out


def _holding_contact_width_lift_stable_v2(
    *,
    measured_width_m: Optional[float],
    grasp_contact: object,
    max_open_width_m: float,
    max_open_margin_m: float,
    lift_width_epsilon_m: float,
    lift_width_trace: dict,
) -> tuple[bool, str, dict]:
    """Check contact, gripper width and holding stability through the lift."""
    upper_limit_m = float(max_open_width_m) - float(max_open_margin_m)
    trace = dict(lift_width_trace or {})
    try:
        sample_count = int(trace.get("sample_count", 0))
    except (TypeError, ValueError):
        sample_count = 0
    try:
        trace_delta_m = float(trace["delta_width_m"])
    except (KeyError, TypeError, ValueError):
        trace_delta_m = None
    contact_ok = grasp_contact is True
    width_ok = (
        measured_width_m is not None
        and float(measured_width_m) < upper_limit_m
    )
    trace_ok = sample_count >= 2 and trace_delta_m is not None
    stable_ok = trace_ok and trace_delta_m < float(lift_width_epsilon_m)
    ok = bool(contact_ok and width_ok and stable_ok)
    if not contact_ok:
        reason = "holding v2 failed: grasp_contact is not true"
    elif measured_width_m is None:
        reason = "holding v2 failed: gripper width unavailable"
    elif not width_ok:
        reason = "holding v2 failed: width is not below max_open-margin"
    elif not trace_ok:
        reason = "holding v2 failed: lift width trace has fewer than two samples"
    elif not stable_ok:
        reason = "holding v2 failed: lift width is unstable"
    else:
        reason = ""
    diagnostics = {
        "semantics": "contact_width_lift_stable_v2",
        "measured_width_m": measured_width_m,
        "grasp_contact": grasp_contact,
        "max_open_width_m": float(max_open_width_m),
        "max_open_margin_m": float(max_open_margin_m),
        "upper_limit_m": upper_limit_m,
        "lift_width_epsilon_m": float(lift_width_epsilon_m),
        "lift_width_trace": trace,
        "contact_ok": contact_ok,
        "width_ok": width_ok,
        "trace_ok": trace_ok,
        "stable_ok": stable_ok,
    }
    return ok, reason, diagnostics


class RecoveryPolicy(Protocol):
    name: str
    version: str
    failure_type: str
    handback_mode: HandbackMode
    # Free-text object/task tag this policy is specialized for (e.g. "carrot" or
    # "pick carrot from box"). Empty = generic (applies to any object). The live
    # coordinator matches it against the runtime object_hint to pick a per-object
    # skill; the offline reflector reads it to know which policy to build on.
    applies_to: str

    def run(self, backend, ctx: RecoveryContext) -> list[StepResult]: ...


@dataclass
class SequentialPolicy:
    """Run a fixed list of steps; short-circuit on the first failure.

    Each op maps to exactly one backend primitive (motion / gripper / perception)
    or a pure post-condition assertion. ``planned_traj`` is collected from motion
    results for the trace.
    """

    name: str
    version: str
    steps: list[RecoveryStep]
    failure_type: str = "hold_lost"
    # Object/task this policy is tuned for; "" = generic. See RecoveryPolicy.
    applies_to: str = ""
    handback_mode: HandbackMode = "full_hold"

    def run(self, backend, ctx: RecoveryContext, *, on_step=None) -> list[StepResult]:
        """Run the steps in order, short-circuiting on the first failure.

        ``on_step`` (optional) is called with each :class:`StepResult` right after the
        step completes, so a host can stream live progress to the operator (the hot
        path is otherwise silent while the arm moves)."""
        results: list[StepResult] = []
        for step in self.steps:
            t0 = time.monotonic()
            res = self._run_step(backend, ctx, step)
            res.dt_s = time.monotonic() - t0
            results.append(res)
            if on_step is not None:
                try:
                    on_step(res)
                except Exception:  # noqa: BLE001 -- logging must never break the loop
                    pass
            if not res.ok:
                break
        return results

    def _run_step(self, backend, ctx: RecoveryContext, step: RecoveryStep) -> StepResult:
        op, p = step.op, step.params
        if op == "retry":
            return self._run_retry(backend, ctx, p)
        if op == "open_gripper":
            return _from_motion(op, backend.open_gripper())
        if op == "close_gripper":
            w = p.get("width")
            ctx.last_close_width = float(w) if w is not None else None
            res = backend.close_gripper(ctx.last_close_width)
            data = {
                "gripper": {
                    "command_width": ctx.last_close_width,
                    "state_after": _state_debug(backend),
                }
            }
            return _from_motion(op, res, data=data)
        if op == "ground_target":
            plan = list(p.get("plan", ["task_target"]))
            # R2: an explicit reflector-supplied point (normalized image fractions)
            # overrides the judge point as the segmenter anchor, so a failure-
            # informed re-grounding steers to the correct object. Same [0,1] target_px
            # semantics; falls back to the judge point when the reflector gave none.
            tp = p.get("target_point")
            target_px = tuple(tp) if isinstance(tp, (list, tuple)) and len(tp) == 2 else ctx.target_px
            try:
                ground = backend.ground_target(
                    ctx.object_hint,
                    plan=plan,
                    target_xy=ctx.target_xy_world,
                    target_px=target_px,
                )
            except AttributeError:
                ground = _default_grounding(ctx, plan)
            except TypeError:
                ground = backend.ground_target(ctx.object_hint, plan=plan)
            ctx.grounding = ground
            if ground is None:
                return StepResult(op, False, "ground_target produced no grounding prior")
            bits = [f"mode={ground.mode!r}"]
            if ground.provenance:
                bits.append(f"provenance={ground.provenance!r}")
            if ground.normalized_point is not None:
                bits.append(f"point={list(map(float, ground.normalized_point))}")
            if ground.world_point is not None:
                bits.append(f"world={list(map(float, ground.world_point))}")
            return StepResult(op, True, " ".join(bits))
        if op == "perceive_grasp":
            if p.get("sam_text_prompts"):
                ctx.sam_text_prompts = [str(x).strip() for x in p["sam_text_prompts"] if str(x).strip()]
            # Once a grounding plan exists, only evidence selected by that plan
            # may reach perception. Do not leak Judge/contact priors around it.
            target_xy = None if ctx.grounding is not None else ctx.target_xy_world
            target_px = None if ctx.grounding is not None else ctx.target_px
            if ctx.grounding is not None:
                if ctx.grounding.normalized_point is not None:
                    target_px = tuple(ctx.grounding.normalized_point)
                if ctx.grounding.world_point is not None and len(ctx.grounding.world_point) >= 2:
                    target_xy = tuple(ctx.grounding.world_point[:2])
            sam_kw = (
                {"sam_text_prompts": ctx.sam_text_prompts}
                if ctx.sam_text_prompts
                else {}
            )
            try:
                g = backend.perceive_grasp(
                    ctx.object_hint,
                    source=str(p.get("source", "topdown")),
                    target_xy=target_xy,
                    target_px=target_px,
                    grounding=ctx.grounding,
                    **sam_kw,
                )
            except TypeError:
                print(
                    "[policy] perceive_grasp: backend rejected new kwargs; "
                    "retrying reduced (capability silently reduced)", flush=True,
                )
                try:
                    g = backend.perceive_grasp(
                        ctx.object_hint,
                        source=str(p.get("source", "topdown")),
                        target_xy=target_xy,
                        target_px=target_px,
                        **sam_kw,
                    )
                except TypeError:
                    try:
                        g = backend.perceive_grasp(
                            ctx.object_hint,
                            source=str(p.get("source", "topdown")),
                            target_xy=target_xy,
                            target_px=target_px,
                        )
                    except TypeError:
                        # backend predates the spatial-disambiguation seam.
                        g = backend.perceive_grasp(ctx.object_hint, source=str(p.get("source", "topdown")))
            ctx.grasp = g
            # Single-grasp perception yields no multi-candidate center; clear any
            # stale anchor so a later center_correct/side_grasp fails safe.
            setattr(backend, "last_object_center_world", None)
            if g:
                src = f" via {g.source}" if g.source else ""
                score = "" if g.score is None else f" score={float(g.score):.3f}"
                return StepResult(
                    op,
                    True,
                    (
                        f"{src}{score} pos={list(map(float, g.position))} "
                        f"quat={list(map(float, g.quaternion))}"
                    ).strip(),
                )
            # Surface the precise cause (camera fetch vs SAM3 grounding vs no
            # candidate) the backend recorded, not a single ambiguous message.
            reason = getattr(backend, "last_perceive_reason", "") or "perception could not ground object"
            return StepResult(op, False, reason)
        if op == "perceive_grasps":
            if p.get("sam_text_prompts"):
                ctx.sam_text_prompts = [str(x).strip() for x in p["sam_text_prompts"] if str(x).strip()]
            top_k = int(p.get("top_k", 15))
            source = str(p.get("source", "auto"))
            # Once a grounding plan exists, only evidence selected by that plan
            # may reach perception. Do not leak Judge/contact priors around it.
            target_xy = None if ctx.grounding is not None else ctx.target_xy_world
            target_px = None if ctx.grounding is not None else ctx.target_px
            if ctx.grounding is not None:
                if ctx.grounding.normalized_point is not None:
                    target_px = tuple(ctx.grounding.normalized_point)
                if ctx.grounding.world_point is not None and len(ctx.grounding.world_point) >= 2:
                    target_xy = tuple(ctx.grounding.world_point[:2])
            sam_kw = (
                {"sam_text_prompts": ctx.sam_text_prompts}
                if ctx.sam_text_prompts
                else {}
            )
            try:
                gs = backend.perceive_grasps(
                    ctx.object_hint,
                    top_k=top_k,
                    source=source,
                    target_xy=target_xy,
                    target_px=target_px,
                    grounding=ctx.grounding,
                    **sam_kw,
                )
            except TypeError:
                print(
                    "[policy] perceive_grasps: backend rejected new kwargs; "
                    "retrying reduced (capability silently reduced)", flush=True,
                )
                try:
                    gs = backend.perceive_grasps(
                        ctx.object_hint,
                        top_k=top_k,
                        source=source,
                        target_xy=target_xy,
                        target_px=target_px,
                        **sam_kw,
                    )
                except TypeError:
                    try:
                        gs = backend.perceive_grasps(
                            ctx.object_hint,
                            top_k=top_k,
                            source=source,
                            target_xy=target_xy,
                            target_px=target_px,
                        )
                    except TypeError:
                        gs = backend.perceive_grasps(ctx.object_hint, top_k=top_k)
            ctx.grasp_candidates = list(gs or [])
            perceive_debug = dict(
                getattr(backend, "last_perceive_debug", {}) or {}
            )
            semantic = perceive_debug.get("semantic_memory")
            if isinstance(semantic, dict) and semantic.get("semantic_memory"):
                semantic = semantic.get("semantic_memory")
            ctx.semantic_selection_witness = (
                dict(semantic)
                if isinstance(semantic, dict)
                and semantic.get("verified") is True
                and semantic.get("verified_candidate_mask_sha256")
                else None
            )
            # Workspace sanity gate at the PERCEIVE boundary: candidates beyond
            # straight-line reach of the robot base are grounding garbage (a
            # background mask puts them metres away) and would poison EVERY
            # downstream consumer -- the goalset planner, the single-grasp path,
            # and the center_correct object-center mean alike. Fail the perceive
            # step with a grounding diagnosis when nothing reachable remains.
            ws_filter = getattr(backend, "_workspace_filter_candidates", None)
            if callable(ws_filter) and ctx.grasp_candidates:
                kept, ws_note = ws_filter(ctx.grasp_candidates)
                if not kept:
                    ctx.grasp_candidates = []
                    ctx.grasp = None
                    setattr(backend, "last_object_center_world", None)
                    return StepResult(
                        op, False,
                        "perception rejected: every grasp candidate outside the "
                        f"reachable workspace ({ws_note}) -- grounding masked a "
                        "background region, not a perception/motion failure",
                    )
                if ws_note:
                    ctx.grasp_candidates = kept
            ctx.grasp_candidate_index = 0 if ctx.grasp_candidates else None
            ctx.grasp_candidate_index_history = (
                [0] if ctx.grasp_candidates else []
            )
            ctx.grasp = ctx.grasp_candidates[0] if ctx.grasp_candidates else None
            # Object-center anchor = mean of candidate contact points. Threaded onto
            # the backend so the ``center_correct`` / ``side_grasp`` goto primitives
            # can retarget away from a single off-center argmax grasp. Cleared when
            # perception found nothing so those primitives fail safe.
            if ctx.grasp_candidates:
                n = len(ctx.grasp_candidates)
                center = [
                    sum(float(g.position[i]) for g in ctx.grasp_candidates) / n
                    for i in range(3)
                ]
                setattr(backend, "last_object_center_world", center)
            else:
                setattr(backend, "last_object_center_world", None)
            # Object z-span (calibrated height signal) recorded by the detector
            # while building the masked cloud; threaded onto the backend for the
            # depth_frac grasp-depth primitive. Missing span -> None (fail safe).
            det = getattr(backend, "detector", None)
            span = None
            for cand in (det, getattr(det, "primary", None), getattr(det, "fallback", None)):
                got = getattr(cand, "last_object_zspan_world", None)
                if got:
                    span = list(got)
                    break
            setattr(backend, "last_object_zspan_world", span)
            if not ctx.grasp_candidates:
                reason = getattr(backend, "last_perceive_reason", "") or "perception found no grasp candidates"
                return StepResult(op, False, reason)
            sources = sorted({g.source for g in ctx.grasp_candidates if g.source})
            return StepResult(
                op,
                True,
                f"n={len(ctx.grasp_candidates)} top_k={top_k} source={source!r} sources={sources}",
                data={
                    "grasp_candidates": _grasp_candidate_data(ctx.grasp_candidates),
                    **({
                        "perceive_debug": perceive_debug,
                    } if getattr(backend, "last_perceive_debug", None) else {}),
                },
            )
        if op == "verify_object_identity":
            fn = getattr(backend, "verify_object_identity", None)
            if fn is None:
                return StepResult(op, True, "identity check skipped (backend lacks primitive)")
            ok, note = fn(ctx.object_hint)
            return StepResult(op, bool(ok), str(note))
        if op == "goto_grasp":
            sources = [getattr(ctx.grasp, "source", "")]
            sources.extend(getattr(g, "source", "") for g in ctx.grasp_candidates)
            if any(str(source or "").lower().startswith("graspgen") for source in sources):
                return StepResult(
                    op, False,
                    "GraspGen candidates require perceive_grasps + goto_grasp_goalset; "
                    "goto_grasp would discard IK alternatives",
                )
            if ctx.grasp is None:
                return StepResult(op, False, "no grasp in context (perceive_grasp first)")
            z = float(p.get("z_approach", ctx.config.pre_grasp_dis))
            # Optional per-object world-Z nudge on the perceived fingertip: negative
            # lowers (descend deeper, e.g. onto a tall cup), positive raises. Defaults
            # to the rig's grasp_z_offset, so a policy that omits it is unchanged.
            z_off = float(p.get("z_offset", ctx.config.grasp_z_offset))
            # Optional per-object world-frame (x, y) nudge on the perceived fingertip
            # (lateral bias correction). Defaults to the rig's grasp_x/y_offset, so a
            # policy that omits them is unchanged. Orientation is untouched.
            x_off = float(p.get("x_offset", ctx.config.grasp_x_offset))
            y_off = float(p.get("y_offset", ctx.config.grasp_y_offset))
            pos = list(ctx.grasp.position)
            # center_correct: snap the fingertip XY to the perceived object center
            # (mean of grasp candidates) to cancel single-view near-face bias. Keeps
            # the perceived descent Z. No-op (fails safe) when no center was recovered.
            center = getattr(backend, "last_object_center_world", None)
            if bool(p.get("center_correct")) and center and len(center) >= 2:
                pos = [float(center[0]), float(center[1]), pos[2]]
            # depth_frac: grasp at a fraction of the object's OWN height below its
            # top (calibrated z-span from perception). Object-agnostic: a physical
            # fraction, not a per-object offset. Ensures the fingers wrap the body
            # instead of pinching the top edge (the dominant lift-slip mode). Only
            # ever DEEPENS the grasp (min), clamped above the object bottom/table.
            # No-op (fails safe) when the span is missing or degenerate.
            depth_frac = p.get("depth_frac")
            zspan = getattr(backend, "last_object_zspan_world", None)
            if depth_frac is not None and zspan and len(zspan) >= 2:
                z_bot, z_top = float(zspan[0]), float(zspan[1])
                h = z_top - z_bot
                if h > 0.015:
                    target_z = z_top - float(depth_frac) * h
                    pos[2] = max(z_bot + 0.008, min(pos[2], target_z))
            if x_off or y_off or z_off:
                pos = [pos[0] + x_off, pos[1] + y_off, pos[2] + z_off]
            verify = bool(p.get("verify", True))
            quat = list(ctx.grasp.quaternion)
            approach = str(p.get("approach", "")).strip().lower()
            if "quaternion" in p:
                quat = [float(v) for v in p["quaternion"]]
            elif approach == "side":
                # side_grasp: horizontal approach toward the object center. Approach
                # from the pre-failure EE direction when known, else default -Y.
                if not (center and len(center) >= 2):
                    return StepResult(op, False, "side_grasp needs an object center (perceive_grasps first)")
                cx, cy = float(center[0]), float(center[1])
                if ctx.target_xy_world is not None and len(ctx.target_xy_world) >= 2:
                    fx, fy = float(ctx.target_xy_world[0]), float(ctx.target_xy_world[1])
                else:
                    fx, fy = cx, cy - 1.0
                q_side = _side_approach_wxyz(cx, cy, fx, fy)
                if q_side is None:
                    return StepResult(op, False, "side_grasp could not build a horizontal approach")
                quat = q_side
                pos = [cx, cy, pos[2]]
            else:
                yaw_mode = str(p.get("yaw_mode", "pca")).strip().lower()
                if yaw_mode == "topdown":
                    # Canonical top-down Panda wrist (symmetric pinch on cylinders).
                    quat = [0.0, 1.0, 0.0, 0.0]
                elif yaw_mode == "pca_perpendicular":
                    # Jaws straddle ACROSS the mask's principal axis: a side-lying
                    # bottle's PCA major axis IS the bottle axis, and closing along
                    # it slides off -- rotate the pinch 90 deg so the fingers wrap
                    # the cylinder body.
                    quat = _yaw_offset_wxyz(quat, 90.0)
                elif yaw_mode not in ("pca", ""):
                    return StepResult(op, False, f"unknown goto_grasp.yaw_mode {yaw_mode!r}")
            yaw_off = float(p.get("yaw_offset_deg", ctx.config.grasp_yaw_offset_deg))
            if yaw_off:
                quat = _yaw_offset_wxyz(quat, yaw_off)
            standoff_only = bool(p.get("standoff_only"))
            if standoff_only:
                # pre_grasp handback: park at the pre-grasp standoff ABOVE the
                # object (do NOT descend onto it or close) so the VLA finishes the
                # fine pick after handback. Raise the target by the approach
                # distance and skip the descent + grasp verify.
                pos = [pos[0], pos[1], pos[2] + z]
                z = 0.0
                verify = False
            tcp = float(getattr(getattr(backend, "cfg", None), "tcp_offset_m", 0.0) or 0.0)
            hand_pos, norm_quat = fingertip_to_hand_pose(pos, quat, tcp)
            data = {
                "selected_grasp": {
                    "fingertip": [round(float(v), 4) for v in pos],
                    "hand": [round(float(v), 4) for v in hand_pos],
                    "quaternion": [round(float(v), 6) for v in norm_quat],
                    "tcp": round(tcp, 6),
                    "source": getattr(ctx.grasp, "source", None),
                    "score": None if getattr(ctx.grasp, "score", None) is None else round(float(ctx.grasp.score), 6),
                    "center_correct": bool(p.get("center_correct")),
                    "approach": approach or "pca",
                }
            }
            return _from_motion(
                op,
                backend.goto_pose(
                    pos,
                    quat,
                    z_approach=z,
                    verify=verify,
                ),
                data=data,
            )
        if op == "goto_grasp_goalset":
            if not ctx.grasp_candidates:
                return StepResult(op, False, "no grasp candidates in context (perceive_grasps first)")
            try:
                try:
                    res = backend.goto_grasp_goalset(
                        ctx.grasp_candidates,
                        z_approach=float(p.get("z_approach", ctx.config.pre_grasp_dis)),
                        verify=bool(p.get("verify", True)),
                        plan_grasp_to_lift=bool(p.get("plan_grasp_to_lift", False)),
                        prefer_low_tilt=p.get("prefer_low_tilt"),
                        max_tilt_deg=p.get("max_tilt_deg"),
                        failopen_tilt_band_deg=p.get("failopen_tilt_band_deg"),
                        allow_tilt_failopen=p.get("allow_tilt_failopen"),
                    )
                except TypeError:
                    print(
                        "[policy] goto_grasp_goalset: backend rejected tilt "
                        "kwargs; retrying without bounds (capability silently "
                        "reduced)", flush=True,
                    )
                    # Use the goalset interface when no separate candidate-advance operation is available.
                    res = backend.goto_grasp_goalset(
                        ctx.grasp_candidates,
                        z_approach=float(p.get("z_approach", ctx.config.pre_grasp_dis)),
                        verify=bool(p.get("verify", True)),
                        plan_grasp_to_lift=bool(p.get("plan_grasp_to_lift", False)),
                    )
            except AttributeError:
                return StepResult(op, False, "backend has no goto_grasp_goalset primitive")
            data = {}
            if ctx.semantic_selection_witness is not None:
                marker = getattr(backend, "record_semantic_execution", None)
                if callable(marker):
                    data["semantic_execution_witness"] = marker(
                        physical_descent_started=bool(
                            getattr(backend, "last_goalset_execution_debug", {})
                            .get("physical_descent_started", False)
                        ),
                        motion_evidence=dict(
                            getattr(backend, "last_goalset_execution_debug", {}) or {}
                        ),
                    )
            return _from_motion(op, res, data=data or None)
        if op == "goto_grasp_topdown_family":
            if ctx.grasp is None or not ctx.grasp_candidates:
                return StepResult(op, False, "no frozen grasp target/candidates")
            family = getattr(backend, "goto_grasp_topdown_family", None)
            if not callable(family):
                return StepResult(op, False, "backend has no top-down yaw-family primitive")
            max_attempts = min(2, max(1, int(p.get("fresh_retry_max_attempts", 1))))
            attempt_data = []
            res = None
            for attempt in range(1, max_attempts + 1):
                if ctx.grasp is None or not ctx.grasp_candidates:
                    return StepResult(op, False, "fresh grounding produced no grasp target/candidates",
                                      data={"fresh_attempts": attempt_data})
                frozen = GraspProposal(
                    position=[float(v) for v in ctx.grasp.position],
                    quaternion=[float(v) for v in ctx.grasp.quaternion],
                    score=ctx.grasp.score, source=ctx.grasp.source,
                    predicted_width_m=ctx.grasp.predicted_width_m,
                )
                frozen_xyz = list(frozen.position)
                fallback_candidates = (
                    [frozen]
                    if bool(p.get("frozen_candidate_only", False))
                    else list(ctx.grasp_candidates)
                )
                res = family(
                    frozen, fallback_candidates,
                    yaw_degs=tuple(float(v) for v in p.get("yaw_degs", [0, 45, 90, 135])),
                    z_approach=float(p.get("z_approach", ctx.config.pre_grasp_dis)),
                    verify=bool(p.get("verify", True)),
                    plan_grasp_to_lift=bool(p.get("plan_grasp_to_lift", False)),
                    fallback_max_tilt_deg=float(p.get("fallback_max_tilt_deg", 40.0)),
                )
                family_debug = dict(getattr(backend, "last_topdown_family_debug", {}) or {})
                semantic_execution = None
                if ctx.semantic_selection_witness is not None:
                    marker = getattr(backend, "record_semantic_execution", None)
                    if callable(marker):
                        goalset_execution = dict(
                            getattr(backend, "last_goalset_execution_debug", {}) or {}
                        )
                        semantic_execution = marker(
                            physical_descent_started=bool(
                                goalset_execution.get("physical_descent_started", False)
                            ),
                            motion_evidence=goalset_execution,
                        )
                attempt_entry = {
                    "attempt": attempt,
                    "pose_index": ctx.grasp_candidate_index,
                    "frozen_target_xyz": frozen_xyz,
                    "ok": bool(res.ok),
                    "reason": str(res.reason),
                    "topdown_family": family_debug,
                    "semantic_execution_witness": semantic_execution,
                }
                attempt_data.append(attempt_entry)
                if list(ctx.grasp.position) != frozen_xyz:
                    return StepResult(op, False, "frozen target mutated during physical attempt",
                                      data={"fresh_attempts": attempt_data})
                if res.ok:
                    return _from_motion(op, res, data={
                        "frozen_target_xyz": frozen_xyz,
                        "topdown_family": family_debug,
                        "fresh_attempts": attempt_data,
                    })
                physical = "requires_fresh_grounding" in str(res.reason)
                if not physical or attempt >= max_attempts:
                    break

                retreat = getattr(backend, "retreat_after_topdown_physical_failure", None)
                if not callable(retreat):
                    return StepResult(op, False, "backend has no topdown jam-retreat primitive",
                                      data={"fresh_attempts": attempt_data})
                retreat_res = retreat()
                attempt_entry["retreat"] = {
                    "ok": bool(retreat_res.ok), "reason": str(retreat_res.reason)}
                if not retreat_res.ok:
                    return StepResult(op, False, "topdown jam retreat failed: " + str(retreat_res.reason),
                                      data={"fresh_attempts": attempt_data})

                # The physical contact invalidated the old target. Acquire a fresh
                # frame and rebuild both grounding and grasps before the sole retry.
                ctx.grounding = None
                ctx.grasp = None
                ctx.grasp_candidates = []
                ground_res = self._run_step(
                    backend, ctx,
                    RecoveryStep("ground_target", dict(p.get("fresh_ground_params", {}))),
                )
                perceive_res = None
                if ground_res.ok:
                    perceive_res = self._run_step(
                        backend, ctx,
                        RecoveryStep("perceive_grasps", dict(p.get("fresh_perceive_params", {}))),
                    )
                attempt_entry["fresh_ground"] = {
                    "ok": bool(ground_res.ok), "reason": str(ground_res.reason)}
                attempt_entry["fresh_perceive"] = None if perceive_res is None else {
                    "ok": bool(perceive_res.ok), "reason": str(perceive_res.reason)}
                if not ground_res.ok or perceive_res is None or not perceive_res.ok:
                    why = ground_res.reason if not ground_res.ok else perceive_res.reason
                    return StepResult(op, False, "fresh re-ground/perceive failed: " + str(why),
                                      data={"fresh_attempts": attempt_data})
            return _from_motion(op, res, data={"fresh_attempts": attempt_data})
        if op == "goto_pose":
            return _from_motion(
                op,
                backend.goto_pose(p["position"], p["quaternion"], z_approach=float(p.get("z_approach", 0.0))),
            )
        if op == "ground_destination":
            ground = getattr(backend, "ground_destination", None)
            if not callable(ground):
                return StepResult(op, False, "backend has no ground_destination primitive")
            return _from_motion(op, ground())
        if op == "goto_place":
            # complete handback: carry the HELD object over the place
            # destination (grounded from the instruction, not oracle) and descend
            # to a release standoff; the following open_gripper releases it. The
            # destination is the same place target previously excluded from grasping,
            # so it is already grounded this episode. Delegates to the backend place
            # primitive (curobo plan to the deprojected destination centroid);
            # fails closed (no arbitrary motion) if the backend lacks it or the
            # destination cannot be grounded.
            place = getattr(backend, "goto_place", None)
            if not callable(place):
                return StepResult(op, False, "backend has no goto_place primitive")
            try:
                res = place(
                    clearance_m=float(p.get("clearance_m", 0.03)),
                    release_bias_m=float(p.get("release_bias_m", 0.02)),
                    standoff_only=bool(p.get("standoff_only", False)),
                )
            except TypeError:
                if bool(p.get("standoff_only", False)):
                    return StepResult(op, False, "backend goto_place lacks required standoff_only contract")
                res = place()
            data = {"place": {"clearance_m": float(p.get("clearance_m", 0.03)),
                              "release_bias_m": float(p.get("release_bias_m", 0.02)),
                              "standoff_only": bool(p.get("standoff_only", False)),
                              "state_after": _state_debug(backend)}}
            return _from_motion(op, res, data=data)
        if op == "verify_place":
            verify = getattr(backend, "verify_place", None)
            if not callable(verify):
                return StepResult(op, False, "backend has no deployable place verifier")
            try:
                verdict = verify(
                    ctx.object_hint,
                    settle_s=float(p.get("settle_s", 1.0)),
                    min_overlap=float(p.get("min_overlap", 0.15)),
                )
            except TypeError:
                verdict = verify(ctx.object_hint)
            if isinstance(verdict, tuple):
                ok, reason = bool(verdict[0]), str(verdict[1])
            else:
                ok, reason = bool(verdict), "visual place verification"
            return StepResult(op, ok, reason)
        if op == "goto_init":
            init_q = p.get("q") or ctx.config.init_q
            if not init_q:
                return StepResult(op, False, "no init_q configured")
            try:
                res = backend.rewind_to_pose(
                    list(init_q), motion_segment="init")
            except TypeError:
                # Source-compatible fake/legacy backends lack observer metadata.
                res = backend.rewind_to_pose(list(init_q))
            return _from_motion(op, res)
        if op == "lift":
            dz = float(p.get("dz", ctx.config.lift_confirm_dz))
            res = backend.lift(dz)
            trace_reader = getattr(backend, "last_lift_width_trace", None)
            width_trace = trace_reader() if callable(trace_reader) else {}
            data = {
                "lift": {
                    "dz": dz,
                    "state_after": _state_debug(backend),
                    "width_trace": width_trace,
                }
            }
            return _from_motion(op, res, data=data)
        if op == "vertical_lift_to_clearance":
            dz = float(p["dz"])
            state_before = _state_debug(backend)
            res = backend.vertical_lift_to_clearance(dz)
            state_after = _state_debug(backend)
            trace_reader = getattr(backend, "last_lift_width_trace", None)
            width_trace = trace_reader() if callable(trace_reader) else {}
            z_start = None
            z_end = None
            try:
                z_start = float(state_before["ee_pos"][2])
                z_end = float(state_after["ee_pos"][2])
            except (KeyError, IndexError, TypeError, ValueError):
                pass
            data = {
                "vertical_lift_to_clearance": {
                    "target_dz_m": dz,
                    "state_before": state_before,
                    "state_after": state_after,
                    "width_trace": width_trace,
                    "ee_z_start_m": z_start,
                    "ee_z_end_m": z_end,
                    "actual_dz_m": (
                        z_end - z_start
                        if z_start is not None and z_end is not None
                        else None
                    ),
                }
            }
            return _from_motion(op, res, data=data)
        if op == "rewind":
            if ctx.rewind_target_q is None:
                return StepResult(op, False, "no rewind target selected")
            res = backend.rewind_to_pose(ctx.rewind_target_q)
            if not res.ok:
                # The primary target is a VLA-visited state a planner may still
                # refuse (padded joint limits / collision margins). Any earlier
                # pre-failure sample is an equally valid handback state.
                for q_fb in ctx.rewind_fallback_qs:
                    fb = backend.rewind_to_pose(list(q_fb))
                    if fb.ok:
                        ctx.rewind_target_q = list(q_fb)  # assert against what we reached
                        fb.reason = (
                            f"primary rewind target infeasible ({res.reason}); "
                            f"used earlier fallback target"
                        )
                        return _from_motion(op, fb)
            return _from_motion(op, res)
        if op == "assert_holding_pose_retry":
            return self._run_pose_step_retry(backend, ctx, p)
        if op == "assert_holding":
            semantics = str(p.get("semantics", "legacy")).strip()
            if semantics == "contact_width_lift_stable_v2":
                state_after = _state_debug(backend)
                measured = state_after.get("w_meas")
                if measured is None:
                    measured = state_after.get("w_cmd")
                backend_cfg = getattr(backend, "cfg", None)
                max_open = getattr(backend_cfg, "gripper_open_width", None)
                if max_open is None:
                    return StepResult(
                        op, False,
                        "holding v2 failed: backend.cfg.gripper_open_width unavailable",
                    )
                try:
                    margin = float(p["max_open_margin_m"])
                    epsilon = float(p["lift_width_epsilon_m"])
                except (KeyError, TypeError, ValueError):
                    return StepResult(op, False, "holding v2 thresholds are missing or invalid")
                if margin <= 0.0 or epsilon <= 0.0 or margin >= float(max_open):
                    return StepResult(op, False, "holding v2 thresholds are out of range")
                trace_reader = getattr(backend, "last_lift_width_trace", None)
                width_trace = trace_reader() if callable(trace_reader) else {}
                ok, reason, diagnostics = _holding_contact_width_lift_stable_v2(
                    measured_width_m=measured,
                    grasp_contact=state_after.get("grasp_contact"),
                    max_open_width_m=float(max_open),
                    max_open_margin_m=margin,
                    lift_width_epsilon_m=epsilon,
                    lift_width_trace=width_trace,
                )
                diagnostics["state_after"] = state_after
                print(
                    "[assert_holding:v2] "
                    f"contact_ok={diagnostics['contact_ok']} "
                    f"width_ok={diagnostics['width_ok']} "
                    f"stable_ok={diagnostics['stable_ok']} "
                    f"measured={measured} delta={diagnostics['lift_width_trace'].get('delta_width_m')}",
                    flush=True,
                )
                return StepResult(
                    op,
                    bool(ok),
                    reason,
                    data={"holding_check": diagnostics},
                )
            if semantics not in ("", "legacy"):
                return StepResult(op, False, f"unknown assert_holding semantics '{semantics}'")
            # Verify holding relative to the width the jaws were last COMMANDED to
            # (a per-step ``close_gripper`` override wins over the rig default), so
            # a tuned-tighter grasp is judged against its own close target.
            closed_cmd = (
                ctx.last_close_width
                if ctx.last_close_width is not None
                else ctx.config.gripper_close_width
            )
            hold_check = getattr(ctx.config, "gripper_hold_check", "command_relative")
            if hold_check == "empty_relative":
                # Binarising rig: a miss fully closes to ~empty_width, so the
                # command-relative bound is unnecessary AND rejects held thin
                # objects. Gate on empty_width + margin only.
                pred = _HoldingPredicate(
                    ctx.config.gripper_empty_width + ctx.config.gripper_hold_margin_m,
                    ctx.config.gripper_open_width,
                )
                threshold = float(ctx.config.gripper_empty_width) + float(
                    ctx.config.gripper_hold_margin_m
                )
            else:
                pred = _HoldingPredicate(
                    ctx.config.gripper_empty_width,
                    ctx.config.gripper_open_width,
                    closed_cmd=closed_cmd,
                    hold_margin=ctx.config.gripper_hold_margin_m,
                )
                threshold = (
                    None
                    if closed_cmd is None
                    else float(closed_cmd) + float(ctx.config.gripper_hold_margin_m)
                )
            ok = backend.geometric_check(pred)
            measured = pred.last_gripper_width
            state_after = _state_debug(backend)
            # Secondary FAIR tactile gate: jaw WIDTH alone cannot separate a thin
            # object HELD (width ~ empty_width) from an empty near-miss (width
            # slightly > empty) -- the thin-box false positive that corrupts
            # chain_ok. If the sim exposes a proprioceptive grasp-contact signal
            # (object-agnostic; NOT the target's pose/identity) and it reports
            # NOTHING gripped, override a width-only "holding" to a miss. None =>
            # signal unavailable => fall back to the width check alone.
            gc = state_after.get("grasp_contact") if isinstance(state_after, dict) else None
            print(f"[assert_holding] width_ok={ok} grasp_contact={gc!r} measured={measured}", flush=True)
            miss_reason = "gripper not holding object (jaws closed past the object -> likely a miss)"
            if ok and gc is False:
                ok = False
                miss_reason = ("gripper not holding object (no gripper-object contact: "
                               "empty near-miss -- width alone false-positives a thin box)")
            data = {
                "holding_check": {
                    "measured_width": None if measured is None else round(float(measured), 6),
                    "closed_cmd_width": None if closed_cmd is None else round(float(closed_cmd), 6),
                    "hold_margin": round(float(ctx.config.gripper_hold_margin_m), 6),
                    "hold_check": hold_check,
                    "grasp_contact": gc,
                    "command_relative_threshold": None if threshold is None else round(float(threshold), 6),
                    "empty_width": round(float(ctx.config.gripper_empty_width), 6),
                    "open_width": round(float(ctx.config.gripper_open_width), 6),
                    "state_after": state_after,
                }
            }
            return StepResult(op, bool(ok), "" if ok else miss_reason, data=data)
        if op == "assert_at_rewind":
            if ctx.rewind_target_q is None:
                return StepResult(op, False, "no rewind target to assert against")
            # Prefer the backend's task-space (EE-position) assertion so this gate
            # agrees with the rewind motion gate: a task-irrelevant wrist-roll residual
            # (unbounded for a symmetric gripper) passes, a real positional miss fails.
            at = getattr(backend, "at_rewind_target", None)
            if callable(at):
                ok, why = at(ctx.rewind_target_q)
                return StepResult(op, bool(ok), "" if ok else (why or "arm not at rewind target"))
            try:
                q_now = backend.current_q()
            except Exception as e:  # noqa: BLE001
                return StepResult(op, False, f"could not read current_q: {e}")
            ok = joints_close(q_now, ctx.rewind_target_q, tol=ctx.config.rewind_assert_tol_rad)
            return StepResult(op, ok, "" if ok else "arm not at rewind target")
        return StepResult(op, False, f"unknown op '{op}'")

    def _run_pose_step_retry(
        self, backend, ctx: RecoveryContext, p: dict
    ) -> StepResult:
        """Advance once to the next GraspGen candidate after a holding failure.

        This sequence reuses the existing grounding, perception and semantic selection."""
        children: list[StepResult] = []
        candidates = ctx.grasp_candidates
        current = ctx.grasp_candidate_index
        ledger = {
            "schema": "carryon-pose-step-retry/v1",
            "candidate_count": len(candidates),
            "initial_pose_index": current,
            "pose_index_history_before": list(ctx.grasp_candidate_index_history),
            "frozen_graspgen_score_order_indices": list(range(len(candidates))),
            "no_next_frozen_pose": False,
            "retry_budget": 1,
            "reperception_forbidden": True,
            "reselection_forbidden": True,
        }

        def finish(ok: bool, reason: str) -> StepResult:
            history = list(ctx.grasp_candidate_index_history)
            monotone = all(b > a for a, b in zip(history, history[1:]))
            ledger.update(
                final_pose_index=ctx.grasp_candidate_index,
                pose_index_history_after=history,
                pose_index_monotone=monotone,
            )
            if not monotone:
                ok = False
                reason = "pose retry index history is not strictly monotone"
            result = StepResult(
                "assert_holding_pose_retry",
                bool(ok),
                reason,
                data={"pose_retry": ledger},
            )
            result.children = children
            return result

        try:
            retry_budget = int(p.get("retry_budget", 1))
        except (TypeError, ValueError):
            return finish(False, "pose retry budget must be an integer")
        if retry_budget != 1:
            return finish(False, "pose retry budget must equal one")
        assert_params = dict(p.get("assert_params", {}))
        if assert_params.get("semantics") != "contact_width_lift_stable_v2":
            return finish(False, "pose retry requires frozen F2 holding semantics")

        first_assert = self._run_step(
            backend, ctx, RecoveryStep("assert_holding", assert_params)
        )
        children.append(first_assert)
        ledger["first_assert_ok"] = bool(first_assert.ok)
        if first_assert.ok:
            return finish(True, "holding passed on initial frozen pose")

        if current is None or not 0 <= int(current) < len(candidates):
            return finish(False, "pose retry has no valid current pose index")
        next_index = int(current) + 1
        if next_index >= len(candidates):
            next_index = None
        ledger["next_pose_index"] = next_index
        if next_index is None:
            ledger["no_next_frozen_pose"] = True
            return finish(
                False,
                "no_retry_pose_available: frozen GraspGen score list is exhausted",
            )

        open_result = self._run_step(
            backend, ctx, RecoveryStep("open_gripper")
        )
        children.append(open_result)
        if not open_result.ok:
            return finish(False, "pose retry open_gripper failed")

        retreat = getattr(backend, "retreat_to_last_goalset_standoff", None)
        if not callable(retreat):
            return finish(
                False,
                "backend lacks frozen goalset standoff retreat primitive",
            )
        retreat_motion = retreat()
        retreat_result = _from_motion(
            "retreat_to_grasp_standoff",
            retreat_motion,
            data={
                "retreat": dict(
                    getattr(backend, "last_pose_retry_retreat_debug", {}) or {}
                )
            },
        )
        children.append(retreat_result)
        if not retreat_result.ok:
            return finish(False, "frozen goalset standoff retreat failed")

        previous = int(current)
        if int(next_index) <= previous:
            return finish(False, "pose retry index did not advance")
        ctx.grasp_candidate_index = int(next_index)
        ctx.grasp_candidate_index_history.append(int(next_index))
        ctx.grasp = candidates[int(next_index)]
        advance = StepResult(
            "advance_grasp_pose",
            True,
            f"advanced frozen pose index {previous}->{int(next_index)}",
            data={
                "previous_pose_index": previous,
                "pose_index": int(next_index),
                "predicted_width_m": getattr(
                    ctx.grasp, "predicted_width_m", None
                ),
            },
        )
        children.append(advance)

        goto_params = dict(p.get("goto_params", {}))
        goto_params["fresh_retry_max_attempts"] = 1
        goto_params["frozen_candidate_only"] = True
        close_params = dict(p.get("close_params", {}))
        lift_params = dict(p.get("lift_params", {}))
        for step in (
            RecoveryStep("goto_grasp_topdown_family", goto_params),
            RecoveryStep("close_gripper", close_params),
            RecoveryStep("lift", lift_params),
            RecoveryStep("assert_holding", assert_params),
        ):
            result = self._run_step(backend, ctx, step)
            children.append(result)
            if not result.ok:
                if step.op == "assert_holding":
                    ledger["second_assert_ok"] = False
                    return finish(
                        False,
                        "pose retry exhausted after second holding failure",
                    )
                return finish(False, f"pose retry {step.op} failed")
        ledger["second_assert_ok"] = True
        return finish(True, "holding passed on next frozen GraspGen-score pose")

    def _run_retry(self, backend, ctx: RecoveryContext, p: dict) -> StepResult:
        """Bounded loop: re-run ``body`` until one attempt is fully ok, or attempts
        run out. This is how a policy expresses "regrasp again on a miss" without any
        hot-path code -- success is decided by the body's own assert_* step(s)."""
        body = coerce_steps(p.get("body"))
        if not body:
            return StepResult("retry", False, "retry has an empty body")
        try:
            max_attempts = int(p.get("max_attempts", 1))
        except (TypeError, ValueError):
            return StepResult("retry", False, "retry max_attempts must be an int")
        if max_attempts < 1:
            return StepResult("retry", False, "retry max_attempts must be >= 1")
        max_attempts = min(max_attempts, MAX_RETRY_ATTEMPTS)  # clamp: stay bounded

        children: list[StepResult] = []
        last_fail = ""
        for attempt in range(1, max_attempts + 1):
            attempt_ok = True
            for inner in body:
                t0 = time.monotonic()
                r = self._run_step(backend, ctx, inner)
                r.dt_s = time.monotonic() - t0
                r.reason = f"[try {attempt}/{max_attempts}] {r.reason}".rstrip()
                children.append(r)
                if not r.ok:
                    attempt_ok = False
                    last_fail = f"{r.op}: {r.reason}"
                    break
            if attempt_ok:
                res = StepResult("retry", True, f"ok on attempt {attempt}/{max_attempts}")
                res.children = children
                return res
        res = StepResult("retry", False, f"failed after {max_attempts} attempt(s); last: {last_fail}")
        res.children = children
        return res


def _from_motion(op: str, res, *, data: Optional[dict] = None) -> StepResult:
    return StepResult(op, bool(getattr(res, "ok", False)), getattr(res, "reason", ""), data=data or {})


def _default_grounding(ctx: RecoveryContext, plan: list[str]) -> GroundingResult:
    if "judge_point" in plan and ctx.target_px is not None:
        return GroundingResult(
            mode="judge_point",
            provenance="RecoveryContext.target_px",
            normalized_point=[float(ctx.target_px[0]), float(ctx.target_px[1])],
            object_hint=ctx.object_hint,
        )
    if "recent_contact" in plan and ctx.target_xy_world is not None:
        return GroundingResult(
            mode="recent_contact",
            provenance="RecoveryContext.target_xy_world",
            world_point=[float(ctx.target_xy_world[0]), float(ctx.target_xy_world[1])],
            object_hint=ctx.object_hint,
        )
    return GroundingResult(mode="task_target", provenance="RecoveryContext.object_hint", object_hint=ctx.object_hint)


def _yaw_offset_wxyz(quat: list[float], yaw_deg: float) -> list[float]:
    """Apply a world-Z yaw offset to a wxyz quaternion."""
    if len(quat) != 4:
        return quat
    half = math.radians(float(yaw_deg)) * 0.5
    rw, rz = math.cos(half), math.sin(half)
    w, x, y, z = [float(v) for v in quat]
    out = [
        rw * w - rz * z,
        rw * x - rz * y,
        rw * y + rz * x,
        rw * z + rz * w,
    ]
    norm = math.sqrt(sum(v * v for v in out))
    return [v / norm for v in out] if norm > 0 else quat


def _side_approach_wxyz(cx: float, cy: float, fx: float, fy: float) -> Optional[list[float]]:
    """Horizontal side-approach wxyz quaternion.

    Gripper local +z (the approach axis) points from ``(fx, fy)`` toward the object
    center ``(cx, cy)`` in the world XY plane; the fingers span the vertical. This
    is the genuine side_grasp orientation (vs. the canonical top-down wrist). Returns
    ``None`` if the approach direction is degenerate (approach origin == center)."""
    ax, ay = cx - fx, cy - fy
    n = math.sqrt(ax * ax + ay * ay)
    if n < 1e-6:
        return None
    az = [ax / n, ay / n, 0.0]                      # approach (gripper local +z)
    up = [0.0, 0.0, 1.0]
    ex = [                                           # local +x = up x approach
        up[1] * az[2] - up[2] * az[1],
        up[2] * az[0] - up[0] * az[2],
        up[0] * az[1] - up[1] * az[0],
    ]
    xn = math.sqrt(sum(v * v for v in ex))
    if xn < 1e-6:
        return None
    ex = [v / xn for v in ex]
    ey = [                                           # local +y = approach x local_x
        az[1] * ex[2] - az[2] * ex[1],
        az[2] * ex[0] - az[0] * ex[2],
        az[0] * ex[1] - az[1] * ex[0],
    ]
    m00, m01, m02 = ex[0], ey[0], az[0]
    m10, m11, m12 = ex[1], ey[1], az[1]
    m20, m21, m22 = ex[2], ey[2], az[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return [w / norm, x / norm, y / norm, z / norm]


# --------------------------------------------------------------------------- #
# registry (mirrors carryon.skills.SkillRegistry; keyed by name@version)      #
# --------------------------------------------------------------------------- #
def _policy_version_key(version: str) -> tuple:
    """Sort key for ``latest`` / ``select``: ``v9-foo`` beats ``v1``; non-``vN``
    strings sort below numeric ``vN`` prefixes (so debug tags don't beat v1)."""
    import re

    v = (version or "").strip()
    m = re.match(r"^v(\d+)", v, re.IGNORECASE)
    if m:
        return (2, int(m.group(1)), v.lower())
    return (1, 0, v.lower())


class RecoveryPolicyRegistry:
    def __init__(self) -> None:
        self._policies: dict[tuple[str, str], RecoveryPolicy] = {}

    def register(self, policy: RecoveryPolicy) -> None:
        key = (policy.name, policy.version)
        if key in self._policies:
            raise ValueError(f"recovery policy already registered: {policy.name}@{policy.version}")
        self._policies[key] = policy

    def get(self, name: str, version: Optional[str] = None) -> Optional[RecoveryPolicy]:
        if version is not None:
            return self._policies.get((name, version))
        return self.latest(name)

    def latest(self, name: str) -> Optional[RecoveryPolicy]:
        versions = [v for (n, v) in self._policies if n == name]
        if not versions:
            return None
        best = max(versions, key=_policy_version_key)
        return self._policies.get((name, best))

    def select(self, name: str, object_hint: str = "") -> Optional[RecoveryPolicy]:
        """Pick the best policy for ``name`` given a runtime object/task hint --
        the deterministic per-object skill selector for the live hot path (no LLM).

        Preference order:
        1. latest version whose ``applies_to`` tag matches ``object_hint``
           (case-insensitive substring, either direction -- so a "carrot" policy
           matches a "pick the carrot" hint and vice versa);
        2. else the latest GENERIC policy (empty ``applies_to``);
        3. else the latest of any (back-compat with pre-tag policies).
        """
        versioned = sorted(
            ((v, p) for (n, v), p in self._policies.items() if n == name),
            key=lambda vp: _policy_version_key(vp[0]),
        )
        if not versioned:
            return None
        hint = (object_hint or "").strip().lower()

        def _tag(p) -> str:
            return (getattr(p, "applies_to", "") or "").strip().lower()

        if hint:
            matched = [p for _, p in versioned if _tag(p) and (_tag(p) in hint or hint in _tag(p))]
            if matched:
                return max(matched, key=lambda p: _policy_version_key(p.version))
        generic = [p for _, p in versioned if not _tag(p)]
        if generic:
            return max(generic, key=lambda p: _policy_version_key(p.version))
        return versioned[-1][1]

    def names(self) -> list[str]:
        return sorted({n for (n, _) in self._policies})

    def keys(self) -> list[tuple[str, str]]:
        return sorted(self._policies.keys())

    def list(self) -> list[RecoveryPolicy]:
        return list(self._policies.values())


# Built-in recovery policy.
def default_hold_lost_policy() -> SequentialPolicy:
    """Rewind to the pre-failure configuration without attempting a new grasp."""
    return SequentialPolicy(
        name="recover_hold_lost",
        version="v0",
        failure_type="hold_lost",
        steps=[
            RecoveryStep("rewind"),
            RecoveryStep("assert_at_rewind"),
        ],
    )


def regrasp_recovery_policy() -> SequentialPolicy:
    """Single-attempt regrasp followed by control handback.

    Retreat to the initialization vantage, perceive and grasp the object, lift to
    check holding, then return to the pre-failure pose and verify. A failed step
    short-circuits the policy and invokes the orchestrator's failure handling."""
    return SequentialPolicy(
        name="recover_hold_lost",
        version="v1",
        failure_type="hold_lost",
        steps=[
            RecoveryStep("goto_init"),
            RecoveryStep("open_gripper"),
            RecoveryStep("perceive_grasp"),
            RecoveryStep("goto_grasp"),
            RecoveryStep("close_gripper"),
            RecoveryStep("lift", {"dz": 0.06}),
            RecoveryStep("assert_holding"),
            RecoveryStep("rewind"),
            RecoveryStep("assert_at_rewind"),
        ],
    )


def default_registry() -> RecoveryPolicyRegistry:
    # Register rewind-only and single-attempt regrasp policies.
    # Additional bounded control flow can be expressed by declarative policies.
    reg = RecoveryPolicyRegistry()
    reg.register(default_hold_lost_policy())
    reg.register(regrasp_recovery_policy())
    return reg
