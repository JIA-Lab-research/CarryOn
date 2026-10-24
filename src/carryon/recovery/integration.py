"""Connect recovery policy selection, recording and control handoff."""

from __future__ import annotations

import asyncio
import os
import re as _re
from dataclasses import replace
from typing import Any, Awaitable, Callable, Optional

from ..types import RecoveryHookResult
from .arbiter import GelloArbiter
from .gello_client import GelloControlClient
from .orchestrator import RecoveryOrchestrator, RecoveryResult
from .policy import RecoveryConfig, RecoveryPolicyRegistry, StepResult
from .schema import EpisodeRecord

RecordProvider = Callable[[Any], Optional[EpisodeRecord]]
HintProvider = Callable[[Any], str]
# Optional coarse normalized (x, y) anchor for the recovery target (Judge point).
PointProvider = Callable[[Any], Optional[tuple]]
LogFn = Callable[[str, Optional[str], Optional[str]], None]
CompleteFn = Callable[[Any, "RecoveryResult"], None]


def _destination_from_instruction(instruction) -> Optional[str]:
    """Extract the place DESTINATION noun from a pick-place instruction:
    "pick up the ketchup and place it in the basket" -> "basket".
    Object-agnostic: the noun phrase after the last in/into/on/onto/inside
    "the". Returns None when no destination phrase is present."""
    if not instruction:
        return None
    matches = _re.findall(
        r"\b(?:in|into|on|onto|inside)\s+the\s+([a-zA-Z][a-zA-Z \-]*?)"
        r"(?:\.|,|;|$)",
        str(instruction),
    )
    if not matches:
        return None
    dest = matches[-1].strip()
    return dest or None


def render_pre_place_residual_instruction(instruction: str) -> str:
    """Render a generic residual from original instruction roles only."""
    text = str(instruction or "").strip()
    destination = _destination_from_instruction(text)
    if not destination:
        raise ValueError("pre_place residual requires an instruction destination")
    match = _re.search(
        r"\b(?:pick\s+up|grasp|take)\s+the\s+(.+?)\s+and\s+"
        r"(?:place|put|drop)\b",
        text,
        flags=_re.IGNORECASE,
    )
    target = match.group(1).strip() if match else "target object"
    return (
        f"You are holding the {target} above the {destination}. "
        "Continue by lowering, release, and confirm that it is stable."
    )


class RecoveryCoordinator:
    def __init__(
        self,
        *,
        backend,
        gello: GelloControlClient,
        registry: RecoveryPolicyRegistry,
        record_provider: RecordProvider,
        object_hint_provider: HintProvider,
        resume_instruction_provider: HintProvider,
        target_px_provider: Optional[PointProvider] = None,
        config: Optional[RecoveryConfig] = None,
        policy_name: str = "recover_hold_lost",
        policy_version: Optional[str] = None,
        failure_type: str = "hold_lost",
        enabled: bool = True,
        max_attempts: int = 1,
        log: Optional[LogFn] = None,
        on_complete: Optional[CompleteFn] = None,
        use_strategy_selector: bool = False,
        reflector=None,
        strategy_library_specs: "Sequence[Any]" = (),
        strategy_canonical_by_bucket: Optional[dict] = None,
        intervention_rule: bool = False,
        strict_reflected_library: bool = False,
        strategy_handback_mode: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.gello = gello
        self.registry = registry
        self.record_provider = record_provider
        self.object_hint_provider = object_hint_provider
        self.target_px_provider = target_px_provider
        self.resume_instruction_provider = resume_instruction_provider
        self.config = config or RecoveryConfig()
        self.policy_name = policy_name
        self.policy_version = policy_version
        self.failure_type = failure_type
        self.enabled = enabled
        # Closed-loop cap: the recovery policy is single-shot, but a job that
        # re-fails after a handback re-triggers recovery. This bounds how many
        # times one subtask (job_key) may do so before the plan finalizes failed.
        self.max_attempts = max(1, int(max_attempts))
        self._attempts: dict[str, int] = {}
        self._log = log
        # Called after a recovery attempt produces a result, so the host can
        # persist the recovery trace back onto the episode (the snapshot the
        # orchestrator stamps is throwaway) and re-dump it for reflection.
        self._on_complete = on_complete
        # Strategy-library routing (config recovery.use_strategy_selector). When
        # enabled, each recovery selects a GATED seed strategy by failure bucket
        # (diagnosis x perception-inferred geometry -- no object-name priors) and
        # runs its compiled policy; anything failing (compile, inference,
        # selection) falls back to the pinned default policy. All seed strategies
        # are compiled ONCE here so the hot path never touches the compiler.
        self._strategy_policies: dict[str, tuple[str, str]] = {}
        # Select from seed and learned strategies by diagnosis, geometry and scene relation.
        self._strategy_selector = None
        # Online reflection: consulted only when a job re-triggers after a
        # failed recovery attempt (keyed by job); None uses the library only.
        self._reflector = reflector
        self._intervention_rule = bool(intervention_rule)
        self._strict_reflected_library = bool(strict_reflected_library)
        if strategy_handback_mode not in (None, "", "full_hold", "pre_grasp", "pre_place", "complete"):
            raise ValueError(
                f"invalid condition strategy handback mode: {strategy_handback_mode!r}"
            )
        self._strategy_handback_mode = strategy_handback_mode or None
        self._library_specs = tuple(strategy_library_specs)
        self._last_failed_invocation: dict[str, Any] = {}
        if use_strategy_selector and (
            not self._strict_reflected_library or self._library_specs
        ):
            from .strategy import (
                CANONICAL_STRATEGY_BY_BUCKET, StrategyCompiler, StrategySelector,
                seed_strategies,
            )

            # Learned entries are canonical for their trigger buckets before selection.
            all_specs = (
                list(self._library_specs)
                if self._strict_reflected_library
                else list(seed_strategies()) + list(self._library_specs)
            )
            canonical = (
                dict(strategy_canonical_by_bucket or {})
                if self._strict_reflected_library
                else dict(strategy_canonical_by_bucket or CANONICAL_STRATEGY_BY_BUCKET)
            )
            self._strategy_selector = StrategySelector(
                strategies=all_specs,
                canonical_by_bucket=canonical,
            )
            for spec in all_specs:
                try:
                    compile_spec = (
                        replace(spec, handback_mode=self._strategy_handback_mode)
                        if self._strategy_handback_mode is not None
                        else spec
                    )
                    pol = StrategyCompiler().compile(
                        compile_spec,
                        version=(
                            f"strat-{spec.strategy_id}"
                            + (f"-{self._strategy_handback_mode}"
                               if self._strategy_handback_mode else "")
                        ),
                        failure_type=self.failure_type,
                        verify_identity=bool(getattr(self.config, "verify_identity", False)),
                        allow_retry_body=not bool(
                            getattr(self.config, "library_oneshot", False)
                        ),
                    )
                    self.registry.register(pol)
                    self._strategy_policies[spec.strategy_id] = (pol.name, pol.version)
                except Exception as e:  # noqa: BLE001 - skip uncompilable spec
                    if self._log is not None:
                        try:
                            self._log(
                                f"strategy {spec.strategy_id} failed to compile: "
                                f"{type(e).__name__}: {e}", None, None,
                            )
                        except Exception:
                            pass

    def _emit(self, msg: str, rec: Any) -> None:
        if self._log is not None:
            try:
                self._log(msg, getattr(rec, "plan_id", None), getattr(rec, "subtask_id", None))
            except Exception:
                pass

    def _maybe_signal_complete(self, result, request_id, rec) -> None:
        """Coordinate simulator completion for non-full_hold handback modes.

        On success, complete requests settling and scoring; pre_grasp resumes VLA
        control from the standoff. Both require acknowledgement. full_hold uses the
        regular handback path."""
        inv = result.invocation
        if getattr(inv, "outcome", None) != "success":
            return
        hb_mode = getattr(inv, "handback_mode", "full_hold")
        ok = True
        label = ""
        try:
            if hb_mode == "complete":
                label = "complete_terminal_ack"
                ok = bool(self.gello.complete_terminal(request_id))
            elif hb_mode == "pre_grasp":
                label = "pregrasp_resume_ack"
                ok = bool(self.gello.pregrasp_resume(request_id))
            else:
                return
        except Exception as e:  # noqa: BLE001
            ok = False
            self._emit(f"{label} raised: {type(e).__name__}: {e}", rec)
        inv.steps = list(inv.steps or [])
        inv.steps.append({
            "op": label,
            "ok": ok,
            "reason": "world-thread acknowledged" if ok else "missing or failed acknowledgement",
            "dt_s": 0.0,
        })
        if not ok:
            inv.outcome = "failed"
            note = f"{label} failed closed"
            inv.notes = f"{inv.notes}; {note}" if inv.notes else note
        self._emit(f"{label} ok={ok}", rec)


    @staticmethod
    def _terminal_handback_attempted(result: Optional[RecoveryResult]) -> bool:
        if result is None:
            return False
        return any(
            str(step.get("op", "")) == "failure_handback_ack"
            for step in list(result.invocation.steps or [])
            if isinstance(step, dict)
        )

    @staticmethod
    def _terminal_handback_ok(result: Optional[RecoveryResult]) -> bool:
        if result is None:
            return False
        return any(
            str(step.get("op", "")) == "failure_handback_ack"
            and bool(step.get("ok"))
            for step in list(result.invocation.steps or [])
            if isinstance(step, dict)
        )

    def _ensure_aborted_handback(
        self,
        result: RecoveryResult,
        arbiter: GelloArbiter,
        rec: Any,
    ) -> None:
        """An outer early-takeover can precede an orchestrator abort."""
        if result.invocation.outcome not in ("failed", "aborted"):
            return
        if self._terminal_handback_attempted(result):
            return
        ok = False
        reason = None
        try:
            ok = bool(arbiter.handback())
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if reason is None:
            reason = getattr(arbiter, "last_handback_reason", None)
        step = StepResult(
            "failure_handback_ack",
            ok,
            "VLA resume acknowledged" if ok else str(reason or "refused"),
            dt_s=0.0,
        )
        result.steps.append(step)
        result.invocation.steps = list(result.invocation.steps or [])
        result.invocation.steps.append(step.to_dict())
        if not ok:
            result.invocation.notes = (
                ((result.invocation.notes + " | ") if result.invocation.notes else "")
                + "handback_failed: " + str(reason or "refused")
            )
        self._emit(
            "preflight/aborted recovery terminal -> mandatory VLA handback "
            f"{'ACK' if ok else 'FAILED'}",
            rec,
        )

    def _resume_without_recovery(self, rec: Any, reason: str) -> bool:
        """Reopen VLA when a bounded recovery attempt is intentionally skipped."""
        request_id = getattr(rec, "job_key", "") or getattr(rec, "vla_job_id", "")
        instruction = self.resume_instruction_provider(rec)
        arbiter = GelloArbiter(
            self.gello,
            request_id=request_id,
            resume_instruction=instruction,
            callback_url=getattr(rec, "callback_url", ""),
        )
        took_over = False
        try:
            took_over = bool(arbiter.takeover())
        except Exception as exc:
            self._emit(f"{reason}: takeover raised {type(exc).__name__}: {exc}", rec)
        ok = False
        try:
            ok = bool(arbiter.handback())
        except Exception as exc:
            self._emit(f"{reason}: handback raised {type(exc).__name__}: {exc}", rec)
        event = {
            "op": "failure_handback_ack",
            "ok": ok,
            "reason": "VLA resume acknowledged" if ok else str(
                getattr(arbiter, "last_handback_reason", None) or "refused"
            ),
            "takeover_ack": took_over,
            "terminal_reason": reason,
        }
        setattr(rec, "recovery_terminal_handback_event", event)
        self._emit(
            f"{reason}: mandatory VLA handback {'ACK' if ok else 'FAILED'}",
            rec,
        )
        return ok

    def recover_sync(self, rec: Any) -> Optional[RecoveryResult]:
        """Blocking recovery for one job record. Returns ``None`` if it could not
        even start (disabled / no record), else a :class:`RecoveryResult`."""
        if not self.enabled:
            return None
        job_key = getattr(rec, "job_key", "") or getattr(rec, "vla_job_id", "")
        prior = self._attempts.get(job_key, 0)
        if prior >= self.max_attempts:
            setattr(rec, "terminal_reason_hint", "cap_refusal")
            self._emit(
                f"recovery cap reached ({prior}/{self.max_attempts}); not re-trying "
                "recovery, returning the remaining horizon to VLA",
                rec,
            )
            self._resume_without_recovery(rec, "cap_refusal")
            return None
        # Use CarryOn's submitted request_id (job_key), not gello's returned
        # job_id. The gello control server tracks _active_request_id by the
        # request_id sent in /execute (e.g. "plan-...:t1"). Cancelling the
        # returned "gello-..." job_id only latches standby for an unknown id; it
        # does NOT mark the active task cancelled, so the VLA policy keeps
        # streaming and fights recovery on the shared bridge.
        request_id = job_key
        instruction = self.resume_instruction_provider(rec)
        # Stop the arm before materializing the recorder window or running any
        # routing/perception. Recorder.build can copy images and trajectory
        # buffers; allowing that work to precede standby races a finite VLA
        # horizon even though recovery itself freezes the budget.
        arbiter = GelloArbiter(
            self.gello, request_id=request_id, resume_instruction=instruction,
            callback_url=getattr(rec, "callback_url", ""),
        )
        try:
            if arbiter.takeover():
                self._emit("judge failed -> VLA standby (takeover before recorder/routing)", rec)
            else:
                self._emit(
                    "early takeover not confirmed "
                    f"({arbiter.last_takeover_reason or 'no reason'}); "
                    "orchestrator will retry", rec,
                )
        except Exception as e:  # noqa: BLE001 - early standby must not kill recovery
            self._emit(f"early takeover raised: {type(e).__name__}: {e}", rec)

        record = self.record_provider(rec)
        if record is None:
            self._emit("recovery skipped: no EpisodeRecord for job", rec)
            ok = False
            try:
                ok = bool(arbiter.handback())
            except Exception as exc:
                self._emit(
                    f"missing-record mandatory handback raised: {type(exc).__name__}: {exc}",
                    rec,
                )
            setattr(
                rec,
                "recovery_terminal_handback_event",
                {
                    "op": "failure_handback_ack",
                    "ok": ok,
                    "reason": "VLA resume acknowledged" if ok else str(
                        getattr(arbiter, "last_handback_reason", None) or "refused"
                    ),
                    "takeover_ack": True,
                    "terminal_reason": "missing_record",
                },
            )
            return None
        self._attempts[job_key] = prior + 1


        try:
            self.backend.task_instruction = str(instruction or "")
        except Exception:  # noqa: BLE001
            pass
        object_hint = self.object_hint_provider(rec)
        target_px = self.target_px_provider(rec) if self.target_px_provider else None
        # Reset identity exclusions once at episode start so they persist across recovery rounds.
        # Exclude the destination from grasp-target identity candidates.
        if prior == 0:
            try:
                self.backend.identity_rejected = []
            except Exception:  # noqa: BLE001
                pass
            dest = _destination_from_instruction(instruction)
            if (
                dest
                and dest.lower() not in str(object_hint or "").lower()
                and os.environ.get("CARRYON_EXCLUDE_DESTINATION", "1") != "0"
                and hasattr(self.backend, "exclude_destination")
            ):
                try:
                    _ok = bool(self.backend.exclude_destination(dest))
                    self._emit(
                        "destination %r pre-excluded from grasp candidates: %s"
                        % (dest, _ok), rec,
                    )
                except Exception as _e:  # noqa: BLE001
                    self._emit(
                        "destination pre-exclusion raised: %s: %s"
                        % (type(_e).__name__, _e), rec,
                    )
        # Strategy-library routing: bucket = failure diagnosis x perception-
        # inferred geometry (calibrated, hint-free). Selection and compilation
        # are deterministic; any failure falls back to the default policy.
        policy_name = self.policy_name
        policy_version = self.policy_version
        inferred_geometry = "unknown"
        if self._intervention_rule and prior > 0:
            prior_inv = self._last_failed_invocation.get(job_key)
            mode = self._rule_handback_mode(prior_inv)
            if mode != "full_hold":
                rule_policy = self._compile_rule_policy(mode, prior)
                policy_name, policy_version = rule_policy.name, rule_policy.version
            self._emit(
                f"C1-rule trace ladder selected handback_mode={mode} "
                f"(no reflector)", rec,
            )
        if self._strategy_policies:
            try:
                from .geometry import infer_geometry_from_backend
                from .strategy import (
                    StrategySelector, TraceBundle, resolve_trace_geometry,
                )

                inferred = infer_geometry_from_backend(
                    self.backend, object_hint, target_px=target_px
                )
                geometry = resolve_trace_geometry(
                    object_hint, inferred_geometry=inferred, use_hints=False
                )
                inferred_geometry = geometry
                trace = TraceBundle(
                    goal=instruction or "",
                    object_hint=object_hint,
                    last_failed_op="assert_holding",
                    last_failure_reason="gripper not holding object",
                    geometry=geometry,
                )
                # After a failed attempt for this job, one bounded VLM
                # reflection may override the deterministic pick. The proposal
                # passes the same safety shell as any reflector output
                # (structural gate + StrategySpec.validate inside
                # resolve_strategy_with_reflector); anything invalid falls open
                # to the library default.
                proposal = None
                prior_inv = self._last_failed_invocation.get(job_key)
                if self._reflector is not None and prior_inv is not None:
                    self._emit(
                        "re-triggered after a failed recovery -> consulting the "
                        "reflector (C2)", rec,
                    )
                    try:
                        from .strategy import seed_strategies as _seeds
                        proposal = self._reflector.propose(
                            record=record,
                            object_hint=object_hint,
                            geometry=geometry,
                            failed_invocation=prior_inv,
                            library_ids=[x.strategy_id for x in _seeds()],
                        )
                    except Exception as e:  # noqa: BLE001 - reflection fails open
                        self._emit(
                            f"reflector raised ({type(e).__name__}: {e}); "
                            "using library default", rec,
                        )
                from .strategy import resolve_strategy_with_reflector
                if self._strict_reflected_library:
                    bucket = trace.bucket()
                    spec = (
                        self._strategy_selector.select_bucket_class(trace)
                        if self._strategy_selector is not None else None
                    )
                    route_match = (
                        "exact" if spec is not None
                        and spec.trigger.key() == bucket.key()
                        else "bucket_class"
                    )
                    resolution = None
                else:
                    resolution = resolve_strategy_with_reflector(
                        trace, proposal, selector=self._strategy_selector
                    )
                    spec = resolution.spec
                routed = (
                    self._strategy_policies.get(spec.strategy_id)
                    if spec is not None else None
                )
                if (
                    routed is None
                    and resolution is not None
                    and resolution.overridden
                ):
                    # Compile the proposed strategy through the policy runtime and register it for this run.
                    from .strategy import StrategyCompiler
                    import uuid as _uuid
                    pol = StrategyCompiler().compile(
                        spec,
                        name=self.policy_name,
                        version=f"reflector-{spec.strategy_id}-{_uuid.uuid4().hex[:6]}",
                        failure_type=self.failure_type,
                        verify_identity=bool(getattr(self.config, "verify_identity", False)),
                    )
                    self.registry.register(pol)
                    self._strategy_policies[spec.strategy_id] = (pol.name, pol.version)
                    routed = (pol.name, pol.version)
                if routed is not None:
                    policy_name, policy_version = routed
                    tag = (
                        " (reflector override)"
                        if resolution is not None and resolution.overridden else ""
                    )
                    match_tag = (
                        f" match={route_match}" if self._strict_reflected_library else ""
                    )
                    self._emit(
                        f"strategy routing: geometry={geometry}{match_tag} -> "
                        f"{spec.strategy_id} ({policy_name}@{policy_version}){tag}",
                        rec,
                    )
            except Exception as e:  # noqa: BLE001 - routing fails open
                self._emit(
                    f"strategy routing failed ({type(e).__name__}: {e}); "
                    "using default policy", rec,
                )
                policy_name, policy_version = self.policy_name, self.policy_version
        # Use the configured default policy for the initial attempt.
        # Retain the default first attempt until a reflection proposal is available.
        if not self._strategy_policies and self._reflector is not None:
            _prior = self._last_failed_invocation.get(job_key)
            if _prior is not None:
                try:
                    from .geometry import infer_geometry_from_backend
                    from .strategy import (
                        StrategyCompiler, TraceBundle,
                        resolve_strategy_with_reflector, resolve_trace_geometry,
                        seed_strategies as _seeds,
                    )
                    import uuid as _uuid
                    _inferred = infer_geometry_from_backend(
                        self.backend, object_hint, target_px=target_px)
                    _geo = resolve_trace_geometry(
                        object_hint, inferred_geometry=_inferred, use_hints=False)
                    inferred_geometry = _geo
                    self._emit(
                        "re-triggered (naive-C1 path) -> consulting the reflector (C2)", rec)
                    _proposal = self._reflector.propose(
                        record=record, object_hint=object_hint, geometry=_geo,
                        failed_invocation=_prior,
                        library_ids=[x.strategy_id for x in _seeds()],
                    )
                    _trace = TraceBundle(
                        goal=instruction or "", object_hint=object_hint,
                        last_failed_op="assert_holding",
                        last_failure_reason="gripper not holding object", geometry=_geo)
                    _res = resolve_strategy_with_reflector(
                        _trace, _proposal, selector=self._strategy_selector
                    )
                    if _res.overridden and _res.spec is not None:
                        _pol = StrategyCompiler().compile(
                            _res.spec, name=self.policy_name,
                            version=f"reflector-{_res.spec.strategy_id}-{_uuid.uuid4().hex[:6]}",
                            failure_type=self.failure_type,
                            verify_identity=bool(getattr(self.config, "verify_identity", False)),
                        )
                        self.registry.register(_pol)
                        policy_name, policy_version = _pol.name, _pol.version
                        self._emit(
                            f"re-trigger reflect -> {_res.spec.strategy_id} "
                            f"(handback_mode={_res.spec.handback_mode}) "
                            f"{policy_name}@{policy_version}", rec)
                except Exception as e:  # noqa: BLE001 - reflection fails open to default
                    self._emit(
                        f"re-trigger reflect failed ({type(e).__name__}: {e}); default", rec)
        # Per-object skill selection (deterministic, no LLM on the hot path): when
        # the version is not pinned, pick the policy whose ``applies_to`` tag best
        # matches this object_hint, falling back to the latest generic policy.
        if policy_version is None:
            selected = self.registry.select(policy_name, object_hint)
            if selected is not None:
                policy_version = selected.version
                tag = getattr(selected, "applies_to", "") or "generic"
                self._emit(
                    f"selected recovery skill {self.policy_name}@{policy_version} "
                    f"(applies_to='{tag}') for object='{object_hint}'",
                    rec,
                )
        # Use an explicitly supplied compiled policy when configured.
        _force_hb = os.environ.get("CARRYON_FORCE_HANDBACK_MODE", "").strip()
        _forced_preplace = False
        _preplace_op_sequence: Optional[tuple[str, ...]] = None
        if _force_hb == "pre_place":
            # PrePlace is assembled from the already factory-gated authored
            # policy, preserving its multi-candidate goalset regrasp. This path
            # is fail-loud: a method mismatch can never silently change modes.
            from .preplace import (
                PREPLACE_CLEARANCE_GENERATION,
                PREPLACE_GENERATION,
                PrePlaceContractError,
                assemble_preplace_policy,
            )
            _fg = os.environ.get("CARRYON_FORCE_GRASP", "").strip()
            if _fg != "topk_goalset":
                raise PrePlaceContractError(
                    "PrePlace requires CARRYON_FORCE_GRASP=topk_goalset; "
                    f"got {_fg!r}"
                )
            _base = self.registry.get(policy_name, policy_version)
            if _base is None:
                raise PrePlaceContractError(
                    f"base policy unavailable: {policy_name}@{policy_version}"
                )
            _clearance_raw = os.environ.get(
                "CARRYON_VERTICAL_LIFT_TO_CLEARANCE_M", ""
            ).strip()
            _generation = (
                PREPLACE_CLEARANCE_GENERATION
                if _clearance_raw
                else PREPLACE_GENERATION
            )
            _clearance = float(_clearance_raw) if _clearance_raw else None
            _assembly = assemble_preplace_policy(
                _base,
                generation=_generation,
                vertical_lift_to_clearance=_clearance,
            )
            if self.registry.get(
                _assembly.policy.name, _assembly.policy.version
            ) is None:
                self.registry.register(_assembly.policy)
            policy_name = _assembly.policy.name
            policy_version = _assembly.policy.version
            _forced_preplace = True
            _preplace_op_sequence = _assembly.op_sequence
            self._emit(
                "[preplace-assembly] "
                f"generation={_generation} "
                f"effective_policy_version={policy_version} "
                "effective_handback_mode=pre_place fallback=false "
                f"ops={','.join(_assembly.op_sequence)}",
                rec,
            )
        elif _force_hb in ("full_hold", "pre_grasp", "complete"):
            try:
                from .strategy import FailureBucket, StrategyCompiler, StrategySpec
                _geo = inferred_geometry if inferred_geometry != "unknown" else "clutter"
                _bkt = FailureBucket("empty_grasp", _geo)
                _fg = os.environ.get(
                    "CARRYON_FORCE_GRASP", "center_correct"
                ).strip()
                _fparams = {"top_k": 15}
                if _fg in ("center_correct", "topdown"):
                    _fparams.update({"yaw_mode": "topdown", "depth_frac": 0.45})
                _spec = StrategySpec(
                    f"forced-{_force_hb}",
                    _bkt,
                    [_bkt],
                    ["task_target", "scene_diff", "judge_point"],
                    perception_plan="hybrid_auto",
                    grasp_plan=_fg,
                    handback_mode=_force_hb,
                    params=_fparams,
                )
                _pol = StrategyCompiler().compile(
                    _spec,
                    name=self.policy_name,
                    version=f"forced-{_force_hb}",
                    failure_type=self.failure_type,
                    verify_identity=bool(
                        getattr(self.config, "verify_identity", False)
                    ),
                )
                if self.registry.get(_pol.name, _pol.version) is None:
                    self.registry.register(_pol)
                policy_name, policy_version = _pol.name, _pol.version
                self._emit(
                    f"FORCED handback_mode={_force_hb} -> "
                    f"{policy_name}@{policy_version}",
                    rec,
                )
            except Exception as _e:  # noqa: BLE001
                self._emit(
                    f"force handback_mode failed "
                    f"({type(_e).__name__}: {_e}); default",
                    rec,
                )
        self._emit(f"judge failed -> recovery taking over (object='{object_hint}')", rec)
        # Reflect on a failed recovery round and retry within the same takeover,
        # subject to the configured inline-round limit.
        reflect_turns = max(0, int(getattr(self.config, "reflect_turns", 1)))
        will_reflect = (
            self._reflector is not None
            and reflect_turns > 0
            and (prior + 2) <= self.max_attempts
        )
        # Return control after an inline reflection round.
        cfg_round1 = self.config
        selected_policy = self.registry.get(policy_name, policy_version)
        if (
            selected_policy is not None
            and getattr(selected_policy, "handback_mode", "full_hold") == "pre_place"
        ):
            try:
                arbiter.resume_instruction = render_pre_place_residual_instruction(
                    instruction
                )
            except ValueError as exc:
                self._emit(
                    f"pre_place residual unavailable: {exc}; "
                    "retaining the frozen resume_instruction for mandatory handback",
                    rec,
                )
        orch = RecoveryOrchestrator(
            self.backend, self.registry, config=cfg_round1, arbiter=arbiter,
            log=lambda msg: self._emit(msg, rec),
        )
        result = orch.recover(
            record,
            object_hint=object_hint,
            target_px=target_px,
            policy_name=policy_name,
            policy_version=policy_version,
            failure_type=self.failure_type,
            triggered_ts=record.first_fail_ts,
            request_id=request_id,
            reset_identity=False,  # coordinator reset once at episode start
        )
        self._ensure_aborted_handback(result, arbiter, rec)
        if _forced_preplace:
            result.invocation.fallback_used = False
            result.invocation.effective_op_sequence = list(
                _preplace_op_sequence or ()
            )
        self._emit(
            f"recovery {policy_name}@{result.invocation.policy_version} "
            f"-> {result.invocation.outcome}: {result.invocation.notes or ''}",
            rec,
        )
        self._maybe_signal_complete(result, request_id, rec)
        turn = 0
        while (
            will_reflect
            and result.invocation.outcome == "failed"
            and not self._terminal_handback_attempted(result)
            and turn < reflect_turns
            and (prior + 2 + turn) <= self.max_attempts
        ):
            turn += 1
            self._attempts[job_key] = prior + 1 + turn
            self._emit(
                f"round {turn} failed -> consulting the reflector "
                f"(turn {turn}/{reflect_turns})", rec,
            )
            proposal = None
            try:
                from .strategy import seed_strategies as _seeds
                proposal = self._reflector.propose(
                    record=record,
                    object_hint=object_hint,
                    geometry=inferred_geometry,
                    failed_invocation=result.invocation,
                    library_ids=[x.strategy_id for x in _seeds()],
                )
            except Exception as e:  # noqa: BLE001 - reflection fails open
                self._emit(
                    f"reflector raised ({type(e).__name__}: {e}); "
                    "retrying with the library default", rec,
                )
            r2_name, r2_version = policy_name, policy_version
            try:
                from .strategy import (
                    StrategyCompiler, TraceBundle, resolve_strategy_with_reflector,
                )
                # Descend through retry children to identify the first failed primitive.
                def _first_failed_leaf(steps):
                    for st in steps or []:
                        get = st.get if isinstance(st, dict) else lambda k, d=None: getattr(st, k, d)
                        if get("ok", True):
                            continue
                        kids = get("children") or []
                        leaf = _first_failed_leaf(kids)
                        if leaf is not None:
                            return leaf
                        return (get("op", "") or "", get("reason", "") or "")
                    return None
                leaf = _first_failed_leaf(result.invocation.steps)
                last_op, last_reason = leaf if leaf else ("", "")
                trace2 = TraceBundle(
                    goal=instruction or "",
                    object_hint=object_hint,
                    last_failed_op=last_op or "assert_holding",
                    last_failure_reason=last_reason or "gripper not holding object",
                    geometry=inferred_geometry,
                )
                resolution = resolve_strategy_with_reflector(
                    trace2, proposal, selector=self._strategy_selector
                )
                spec2 = resolution.spec
                routed2 = self._strategy_policies.get(spec2.strategy_id)
                if routed2 is None and resolution.overridden:
                    import uuid as _uuid
                    pol2 = StrategyCompiler().compile(
                        spec2,
                        name=self.policy_name,
                        version=f"reflector-{spec2.strategy_id}-{_uuid.uuid4().hex[:6]}",
                        failure_type=self.failure_type,
                        verify_identity=bool(getattr(self.config, "verify_identity", False)),
                    )
                    self.registry.register(pol2)
                    self._strategy_policies[spec2.strategy_id] = (pol2.name, pol2.version)
                    routed2 = (pol2.name, pol2.version)
                if routed2 is not None:
                    r2_name, r2_version = routed2
                tag2 = " (reflector override)" if resolution.overridden else " (library default)"
                self._emit(
                    f"reflected retry: {spec2.strategy_id} "
                    f"({r2_name}@{r2_version}){tag2}", rec,
                )
            except Exception as e:  # noqa: BLE001 - fall back to round-1 routing
                self._emit(
                    f"reflected re-route failed ({type(e).__name__}: {e}); "
                    "retrying round-1 policy", rec,
                )
            _last_possible = (
                turn >= reflect_turns or (prior + 2 + turn) > self.max_attempts
            )
            orch2 = RecoveryOrchestrator(
                self.backend, self.registry,
                config=(self.config if _last_possible else cfg_round1),
                arbiter=arbiter,
                log=lambda msg: self._emit(msg, rec),
            )
            result = orch2.recover(
                record,
                object_hint=object_hint,
                target_px=target_px,
                policy_name=r2_name,
                policy_version=r2_version,
                failure_type=self.failure_type,
                triggered_ts=record.first_fail_ts,
                request_id=request_id,
                reset_identity=False,  # persist exclusions across rounds
            )
            self._emit(
                f"reflected recovery {r2_name}@{result.invocation.policy_version} "
                f"-> {result.invocation.outcome}: {result.invocation.notes or ''}",
                rec,
            )
            self._maybe_signal_complete(result, request_id, rec)
        if self._on_complete is not None:
            try:
                self._on_complete(rec, result)
            except Exception as e:  # noqa: BLE001 -- persistence must not break recovery
                self._emit(f"recovery on_complete raised: {type(e).__name__}: {e}", rec)
        # Remember the last recovery attempt for reflection after a re-trigger.
        # Store successful attempts too: a chain_ok recovery that is
        # then re-triggered means the env failed AFTER handback, and the reflector
        # needs that chain_ok invocation to propose a deeper intervention
        # (complete/pre_grasp). Only consumed if a re-trigger actually occurs (the
        # judge re-fires); a genuinely-finished episode never calls recover() again.
        self._last_failed_invocation[job_key] = result.invocation
        return result

    @staticmethod
    def _rule_handback_mode(invocation) -> str:
        """Apply the fixed recovery ladder to the recorded recovery trace."""
        if invocation is None:
            return "full_hold"
        if getattr(invocation, "outcome", None) == "success":
            return "complete"

        def flatten(steps):
            for step in steps or []:
                yield step
                children = step.get("children") if isinstance(step, dict) else None
                if children:
                    yield from flatten(children)

        flat = list(flatten(getattr(invocation, "steps", None)))
        reachable = any(
            isinstance(step, dict)
            and step.get("op") in {"goto_grasp", "goto_grasp_goalset"}
            and bool(step.get("ok"))
            for step in flat
        )
        holding_failed = any(
            isinstance(step, dict)
            and step.get("op") == "assert_holding"
            and not bool(step.get("ok"))
            for step in flat
        )
        return "pre_grasp" if reachable and holding_failed else "full_hold"

    def _compile_rule_policy(self, mode: str, prior: int):
        from .strategy import FailureBucket, StrategyCompiler, StrategySpec

        bucket = FailureBucket("empty_grasp", "unknown", "tabletop")
        spec = StrategySpec(
            strategy_id="generic-topdown",
            trigger=bucket,
            validation_buckets=[bucket],
            grounding_plan=["task_target"],
            perception_plan="single_view_topdown",
            grasp_plan="topdown",
            verification_plan="lift_assert",
            handback_mode=mode,
            params={},
            description="fixed non-reflective intervention ladder control",
        )
        policy = StrategyCompiler().compile(
            spec,
            name=self.policy_name,
            version=f"c1-rule-{mode}-r{prior + 1}",
            failure_type=self.failure_type,
            verify_identity=False,
            allow_retry_body=False,
        )
        self.registry.register(policy)
        return policy

    async def on_judge_failure(self, rec: Any) -> RecoveryHookResult:
        """JobManager hook. Runs blocking recovery off the loop.

        Returns the tri-valued ``RecoveryHookResult`` (see carryon.types). This
        is intentionally broader than ``result.ok``: a recovery policy can fail
        after takeover (e.g. grasp miss) and still leave gello safely in standby
        after a fail-safe rewind (``"handled_failed"``). In that case the manager
        must NOT send a second judge-driven cancel(return_home=True), which would
        make the arm move again after recovery has already stopped it.
        """
        try:
            result = await asyncio.get_running_loop().run_in_executor(None, self.recover_sync, rec)
        except Exception as e:  # noqa: BLE001
            setattr(rec, "terminal_reason_hint", "exception")
            self._emit(f"recovery hook raised: {type(e).__name__}: {e}", rec)
            return False
        if result is None:
            event = getattr(rec, "recovery_terminal_handback_event", None)
            return "resumed" if isinstance(event, dict) and event.get("ok") else False
        if result.invocation.outcome == "success":
            return "resumed"
        if self._terminal_handback_ok(result):
            setattr(rec, "terminal_reason_hint", "recovery_failed_handed_back")
            return "resumed"
        if result.invocation.outcome == "failed":
            notes = str(result.invocation.notes or "")
            setattr(
                rec,
                "terminal_reason_hint",
                "handback_failure" if "handback_failed:" in notes else "normal_failure",
            )
            return "handled_failed"
        return False
