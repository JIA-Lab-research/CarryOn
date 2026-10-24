"""Reflect on recorded recovery attempts and validate policy proposals.

An injected ReflectionLLM proposes declarative steps from the allowed operation
vocabulary. Structural checks validate operations, parameters, ordering and
version identity; evidence checks validate the source episode records before
registration. Registration follows the cap-x-style versioned approach."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence
from .assertions import run_checks
from .geometry import GeometryValidator
from .policy import MAX_RETRY_ATTEMPTS, OPS, RecoveryPolicyRegistry, RecoveryStep, SequentialPolicy, coerce_steps
from .schema import EpisodeRecord
from .strategy import FailureBucket, StrategySpec
_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {'goto_pose': ('position', 'quaternion')}

def _all_steps(steps: Sequence[RecoveryStep]) -> list[RecoveryStep]:
    """Every step incl. those nested inside ``retry`` bodies (for op/param checks)."""
    out: list[RecoveryStep] = []
    for s in steps:
        out.append(s)
        if s.op == 'retry':
            out.extend(_all_steps(coerce_steps(s.params.get('body'))))
    return out

def _flatten_ops(steps: Sequence[RecoveryStep]) -> list[str]:
    """Op sequence with ``retry`` bodies expanded *inline* (the construct is pure
    grouping), so the existing recovery invariants can be checked on a flat order
    without special-casing nesting."""
    out: list[str] = []
    for s in steps:
        if s.op == 'retry':
            out.extend(_flatten_ops(coerce_steps(s.params.get('body'))))
        else:
            out.append(s.op)
    return out

@dataclass
class PolicyProposal:
    """An LLM-proposed declarative policy (improvement signal, pre-gate)."""
    name: str
    version: str
    steps: list[RecoveryStep]
    failure_type: str = 'hold_lost'
    rationale: str = ''
    source_episode_ids: list[str] = field(default_factory=list)
    applies_to: str = ''
    handback_mode: str = 'full_hold'
    escalate: bool = False
    escalate_reason: str = ''

    @classmethod
    def from_dict(cls, d: dict) -> 'PolicyProposal':
        steps = coerce_steps(d.get('steps', []))
        return cls(name=d.get('name', ''), version=d.get('version', ''), steps=steps, failure_type=d.get('failure_type', 'hold_lost'), rationale=d.get('rationale', ''), source_episode_ids=list(d.get('source_episode_ids', [])), applies_to=str(d.get('applies_to', '') or ''), handback_mode=str(d.get('handback_mode', 'full_hold') or 'full_hold'), escalate=bool(d.get('escalate', False)), escalate_reason=str(d.get('escalate_reason', '') or ''))

    def to_policy(self) -> SequentialPolicy:
        return SequentialPolicy(name=self.name, version=self.version, steps=list(self.steps), failure_type=self.failure_type, applies_to=self.applies_to, handback_mode=self.handback_mode)

@dataclass
class PrimitiveProposal:
    """Record a proposed primitive for later implementation; it is not executable."""
    op_name: str
    rationale: str = ''
    code_sketch: str = ''
    source_episode_ids: list[str] = field(default_factory=list)

@dataclass
class StrategyReflectionProposal:
    """Reflector output for the Recovery Strategy layer."""
    strategy: Optional[StrategySpec] = None
    primitive_proposals: list[PrimitiveProposal] = field(default_factory=list)
    rationale: str = ''
    source_episode_ids: list[str] = field(default_factory=list)
    selected_category: Optional[str] = None
    override_default: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'StrategyReflectionProposal':
        strat = None
        if d.get('strategy'):
            strat_obj = dict(d['strategy'])
            trig = strat_obj.get('trigger', {})
            buckets = strat_obj.get('validation_buckets', [])
            strat = StrategySpec(strategy_id=str(strat_obj.get('strategy_id', '')), trigger=FailureBucket(**trig), validation_buckets=[FailureBucket(**b) for b in buckets], grounding_plan=list(strat_obj.get('grounding_plan', ['task_target'])), perception_plan=strat_obj.get('perception_plan', 'single_view_topdown'), grasp_plan=strat_obj.get('grasp_plan', 'topdown'), verification_plan=strat_obj.get('verification_plan', 'lift_assert'), handback_mode=strat_obj.get('handback_mode', 'full_hold'), params=dict(strat_obj.get('params', {})), fallback_strategy_id=str(strat_obj.get('fallback_strategy_id', 'generic-topdown')), description=str(strat_obj.get('description', '')))
        prims = [PrimitiveProposal(op_name=str(p.get('op_name', '')), rationale=str(p.get('rationale', '')), code_sketch=str(p.get('code_sketch', '')), source_episode_ids=list(p.get('source_episode_ids', []))) for p in d.get('primitive_proposals', []) or []]
        return cls(strategy=strat, primitive_proposals=prims, rationale=str(d.get('rationale', '')), source_episode_ids=list(d.get('source_episode_ids', [])), selected_category=d.get('selected_category'), override_default=bool(d.get('override_default', False)))

@dataclass
class ProposalIssue:
    id: str
    ok: bool
    detail: str = ''

@dataclass
class ProposalReport:
    name: str
    version: str
    issues: list[ProposalIssue] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    frozen: bool = False

    @property
    def structural_ok(self) -> bool:
        return all((i.ok for i in self.issues))

    @property
    def ok(self) -> bool:
        return self.structural_ok

    def summary(self) -> str:
        bad = [i.id for i in self.issues if not i.ok]
        head = f"{self.name}@{self.version}: {('OK' if self.ok else 'REJECTED')}"
        if bad:
            head += f" (failed: {', '.join(bad)})"
        if self.frozen:
            head += ' [frozen]'
        return head

class ReflectionLLM(Protocol):
    """Injected reflection model. Real impl wraps an LLM client; given the recorded
    episodes (and a summary of the current library) it returns ONE proposal."""

    def propose(self, episodes: Sequence[EpisodeRecord], library: list[tuple[str, str]]) -> PolicyProposal:
        ...

def _check_structure(p: PolicyProposal, registry: RecoveryPolicyRegistry) -> list[ProposalIssue]:
    issues: list[ProposalIssue] = []

    def add(cid: str, ok: bool, detail: str='') -> None:
        issues.append(ProposalIssue(cid, ok, detail))
    add('has_identity', bool(p.name) and bool(p.version), 'name and version required')
    add('nonempty', len(p.steps) > 0, 'policy has no steps')
    all_steps = _all_steps(p.steps)
    unknown = [s.op for s in all_steps if s.op not in OPS]
    add('ops_known', not unknown, f'unknown ops: {unknown}' if unknown else '')
    missing_params = [s.op for s in all_steps if s.op in _REQUIRED_PARAMS and (not all((k in s.params for k in _REQUIRED_PARAMS[s.op])))]
    add('params_present', not missing_params, f'missing required params for: {missing_params}' if missing_params else '')
    retry_problems: list[str] = []
    for s in all_steps:
        if s.op != 'retry':
            continue
        body = coerce_steps(s.params.get('body'))
        ma = s.params.get('max_attempts')
        if not body:
            retry_problems.append('empty body')
        elif not body[-1].op.startswith('assert_'):
            retry_problems.append(f"body must end on an assert_* (got '{body[-1].op}')")
        if not isinstance(ma, int) or isinstance(ma, bool) or (not 1 <= ma <= MAX_RETRY_ATTEMPTS):
            retry_problems.append(f'max_attempts must be int in [1, {MAX_RETRY_ATTEMPTS}] (got {ma!r})')
    add('retry_bounded', not retry_problems, '; '.join(retry_problems))
    ops = _flatten_ops(p.steps)
    pre_place = p.handback_mode == 'pre_place'
    add('handback_mode_known', p.handback_mode in {'full_hold', 'pre_grasp', 'pre_place', 'complete'}, f'unsupported handback_mode: {p.handback_mode!r}')
    add('has_rewind', pre_place or 'rewind' in ops, "non-pre_place policy must contain a 'rewind' step")
    if pre_place:
        add('pre_place_destination', 'ground_destination' in ops and 'goto_place' in ops, 'pre_place requires ground_destination followed by goto_place')
        first_close = ops.index('close_gripper') if 'close_gripper' in ops else -1
        add('pre_place_no_release', first_close >= 0 and 'open_gripper' not in ops[first_close + 1:], 'pre_place may not open the gripper after the first close')
    has_perception = 'perceive_grasp' in ops or 'perceive_grasps' in ops
    restores_hold = has_perception and 'close_gripper' in ops
    add('restores_hold', restores_hold, 'policy must re-grasp (perceive[_s] + close_gripper)')
    if 'goto_grasp' in ops:
        perception_indices = [ops.index(op) for op in ('perceive_grasp', 'perceive_grasps') if op in ops]
        add('perceive_before_goto', bool(perception_indices) and min(perception_indices) < ops.index('goto_grasp'), 'goto_grasp requires a preceding perceive_grasp or perceive_grasps')
    if 'goto_grasp_goalset' in ops:
        add('perceive_many_before_goalset', ops.index('perceive_grasps') < ops.index('goto_grasp_goalset') if 'perceive_grasps' in ops else False, 'goto_grasp_goalset requires a preceding perceive_grasps')
    add('ends_with_assert', ops[-1].startswith('assert_') if ops else False, 'policy must end on an assert_* verification step')
    add('version_new', registry.get(p.name, p.version) is None, f'{p.name}@{p.version} already registered (versions are immutable)')
    reopen_problems: list[str] = []
    for s in all_steps:
        if s.op != 'retry':
            continue
        body_ops = [b.op for b in coerce_steps(s.params.get('body'))]
        if 'close_gripper' in body_ops:
            ci = body_ops.index('close_gripper')
            if 'open_gripper' not in body_ops[:ci]:
                reopen_problems.append("retry body must 'open_gripper' before 'close_gripper' (each attempt must reopen the jaws)")
    if 'close_gripper' in ops:
        ci = ops.index('close_gripper')
        if 'open_gripper' not in ops[:ci]:
            reopen_problems.append("a 'close_gripper' must be preceded by an 'open_gripper'")
    add('reopen_before_regrasp', not reopen_problems, '; '.join(reopen_problems))
    if 'rewind' in ops:
        last_rw = len(ops) - 1 - ops[::-1].index('rewind')
        after_rewind = [o for o in ops[last_rw + 1:] if not o.startswith('assert_')]
        add('rewind_is_terminal', not after_rewind, f"only assert_* may follow 'rewind'; got arm-moving ops after it: {after_rewind}" if after_rewind else '')
    return issues

def _bucket_key(bucket: FailureBucket) -> str:
    return bucket.key()

def _check_strategy_structure(p: StrategyReflectionProposal) -> list[ProposalIssue]:
    issues: list[ProposalIssue] = []

    def add(cid: str, ok: bool, detail: str='') -> None:
        issues.append(ProposalIssue(cid, ok, detail))
    add('has_strategy_or_primitive_proposal', p.strategy is not None or bool(p.primitive_proposals), 'reflector output must contain a StrategySpec or a primitive proposal')
    if p.strategy is not None:
        problems = p.strategy.validate()
        add('strategy_spec_valid', not problems, '; '.join(problems))
    for prim in p.primitive_proposals:
        add(f"primitive_proposal_offline_only:{prim.op_name or '<missing>'}", False, 'new primitive proposals are recorded for offline review; they are not hot-path deployable')
    return issues
_GROUNDING_DIAGNOSES = {'wrong_object', 'perception_fail'}

def _is_grounding_bucket(bucket: FailureBucket) -> bool:
    return bucket.diagnosis in _GROUNDING_DIAGNOSES

def _check_bucket_validation(strategy: StrategySpec, bucket_results: dict[str, dict], *, min_improvement: float=0.05, max_neighbor_drop: float=0.05, min_dist_improvement: float=0.05, max_neighbor_dist_regression: float=0.05) -> list[ProposalIssue]:
    """Bucket-type-aware improvement gate. Both metrics are read as candidate -
    baseline DELTAS -- never an absolute threshold, never a raw GT coordinate:

    * grasp-hold buckets (slip / miss): ``candidate_success >= baseline_success +
      min_improvement`` (assert_holding rate; GT anchors perception identically
      in both arms and cancels in the delta).
    * grounding buckets (wrong-object / perception): ``candidate_grounding_dist
      <= baseline_grounding_dist - min_dist_improvement`` -- each arm does its OWN
      grounding (GT is NOT the perception anchor here, or it would mask the very
      thing under test); GT true-object-location is used ONLY as the yardstick
      that turns each arm's grounded point into a distance, and the gate reads the
      distance DELTA.
    """
    issues: list[ProposalIssue] = []

    def add(cid: str, ok: bool, detail: str='') -> None:
        issues.append(ProposalIssue(cid, ok, detail))
    for bucket in strategy.validation_buckets:
        key = _bucket_key(bucket)
        row = bucket_results.get(key)
        if not row:
            add(f'bucket_result_present:{key}', False, 'missing bucket-level probe result')
            continue
        n = int(row.get('n', 0))
        add(f'bucket_sample_size:{key}', n > 0, f'n must be > 0 (got {n})')
        if _is_grounding_bucket(bucket):
            cand = row.get('candidate_grounding_dist')
            base = row.get('baseline_grounding_dist')
            if cand is None or base is None:
                add(f'grounding_metric_present:{key}', False, 'grounding bucket requires baseline/candidate_grounding_dist (point-to-true-object distance) probe results')
                continue
            cand, base = (float(cand), float(base))
            add(f'bucket_improves:{key}', cand <= base - min_dist_improvement, f'candidate_grounding_dist={cand:.3f}, baseline_grounding_dist={base:.3f}, required <= baseline - {min_dist_improvement:.3f} (closer to true object)')
        else:
            cand = float(row.get('candidate_success', 0.0))
            base = float(row.get('baseline_success', 0.0))
            add(f'bucket_improves:{key}', cand >= base + min_improvement, f'candidate_success={cand:.3f}, baseline_success={base:.3f}, required +{min_improvement:.3f}')
    for key, row in bucket_results.items():
        if not bool(row.get('non_regression', False)):
            continue
        if row.get('candidate_grounding_dist') is not None and row.get('baseline_grounding_dist') is not None:
            cand = float(row['candidate_grounding_dist'])
            base = float(row['baseline_grounding_dist'])
            add(f'neighbor_non_regression:{key}', cand <= base + max_neighbor_dist_regression, f'candidate_grounding_dist={cand:.3f}, baseline_grounding_dist={base:.3f}, max regress {max_neighbor_dist_regression:.3f}')
        else:
            cand = float(row.get('candidate_success', 0.0))
            base = float(row.get('baseline_success', 0.0))
            add(f'neighbor_non_regression:{key}', cand >= base - max_neighbor_drop, f'candidate_success={cand:.3f}, baseline_success={base:.3f}, max drop {max_neighbor_drop:.3f}')
    return issues

class ReflectionEngine:
    """Drives reflect -> gate -> freeze. Pure orchestration over injected seams."""

    def __init__(self, registry: RecoveryPolicyRegistry, llm: Optional[ReflectionLLM]=None, *, geometry: Optional[GeometryValidator]=None, require_evidence: bool=True) -> None:
        self.registry = registry
        self.llm = llm
        self.geometry = geometry
        self.require_evidence = require_evidence

    def validate(self, proposal: PolicyProposal, episodes: Sequence[EpisodeRecord]) -> ProposalReport:
        report = ProposalReport(proposal.name, proposal.version)
        report.issues = _check_structure(proposal, self.registry)
        ep_by_id = {e.episode_id: e for e in episodes}
        used = [ep_by_id[i] for i in proposal.source_episode_ids if i in ep_by_id]
        if self.require_evidence:
            report.issues.append(ProposalIssue('evidence_present', len(used) > 0, 'no recorded source episode to justify the change' if not used else ''))
        for ep in used:
            rep = run_checks(ep, geometry=self.geometry)
            report.evidence.append(rep.summary())
            report.issues.append(ProposalIssue(f'evidence_freezable:{ep.episode_id}', rep.freezable, '' if rep.freezable else f'blocking checks failed: {[r.id for r in rep.blocking_failures]}'))
        return report

    def validate_strategy(self, proposal: StrategyReflectionProposal, episodes: Sequence[EpisodeRecord], *, bucket_results: Optional[dict[str, dict]]=None) -> ProposalReport:
        """Validate a strategy proposal; reject primitives outside the allowed vocabulary."""
        name = proposal.strategy.strategy_id if proposal.strategy else 'primitive-proposal'
        report = ProposalReport(name, 'strategy')
        report.issues = _check_strategy_structure(proposal)
        ep_by_id = {e.episode_id: e for e in episodes}
        used = [ep_by_id[i] for i in proposal.source_episode_ids if i in ep_by_id]
        if self.require_evidence:
            report.issues.append(ProposalIssue('evidence_present', len(used) > 0, 'no recorded source episode to justify the strategy' if not used else ''))
        for ep in used:
            rep = run_checks(ep, geometry=self.geometry)
            report.evidence.append(rep.summary())
            report.issues.append(ProposalIssue(f'evidence_freezable:{ep.episode_id}', rep.freezable, '' if rep.freezable else f'blocking checks failed: {[r.id for r in rep.blocking_failures]}'))
        if proposal.strategy is not None:
            report.issues.extend(_check_bucket_validation(proposal.strategy, bucket_results or {}))
        return report

    def freeze(self, proposal: PolicyProposal, report: ProposalReport) -> bool:
        """Register the proposal iff its report is clean. Returns whether frozen."""
        if not report.ok:
            return False
        self.registry.register(proposal.to_policy())
        report.frozen = True
        return True

    def reflect(self, episodes: Sequence[EpisodeRecord]) -> tuple[PolicyProposal, ProposalReport]:
        """Full cycle: ask the LLM for a proposal, gate it, freeze if clean."""
        if self.llm is None:
            raise RuntimeError('ReflectionEngine has no LLM wired')
        proposal = self.llm.propose(episodes, self.registry.keys())
        report = self.validate(proposal, episodes)
        self.freeze(proposal, report)
        return (proposal, report)

def load_authored_policy(registry: RecoveryPolicyRegistry, path: str) -> tuple[str, str, bool, list[str]]:
    """Validate a reflection response and register the resulting policy."""
    import json as _json
    try:
        spec = _json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        return ('', '', False, [f'could not read {path!r}: {type(e).__name__}: {e}'])
    try:
        proposal = PolicyProposal.from_dict(spec)
    except Exception as e:
        return ('', '', False, [f'malformed policy {path!r}: {type(e).__name__}: {e}'])
    report = ReflectionEngine(registry, require_evidence=False).validate(proposal, [])
    if not report.structural_ok:
        problems = [f'{i.id}: {i.detail}' for i in report.issues if not i.ok]
        return (proposal.name, proposal.version, False, problems)
    registry.register(proposal.to_policy())
    return (proposal.name, proposal.version, True, [])
