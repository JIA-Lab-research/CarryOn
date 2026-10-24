"""Typed recovery strategies: failure buckets -> safe policy templates.

This layer is intentionally higher level than ``RecoveryPolicy``. A strategy is
selected from replayable evidence, then compiled into the existing frozen policy
runtime. The hot path never executes arbitrary reflector code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from .policy import RecoveryStep, SequentialPolicy, coerce_steps
from .schema import EpisodeRecord, GroundingResult, HandbackMode
FailureDiagnosis = Literal['generic', 'wrong_object', 'empty_grasp', 'knockdown', 'rewind_fail', 'perception_fail', 'held_object_collision', 'rim_contact', 'insufficient_clearance']
ObjectGeometry = Literal['unknown', 'tall_box', 'cylinder', 'low_object', 'clutter']
GroundingPrimitive = Literal['scene_diff', 'task_target', 'judge_point', 'recent_contact', 'point_model', 'multiview']
PerceptionPrimitive = Literal['single_view_topdown', 'hybrid_auto', 'multiview_obb']
GraspPrimitive = Literal['topdown', 'topdown_family_goalset', 'center_correct', 'yaw_sweep', 'side_grasp', 'topk_goalset']
VerificationPrimitive = Literal['lift_assert', 'retry_then_lift_assert']
HANDBACK_MODES = {'full_hold', 'pre_grasp', 'pre_place', 'complete'}
GROUNDING_PRIMITIVES = {'scene_diff', 'task_target', 'judge_point', 'recent_contact', 'point_model', 'multiview'}
PERCEPTION_PRIMITIVES = {'single_view_topdown', 'hybrid_auto', 'multiview_obb'}
GRASP_PRIMITIVES = {'topdown', 'topdown_family_goalset', 'center_correct', 'yaw_sweep', 'side_grasp', 'topk_goalset'}
VERIFICATION_PRIMITIVES = {'lift_assert', 'retry_then_lift_assert'}
CANONICAL_STRATEGY_BY_BUCKET: dict[str, str] = {'empty_grasp:clutter:tabletop': 'bottle-balanced-goalset', 'empty_grasp:tall_box:tabletop': 'tall-carton-center-correct', 'empty_grasp:cylinder:tabletop': 'cylinder-yaw-or-side', 'empty_grasp:low_object:tabletop': 'low-flat-topdown-deep'}
_GEOMETRY_ALIASES: dict[str, ObjectGeometry] = {'carton': 'tall_box', 'box': 'tall_box', 'bottle': 'clutter'}
_OBJECT_CATEGORY_HINTS: dict[str, ObjectGeometry] = {'ketchup': 'clutter', 'bbq sauce': 'clutter', 'cream cheese': 'low_object', 'chocolate pudding': 'low_object', 'alphabet soup': 'cylinder', 'tomato sauce': 'cylinder', 'salad dressing': 'clutter', 'milk': 'tall_box', 'butter': 'tall_box'}

def normalize_object_geometry(geom: str) -> ObjectGeometry:
    """Map CLI/metadata geometry tags to ``ObjectGeometry`` without object names."""
    g = (geom or 'unknown').strip().lower().replace('-', '_')
    if g in _GEOMETRY_ALIASES:
        return _GEOMETRY_ALIASES[g]
    if g in ('unknown', 'clutter', 'tall_box', 'cylinder', 'low_object'):
        return g
    return 'unknown'

def category_hint_for_object(object_hint: str) -> ObjectGeometry | None:
    """Lookup declared category for an object phrase (routing only, no offsets)."""
    key = (object_hint or '').strip().lower()
    if not key:
        return None
    if key in _OBJECT_CATEGORY_HINTS:
        return _OBJECT_CATEGORY_HINTS[key]
    for phrase, geom in _OBJECT_CATEGORY_HINTS.items():
        if phrase in key or key in phrase:
            return geom
    return None

def resolve_trace_geometry(object_hint: str, *, cli_geometry: str='', inferred_geometry: ObjectGeometry | None=None, reflector_category: ObjectGeometry | None=None, use_hints: bool=False) -> ObjectGeometry:
    """Resolve ``TraceBundle.geometry`` from reflector / CLI / mask / hints."""
    if reflector_category is not None:
        return reflector_category
    if cli_geometry:
        cli = normalize_object_geometry(cli_geometry)
        if cli != 'unknown':
            return cli
    if inferred_geometry is not None and inferred_geometry != 'unknown':
        return inferred_geometry
    if use_hints:
        hinted = category_hint_for_object(object_hint)
        if hinted is not None:
            return hinted
    return 'unknown'

@dataclass(frozen=True)
class FailureBucket:
    diagnosis: FailureDiagnosis = 'generic'
    geometry: ObjectGeometry = 'unknown'
    scene_relation: str = 'tabletop'

    def key(self) -> str:
        return f'{self.diagnosis}:{self.geometry}:{self.scene_relation}'

@dataclass
class TraceBundle:
    """Evidence bundle used by the deterministic strategy selector."""
    goal: str
    object_hint: str = ''
    failure_type: str = 'hold_lost'
    last_failed_op: str = ''
    last_failure_reason: str = ''
    judge_point: Optional[tuple[float, float]] = None
    recent_contact_xy: Optional[tuple[float, float]] = None
    geometry: ObjectGeometry = 'unknown'
    scene_relation: str = 'tabletop'

    @classmethod
    def from_episode(cls, rec: EpisodeRecord, *, object_hint: str='', judge_point: Optional[tuple[float, float]]=None, recent_contact_xy: Optional[tuple[float, float]]=None, geometry: ObjectGeometry='unknown', scene_relation: str='tabletop') -> 'TraceBundle':
        last_op = ''
        last_reason = ''
        failure_type = rec.recovery.failure_type if rec.recovery else 'hold_lost'
        if rec.recovery and rec.recovery.steps:
            for step in rec.recovery.steps:
                if not bool(step.get('ok', False)):
                    last_op = str(step.get('op', ''))
                    last_reason = str(step.get('reason', ''))
                    break
        return cls(goal=rec.goal, object_hint=object_hint, failure_type=failure_type, last_failed_op=last_op, last_failure_reason=last_reason, judge_point=judge_point, recent_contact_xy=recent_contact_xy, geometry=geometry, scene_relation=scene_relation)

    def bucket(self) -> FailureBucket:
        reason = self.last_failure_reason.lower()
        op = self.last_failed_op
        diagnosis: FailureDiagnosis = 'generic'
        if 'rim contact' in reason or 'rim_collision' in reason:
            diagnosis = 'rim_contact'
        elif 'insufficient clearance' in reason or 'clearance too low' in reason:
            diagnosis = 'insufficient_clearance'
        elif ('held object' in reason or 'carried object' in reason) and ('collision' in reason or 'contact' in reason):
            diagnosis = 'held_object_collision'
        elif 'wrong object' in reason or 'wrong-object' in reason:
            diagnosis = 'wrong_object'
        elif op.startswith('perceive_grasp'):
            diagnosis = 'perception_fail'
        elif op == 'assert_holding' or 'not holding' in reason or 'miss' in reason:
            diagnosis = 'empty_grasp'
        elif 'knock' in reason or 'tip' in reason or '倒' in reason:
            diagnosis = 'knockdown'
        elif 'rewind' in op or 'rewind' in reason:
            diagnosis = 'rewind_fail'
        return FailureBucket(diagnosis, self.geometry, self.scene_relation)

@dataclass
class StrategySpec:
    """Propose recovery strategies from failure evidence."""
    strategy_id: str
    trigger: FailureBucket
    validation_buckets: list[FailureBucket]
    grounding_plan: list[GroundingPrimitive] = field(default_factory=lambda: ['task_target'])
    perception_plan: PerceptionPrimitive = 'single_view_topdown'
    grasp_plan: GraspPrimitive = 'topdown'
    verification_plan: VerificationPrimitive = 'lift_assert'
    handback_mode: HandbackMode = 'full_hold'
    params: dict[str, Any] = field(default_factory=dict)
    fallback_strategy_id: str = 'generic-topdown'
    description: str = ''

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.strategy_id:
            problems.append('strategy_id required')
        if not self.validation_buckets:
            problems.append('at least one validation bucket required')
        unknown_grounding = [g for g in self.grounding_plan if g not in GROUNDING_PRIMITIVES]
        if unknown_grounding:
            problems.append(f'unknown grounding primitives: {unknown_grounding}')
        if self.perception_plan not in PERCEPTION_PRIMITIVES:
            problems.append(f'unknown perception primitive: {self.perception_plan}')
        if self.grasp_plan not in GRASP_PRIMITIVES:
            problems.append(f'unknown grasp primitive: {self.grasp_plan}')
        if self.perception_plan == 'hybrid_auto' and self.grasp_plan != 'topk_goalset':
            problems.append('GraspGen/hybrid_auto requires topk_goalset; single goto_grasp discards IK alternatives')
        if self.verification_plan not in VERIFICATION_PRIMITIVES:
            problems.append(f'unknown verification primitive: {self.verification_plan}')
        if self.handback_mode not in HANDBACK_MODES:
            problems.append(f'unknown handback_mode: {self.handback_mode}')
        if self.trigger.diagnosis == 'generic' and self.trigger.geometry == 'unknown' and (self.strategy_id != 'generic-topdown'):
            problems.append('non-generic strategies must declare a non-generic trigger')
        attempts = self.params.get('max_attempts')
        if attempts is not None and (not isinstance(attempts, int) or not 1 <= attempts <= 5):
            problems.append('max_attempts must be an int in [1, 5]')
        top_k = self.params.get('top_k')
        if top_k is not None and (not isinstance(top_k, int) or not 1 <= top_k <= 50):
            problems.append('top_k must be an int in [1, 50]')
        max_tilt = self.params.get('max_tilt_deg')
        if max_tilt is not None:
            try:
                v = float(max_tilt)
            except (TypeError, ValueError):
                problems.append('max_tilt_deg must be a number in [0, 90]')
            else:
                if not 0.0 <= v <= 90.0:
                    problems.append('max_tilt_deg must be a number in [0, 90]')
        band = self.params.get('failopen_tilt_band_deg')
        if band is not None:
            try:
                v = float(band)
            except (TypeError, ValueError):
                problems.append('failopen_tilt_band_deg must be a number in [0, 45]')
            else:
                if not 0.0 <= v <= 45.0:
                    problems.append('failopen_tilt_band_deg must be a number in [0, 45]')
        prefer = self.params.get('prefer_low_tilt')
        if prefer is not None and (not isinstance(prefer, bool)):
            problems.append('prefer_low_tilt must be a bool')
        allow_failopen = self.params.get('allow_tilt_failopen')
        if allow_failopen is not None and (not isinstance(allow_failopen, bool)):
            problems.append('allow_tilt_failopen must be a bool')
        yaw_mode = self.params.get('yaw_mode')
        if yaw_mode is not None and str(yaw_mode).strip().lower() not in ('pca', 'pca_perpendicular', 'topdown'):
            problems.append("yaw_mode must be one of ['pca', 'pca_perpendicular', 'topdown']")
        depth_frac = self.params.get('depth_frac')
        if depth_frac is not None:
            try:
                v = float(depth_frac)
                if not 0.0 <= v <= 0.6:
                    problems.append('depth_frac must be a number in [0, 0.6]')
            except (TypeError, ValueError):
                problems.append('depth_frac must be a number in [0, 0.6]')
        close_width = self.params.get('close_width')
        if close_width is not None:
            try:
                v = float(close_width)
            except (TypeError, ValueError):
                problems.append('close_width must be a number in [0, 0.08]')
            else:
                if not 0.0 <= v <= 0.08:
                    problems.append('close_width must be a number in [0, 0.08]')
        sam_prompts = self.params.get('sam_text_prompts')
        if sam_prompts is not None:
            if not isinstance(sam_prompts, list) or not sam_prompts:
                problems.append('sam_text_prompts must be a non-empty list of strings')
            elif len(sam_prompts) > 5:
                problems.append('sam_text_prompts must have at most 5 entries')
            else:
                for i, p in enumerate(sam_prompts):
                    if not isinstance(p, str) or not p.strip():
                        problems.append(f'sam_text_prompts[{i}] must be a non-empty string')
                    elif len(p.strip()) > 64:
                        problems.append(f'sam_text_prompts[{i}] must be at most 64 chars')
        target_point = self.params.get('target_point')
        if target_point is not None:
            if not isinstance(target_point, (list, tuple)) or len(target_point) != 2:
                problems.append('target_point must be a [x, y] pair')
            else:
                for axis, val in zip('xy', target_point):
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        problems.append(f'target_point.{axis} must be a number in [0, 1]')
                    else:
                        if not 0.0 <= v <= 1.0:
                            problems.append(f'target_point.{axis} must be a normalized fraction in [0, 1]')
        place_clear = self.params.get('place_clearance_m')
        if place_clear is not None:
            try:
                v = float(place_clear)
                if not 0.0 <= v <= 0.15:
                    problems.append('place_clearance_m must be a number in [0, 0.15]')
            except (TypeError, ValueError):
                problems.append('place_clearance_m must be a number in [0, 0.15]')
        vertical_clear = self.params.get('vertical_lift_to_clearance')
        if vertical_clear is not None:
            try:
                v = float(vertical_clear)
                if not 0.0 < v <= 0.2:
                    problems.append('vertical_lift_to_clearance must be a number in (0, 0.20]')
            except (TypeError, ValueError):
                problems.append('vertical_lift_to_clearance must be a number in (0, 0.20]')
            if self.handback_mode != 'pre_place':
                problems.append('vertical_lift_to_clearance requires handback_mode=pre_place')
            if self.trigger.diagnosis not in {'held_object_collision', 'rim_contact', 'insufficient_clearance'}:
                problems.append('reflector may adjust vertical_lift_to_clearance only for held_object_collision, rim_contact, or insufficient_clearance')
        release_bias = self.params.get('place_release_bias_m')
        if release_bias is not None:
            try:
                v = float(release_bias)
                if not 0.0 <= v <= 0.08:
                    problems.append('place_release_bias_m must be a number in [0, 0.08]')
            except (TypeError, ValueError):
                problems.append('place_release_bias_m must be a number in [0, 0.08]')
        return problems

def seed_strategies() -> list[StrategySpec]:
    generic = FailureBucket('generic', 'unknown', 'tabletop')
    return [StrategySpec('generic-topdown', generic, [generic], ['scene_diff', 'judge_point', 'task_target'], perception_plan='hybrid_auto', grasp_plan='topk_goalset', params={'top_k': 15}, description='Generic GraspGen goalset strategy.'), StrategySpec('recent-contact-grounded-topdown', FailureBucket('wrong_object', 'clutter', 'tabletop'), [FailureBucket('wrong_object', 'clutter', 'tabletop')], ['scene_diff', 'judge_point', 'recent_contact', 'task_target'], perception_plan='hybrid_auto', grasp_plan='topk_goalset', params={'top_k': 15}, description='Use recent VLA/EE contact evidence to disambiguate clutter.'), StrategySpec('bottle-low-tilt-goalset', FailureBucket('empty_grasp', 'clutter', 'tabletop'), [FailureBucket('empty_grasp', 'clutter', 'tabletop')], ['scene_diff', 'judge_point', 'recent_contact', 'task_target'], perception_plan='hybrid_auto', grasp_plan='topk_goalset', params={'top_k': 15, 'prefer_low_tilt': True, 'max_tilt_deg': 25.0, 'failopen_tilt_band_deg': 5.0}, description='Constrain top-k goalset to near-top-down bottle grasps; if no candidate is under the cap, fail open only to the flattest small band.'),   StrategySpec('bottle-balanced-goalset', FailureBucket('empty_grasp', 'clutter', 'tabletop'), [FailureBucket('empty_grasp', 'clutter', 'tabletop')], ['scene_diff', 'judge_point', 'recent_contact', 'task_target'], perception_plan='hybrid_auto', grasp_plan='topk_goalset', params={'top_k': 15, 'prefer_low_tilt': True, 'max_tilt_deg': 35.0, 'failopen_tilt_band_deg': 5.0, 'allow_tilt_failopen': False}, verification_plan='retry_then_lift_assert', description='Use the bounded balanced tilt cap with the full grounding chain; reject side-grasp-only goalsets.'),  StrategySpec('tall-carton-center-correct', FailureBucket('empty_grasp', 'tall_box', 'tabletop'), [FailureBucket('empty_grasp', 'tall_box', 'tabletop')], ['scene_diff', 'judge_point', 'task_target'], grasp_plan='center_correct', params={'sam_text_prompts': ['carton', 'box']}, verification_plan='retry_then_lift_assert', description='Correct single-view near-face bias for tall box-like objects; bounded SAM text fallbacks when the task phrase misses.'), StrategySpec('low-flat-topdown-deep', FailureBucket('empty_grasp', 'low_object', 'tabletop'), [FailureBucket('empty_grasp', 'low_object', 'tabletop')], ['scene_diff', 'judge_point', 'task_target'], perception_plan='hybrid_auto', grasp_plan='topk_goalset', params={'top_k': 15, 'prefer_low_tilt': True, 'max_tilt_deg': 25.0}, verification_plan='retry_then_lift_assert', description='For low, flat tabletop objects, retain the GraspGen candidate set and let cuRobo select a reachable low-tilt goal. Use judge_point to localise the object after a drop.'), StrategySpec('cylinder-yaw-or-side', FailureBucket('empty_grasp', 'cylinder', 'tabletop'), [FailureBucket('empty_grasp', 'cylinder', 'tabletop')], ['scene_diff', 'judge_point', 'task_target'], grasp_plan='yaw_sweep', params={'yaw_offset_deg': 90.0, 'z_offset': 0.01}, description='Rotate the top-down pinch for cylindrical objects and raise the contact height before escalating to a side grasp primitive.'), StrategySpec('multiview-obb-grasp', FailureBucket('knockdown', 'unknown', 'tabletop'), [FailureBucket('knockdown', 'unknown', 'tabletop')], ['scene_diff', 'judge_point', 'task_target', 'multiview'], perception_plan='multiview_obb', grasp_plan='center_correct', description='Use multiview 3D geometry when single-view center is unreliable.')]
def resolve_strategy_by_id(strategy_id: str) -> StrategySpec | None:
    """Resolve a strategy identifier from the registered strategy collections."""
    for spec in seed_strategies():
        if spec.strategy_id == strategy_id:
            return spec
    return None

@dataclass
class StrategyResolution:
    """Result of default vs reflector strategy resolution (for attribution)."""
    spec: StrategySpec
    default_spec: StrategySpec
    overridden: bool
    default_id: str
    chosen_id: str
    default_category: ObjectGeometry
    chosen_category: ObjectGeometry

    def attribution(self) -> dict[str, object]:
        return {'default_id': self.default_id, 'chosen_id': self.chosen_id, 'default_category': self.default_category, 'chosen_category': self.chosen_category, 'overridden': self.overridden}

class StrategySelector:

    def __init__(self, strategies: Optional[list[StrategySpec]]=None, *, canonical_by_bucket: Optional[dict[str, str]]=None) -> None:
        self.strategies = strategies or seed_strategies()
        self.canonical_by_bucket = canonical_by_bucket or CANONICAL_STRATEGY_BY_BUCKET

    def select_bucket_class(self, trace: TraceBundle) -> StrategySpec | None:
        """Match strategies by diagnosis, scene relation and perceived geometry."""
        bucket = trace.bucket()
        class_match = [spec for spec in self.strategies if spec.trigger.diagnosis == bucket.diagnosis and spec.trigger.scene_relation == bucket.scene_relation]
        if not class_match:
            return None
        exact = [spec for spec in class_match if spec.trigger.geometry == bucket.geometry]
        candidates = exact or class_match
        canonical_ids = set(self.canonical_by_bucket.values())
        if exact:
            preferred = self.canonical_by_bucket.get(bucket.key())
            if preferred:
                for spec in exact:
                    if spec.strategy_id == preferred:
                        return spec
        for spec in candidates:
            if spec.strategy_id in canonical_ids:
                return spec
        return candidates[0]

    def select(self, trace: TraceBundle) -> StrategySpec:
        bucket = trace.bucket()
        exact = [s for s in self.strategies if s.trigger.diagnosis == bucket.diagnosis and s.trigger.geometry == bucket.geometry and (s.trigger.scene_relation == bucket.scene_relation)]
        if exact:
            preferred = self.canonical_by_bucket.get(bucket.key())
            if preferred:
                for s in exact:
                    if s.strategy_id == preferred:
                        return s
            return exact[0]
        diagnosis_match = [s for s in self.strategies if s.trigger.diagnosis == bucket.diagnosis]
        if diagnosis_match:
            canonical_ids = set(self.canonical_by_bucket.values())
            for s in diagnosis_match:
                if s.strategy_id in canonical_ids:
                    return s
            return diagnosis_match[0]
        return next((s for s in self.strategies if s.strategy_id == 'generic-topdown'))

def resolve_strategy_with_reflector(trace: TraceBundle, proposal: 'StrategyReflectionProposal | None'=None, *, default_trace: TraceBundle | None=None, selector: StrategySelector | None=None) -> StrategyResolution:
    """Resolve strategy: deterministic default, optional reflector override."""
    sel = selector or StrategySelector()
    base_trace = default_trace or trace
    default = sel.select(base_trace)
    base = StrategyResolution(spec=default, default_spec=default, overridden=False, default_id=default.strategy_id, chosen_id=default.strategy_id, default_category=base_trace.geometry, chosen_category=trace.geometry)
    if proposal is None or proposal.strategy is None or (not proposal.override_default):
        if trace.geometry != base_trace.geometry:
            chosen = sel.select(trace)
            return StrategyResolution(spec=chosen, default_spec=default, overridden=chosen.strategy_id != default.strategy_id, default_id=default.strategy_id, chosen_id=chosen.strategy_id, default_category=base_trace.geometry, chosen_category=trace.geometry)
        return base
    chosen = proposal.strategy
    if chosen.validate():
        return base
    lib = resolve_strategy_by_id(chosen.strategy_id)
    if lib is not None:
        chosen = lib
    chosen_category = normalize_object_geometry(proposal.selected_category or chosen.trigger.geometry)
    return StrategyResolution(spec=chosen, default_spec=default, overridden=chosen.strategy_id != default.strategy_id or chosen_category != base_trace.geometry, default_id=default.strategy_id, chosen_id=chosen.strategy_id, default_category=base_trace.geometry, chosen_category=chosen_category)

class StrategyCompiler:
    """Compile a safe StrategySpec into the existing RecoveryPolicy runtime."""

    def compile(self, spec: StrategySpec, *, name: str='recover_hold_lost', version: str='strategy-v1', failure_type: str='hold_lost', skip_init: bool=False, verify_identity: bool=False, allow_retry_body: bool=True) -> SequentialPolicy:
        problems = spec.validate()
        if problems:
            raise ValueError('; '.join(problems))
        source = {'single_view_topdown': 'topdown', 'hybrid_auto': 'auto', 'multiview_obb': 'multiview_obb'}[spec.perception_plan]
        goto_params: dict[str, Any] = {'source_strategy': spec.strategy_id}
        for key in ('x_offset', 'y_offset', 'z_offset'):
            if key in spec.params:
                goto_params[key] = spec.params[key]
        if spec.grasp_plan == 'yaw_sweep' and 'yaw_offset_deg' in spec.params:
            goto_params['yaw_offset_deg'] = spec.params['yaw_offset_deg']
        perceive_extra: dict[str, Any] = {}
        if 'sam_text_prompts' in spec.params:
            perceive_extra['sam_text_prompts'] = [str(p).strip() for p in spec.params['sam_text_prompts'] if str(p).strip()]
        if spec.grasp_plan == 'topdown_family_goalset':
            perceive_step = RecoveryStep('perceive_grasps', {'source': source, 'top_k': int(spec.params.get('top_k', 15)), **perceive_extra})
            goto_step = RecoveryStep('goto_grasp_topdown_family', {'source_strategy': spec.strategy_id, 'yaw_degs': list(spec.params.get('topdown_yaw_degs', [0, 45, 90, 135])), 'z_approach': float(spec.params.get('z_approach', 0.1)), 'fallback_max_tilt_deg': float(spec.params.get('fallback_max_tilt_deg', 40.0))})
        elif spec.grasp_plan == 'topk_goalset':
            perceive_step = RecoveryStep('perceive_grasps', {'source': source, 'top_k': int(spec.params.get('top_k', 15)), **perceive_extra})
            goto_step = RecoveryStep('goto_grasp_goalset', {'source_strategy': spec.strategy_id, 'z_approach': float(spec.params.get('z_approach', 0.1)), **({'prefer_low_tilt': bool(spec.params['prefer_low_tilt'])} if 'prefer_low_tilt' in spec.params else {}), **({'max_tilt_deg': float(spec.params['max_tilt_deg'])} if 'max_tilt_deg' in spec.params else {}), **({'failopen_tilt_band_deg': float(spec.params['failopen_tilt_band_deg'])} if 'failopen_tilt_band_deg' in spec.params else {}), **({'allow_tilt_failopen': bool(spec.params['allow_tilt_failopen'])} if 'allow_tilt_failopen' in spec.params else {})})
        elif spec.grasp_plan in ('center_correct', 'side_grasp'):
            perceive_step = RecoveryStep('perceive_grasps', {'source': source, 'top_k': int(spec.params.get('top_k', 15)), **perceive_extra})
            goto_step = RecoveryStep('goto_grasp', {**goto_params, **({'center_correct': True} if spec.grasp_plan == 'center_correct' else {}), **({'approach': 'side'} if spec.grasp_plan == 'side_grasp' else {}), **({'yaw_mode': str(spec.params['yaw_mode']).strip().lower()} if spec.grasp_plan != 'side_grasp' and 'yaw_mode' in spec.params else {})})
        else:
            perceive_step = RecoveryStep('perceive_grasp', {'source': source, **perceive_extra})
            if spec.grasp_plan == 'topdown':
                goto_params['yaw_mode'] = 'topdown'
            goto_step = RecoveryStep('goto_grasp', goto_params)
        if 'depth_frac' in spec.params and goto_step.op == 'goto_grasp':
            goto_step.params['depth_frac'] = float(spec.params['depth_frac'])
        ground_plan = list(spec.grounding_plan)
        ground_params: dict[str, Any] = {}
        if 'target_point' in spec.params:
            tp = spec.params['target_point']
            ground_params['target_point'] = [float(tp[0]), float(tp[1])]
            ground_plan = ['judge_point', *[g for g in ground_plan if g != 'judge_point']]
        ground_params['plan'] = ground_plan
        if spec.grasp_plan == 'topdown_family_goalset':
            goto_step.params.update({'fresh_retry_max_attempts': min(2, max(1, int(spec.params.get('max_attempts', 2)))), 'fresh_ground_params': dict(ground_params), 'fresh_perceive_params': dict(perceive_step.params)})
        prefix = [RecoveryStep('goto_init'), RecoveryStep('open_gripper'), RecoveryStep('ground_target', ground_params), perceive_step, *([RecoveryStep('verify_object_identity')] if verify_identity else [])]
        close_step = RecoveryStep('close_gripper', {'width': float(spec.params['close_width'])} if 'close_width' in spec.params else {})
        lift_step = RecoveryStep('lift', {'dz': float(spec.params.get('lift_dz', 0.06))})
        if spec.handback_mode == 'pre_grasp':
            pre_goto = RecoveryStep(goto_step.op, {**goto_step.params, 'standoff_only': True})
            body = [*prefix, pre_goto, RecoveryStep('open_gripper')]
            tail: list[RecoveryStep] = []
        elif spec.handback_mode == 'pre_place':
            pre_place_prefix = [step for step in prefix if step.op != 'open_gripper']
            frozen_target_plan = spec.grasp_plan == 'topdown_family_goalset'
            grasp_attempt = [goto_step, close_step, lift_step, RecoveryStep('assert_holding')]
            if spec.verification_plan == 'retry_then_lift_assert' and allow_retry_body:
                if not frozen_target_plan:
                    retry_body = [*pre_place_prefix, *grasp_attempt]
                    grasp_attempt = [RecoveryStep('retry', {'max_attempts': int(spec.params.get('max_attempts', 2)), 'body': retry_body})]
            grasp_body = [*pre_place_prefix, *grasp_attempt] if frozen_target_plan else grasp_attempt
            body = [*grasp_body, *([RecoveryStep('vertical_lift_to_clearance', {'dz': float(spec.params['vertical_lift_to_clearance'])})] if 'vertical_lift_to_clearance' in spec.params else []), RecoveryStep('ground_destination'), RecoveryStep('goto_place', {'standoff_only': True, **({'clearance_m': float(spec.params['place_clearance_m'])} if 'place_clearance_m' in spec.params else {}), **({'release_bias_m': float(spec.params['place_release_bias_m'])} if 'place_release_bias_m' in spec.params else {})}), RecoveryStep('assert_holding')]
            tail = []
        elif spec.handback_mode == 'complete':
            place_step = RecoveryStep('goto_place', {**({'clearance_m': float(spec.params['place_clearance_m'])} if 'place_clearance_m' in spec.params else {}), **({'release_bias_m': float(spec.params['place_release_bias_m'])} if 'place_release_bias_m' in spec.params else {})})
            grasp_body = [*prefix, goto_step, close_step, lift_step, RecoveryStep('assert_holding')]
            if spec.verification_plan == 'retry_then_lift_assert' and allow_retry_body:
                grasp_body = [RecoveryStep('retry', {'max_attempts': int(spec.params.get('max_attempts', 2)), 'body': grasp_body})]
            body = [*grasp_body, place_step, RecoveryStep('open_gripper'), RecoveryStep('verify_place', {'settle_s': float(spec.params.get('place_settle_s', 1.0)), 'min_overlap': float(spec.params.get('place_min_overlap', 0.15))})]
            tail = []
        else:
            body = [*prefix, goto_step, close_step, lift_step, RecoveryStep('assert_holding')]
            if spec.verification_plan == 'retry_then_lift_assert' and allow_retry_body:
                body = [RecoveryStep('retry', {'max_attempts': int(spec.params.get('max_attempts', 2)), 'body': body})]
            tail = [RecoveryStep('rewind'), RecoveryStep('assert_at_rewind')]
        steps = coerce_steps(body + tail)
        if skip_init:
            steps = [s for s in steps if s.op != 'goto_init']
        return SequentialPolicy(name=name, version=version, steps=steps, failure_type=failure_type, handback_mode=spec.handback_mode)

def grounding_from_trace(trace: TraceBundle, plan: list[GroundingPrimitive]) -> GroundingResult:
    """Resolve grounding branches in their declared order."""
    for prim in plan:
        if prim == 'judge_point' and trace.judge_point is not None:
            return GroundingResult(mode='judge_point', provenance='trace.judge_point', normalized_point=[float(trace.judge_point[0]), float(trace.judge_point[1])], object_hint=trace.object_hint, plan=list(plan))
        if prim == 'recent_contact' and trace.recent_contact_xy is not None:
            return GroundingResult(mode='recent_contact', provenance='trace.recent_contact_xy', world_point=[float(trace.recent_contact_xy[0]), float(trace.recent_contact_xy[1])], object_hint=trace.object_hint, plan=list(plan))
        if prim == 'task_target':
            break
    return GroundingResult(mode='task_target', provenance='trace.goal', object_hint=trace.object_hint, plan=list(plan))
