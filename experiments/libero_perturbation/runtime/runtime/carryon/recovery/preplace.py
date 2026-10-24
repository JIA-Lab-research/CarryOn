"""Assemble and validate the PrePlace recovery policy."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .policy import RecoveryPolicyRegistry, RecoveryStep, SequentialPolicy, default_registry
from .reflection import load_authored_policy
PREPLACE_GENERATION = 'topdown_preplace/v1'
PREPLACE_VERSION_SUFFIX = '__TOPDOWN_PREPLACE_V1'
PREPLACE_CLEARANCE_GENERATION = 'topdown_preplace_clearance/v1'
PREPLACE_CLEARANCE_VERSION_SUFFIX = '__TOPDOWN_PREPLACE_CLEARANCE_V1'

class PrePlaceContractError(RuntimeError):
    """The configured runtime cannot execute the required PrePlace method."""

@dataclass(frozen=True)
class PrePlaceAssembly:
    generation: str
    policy: SequentialPolicy
    base_policy_version: str
    effective_policy_version: str
    effective_handback_mode: str
    fallback_used: bool
    op_sequence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {'generation': self.generation, 'base_policy_version': self.base_policy_version, 'effective_policy_version': self.effective_policy_version, 'effective_handback_mode': self.effective_handback_mode, 'fallback_used': self.fallback_used, 'op_sequence': list(self.op_sequence)}

def _first_index(ops: list[str], value: str, *, after: int=-1) -> int:
    try:
        return ops.index(value, after + 1)
    except ValueError as exc:
        raise PrePlaceContractError(f'missing required op: {value}') from exc

def validate_preplace_policy(policy: SequentialPolicy, *, fallback_used: bool, require_vertical_clearance: bool=False) -> tuple[str, ...]:
    """Validate the effective policy, including the staged open-gripper rule."""
    errors: list[str] = []
    ops = [step.op for step in policy.steps]
    if policy.handback_mode != 'pre_place':
        errors.append(f'handback_mode={policy.handback_mode!r}')
    if fallback_used:
        errors.append('fallback_used=true')
    if ops.count('goto_grasp_goalset') != 1:
        errors.append(f"goto_grasp_goalset_count={ops.count('goto_grasp_goalset')}")
    if 'goto_grasp' in ops:
        errors.append('single_goto_grasp_forbidden')
    expected_clearance_count = 1 if require_vertical_clearance else 0
    if ops.count('vertical_lift_to_clearance') != expected_clearance_count:
        errors.append(f"vertical_lift_to_clearance_count={ops.count('vertical_lift_to_clearance')} expected={expected_clearance_count}")
    try:
        goalset = _first_index(ops, 'goto_grasp_goalset')
        close = _first_index(ops, 'close_gripper', after=goalset)
        holding_1 = _first_index(ops, 'assert_holding', after=close)
        vertical_clearance = _first_index(ops, 'vertical_lift_to_clearance', after=holding_1) if require_vertical_clearance else holding_1
        destination = _first_index(ops, 'ground_destination', after=vertical_clearance)
        place = _first_index(ops, 'goto_place', after=destination)
        holding_2 = _first_index(ops, 'assert_holding', after=place)
        if not policy.steps[place].params.get('standoff_only') is True:
            errors.append('goto_place_not_standoff_only')
        opens = [index for index, op in enumerate(ops) if op == 'open_gripper']
        if len([index for index in opens if index < goalset]) > 1:
            errors.append('more_than_one_pregrasp_open_gripper')
        if any((index > close and index < len(ops) for index in opens)):
            errors.append('post_close_open_gripper')
        if any((index > holding_1 and index < len(ops) for index in opens)):
            errors.append('post_holding_open_gripper')
        if holding_2 != len(ops) - 1:
            errors.append('second_holding_not_terminal_policy_op')
    except PrePlaceContractError as exc:
        errors.append(str(exc))
    if 'perceive_grasps' not in ops:
        errors.append('multi_candidate_perception_missing')
    forbidden = sorted({'rewind', 'assert_at_rewind', 'verify_place'} & set(ops))
    if forbidden:
        errors.append(f'forbidden_tail_ops={forbidden}')
    if errors:
        raise PrePlaceContractError('; '.join(errors))
    return tuple(ops)

def assemble_preplace_policy(base: SequentialPolicy, *, generation: str=PREPLACE_GENERATION, vertical_lift_to_clearance: float | None=None) -> PrePlaceAssembly:
    """Preserve the authored goalset regrasp and replace only its handback tail."""
    if generation not in {PREPLACE_GENERATION, PREPLACE_CLEARANCE_GENERATION}:
        raise PrePlaceContractError(f'generation mismatch: {generation!r}')
    if generation == PREPLACE_GENERATION and vertical_lift_to_clearance is not None:
        raise PrePlaceContractError('PrePlace without clearance cannot carry a clearance distance')
    require_vertical_clearance = generation == PREPLACE_CLEARANCE_GENERATION
    if require_vertical_clearance:
        try:
            vertical_lift_to_clearance = float(vertical_lift_to_clearance)
        except (TypeError, ValueError) as exc:
            raise PrePlaceContractError('clearance generation requires a finite scalar') from exc
        if not 0.0 < vertical_lift_to_clearance <= 0.2:
            raise PrePlaceContractError('vertical_lift_to_clearance must be in (0, 0.20] m')
    base_ops = [step.op for step in base.steps]
    if 'goto_grasp' in base_ops:
        raise PrePlaceContractError('base policy contains single goto_grasp')
    goalset = _first_index(base_ops, 'goto_grasp_goalset')
    close = _first_index(base_ops, 'close_gripper', after=goalset)
    holding = _first_index(base_ops, 'assert_holding', after=close)
    prefix = [RecoveryStep(step.op, dict(step.params)) for step in base.steps[:holding + 1]]
    policy = SequentialPolicy(name=base.name, version=base.version + (PREPLACE_CLEARANCE_VERSION_SUFFIX if require_vertical_clearance else PREPLACE_VERSION_SUFFIX), failure_type=base.failure_type, applies_to=base.applies_to, handback_mode='pre_place', steps=[*prefix, *([RecoveryStep('vertical_lift_to_clearance', {'dz': vertical_lift_to_clearance})] if require_vertical_clearance else []), RecoveryStep('ground_destination'), RecoveryStep('goto_place', {'standoff_only': True}), RecoveryStep('assert_holding')])
    ops = validate_preplace_policy(policy, fallback_used=False, require_vertical_clearance=require_vertical_clearance)
    return PrePlaceAssembly(generation=generation, policy=policy, base_policy_version=base.version, effective_policy_version=policy.version, effective_handback_mode=policy.handback_mode, fallback_used=False, op_sequence=ops)

def load_and_assemble_preplace_policy(policy_path: str | Path, *, expected_name: str='recover_hold_lost', generation: str=PREPLACE_GENERATION, vertical_lift_to_clearance: float | None=None, registry: RecoveryPolicyRegistry | None=None) -> PrePlaceAssembly:
    """Use the same authored-policy gate as factory, then assemble PrePlace."""
    target = Path(policy_path).resolve()
    if not target.is_file():
        raise PrePlaceContractError(f'authored policy missing: {target}')
    active_registry = registry or default_registry()
    name, version, ok, problems = load_authored_policy(active_registry, str(target))
    if not ok:
        raise PrePlaceContractError(f'authored policy rejected: name={name!r} version={version!r} problems={problems}')
    if name != expected_name:
        raise PrePlaceContractError(f'policy name mismatch: expected={expected_name!r} actual={name!r}')
    base = active_registry.get(name, version)
    if not isinstance(base, SequentialPolicy):
        raise PrePlaceContractError(f'authored policy is not SequentialPolicy: {type(base).__name__}')
    return assemble_preplace_policy(base, generation=generation, vertical_lift_to_clearance=vertical_lift_to_clearance)
