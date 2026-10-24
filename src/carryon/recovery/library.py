"""Versioned library of learned StrategySpec entries.

Each child library records its parent and reflection round. Entries describe
grounding, perception, grasp, verification and handback with bounded parameters.
merge_for_selector layers learned entries over seed_strategies and makes them
canonical for their trigger buckets."""

from __future__ import annotations

import hashlib
import json
import re
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reflection import StrategyReflectionProposal
from .strategy import FailureBucket, StrategySpec

# Remove per-scene coordinate parameters before storing transferable strategies.
# Grounding and perception derive coordinates from each scene at execution time.
_SCENE_SPECIFIC_PARAMS = ("target_point",)


def strip_scene_specific_params(spec: StrategySpec) -> StrategySpec:
    """Return ``spec`` with any per-scene coordinate params removed (a no-op copy
    when there are none). The remaining grounding_plan / perception_plan /
    sam_text_prompts regenerate the point at test time, per scene."""
    if not any(p in spec.params for p in _SCENE_SPECIFIC_PARAMS):
        return spec
    params = {k: v for k, v in spec.params.items() if k not in _SCENE_SPECIFIC_PARAMS}
    return replace(spec, params=params)


def _strict_bool(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")
    return result


def _bucket_to_dict(b: FailureBucket) -> dict[str, Any]:
    return {
        "diagnosis": b.diagnosis,
        "geometry": b.geometry,
        "scene_relation": b.scene_relation,
    }


def spec_to_dict(spec: StrategySpec) -> dict[str, Any]:
    """Serialize a :class:`StrategySpec` to the same dict shape the reflector
    emits, so freeze <-> load round-trips are stable."""
    return {
        "strategy_id": spec.strategy_id,
        "trigger": _bucket_to_dict(spec.trigger),
        "validation_buckets": [_bucket_to_dict(b) for b in spec.validation_buckets],
        "grounding_plan": list(spec.grounding_plan),
        "perception_plan": spec.perception_plan,
        "grasp_plan": spec.grasp_plan,
        "verification_plan": spec.verification_plan,
        "handback_mode": spec.handback_mode,
        "params": dict(spec.params),
        "fallback_strategy_id": spec.fallback_strategy_id,
        "description": spec.description,
    }


def spec_from_dict(d: Mapping[str, Any]) -> StrategySpec:
    """Parse a strategy dict via the reflector's OWN parser
    (:meth:`StrategyReflectionProposal.from_dict`), so a frozen library and a
    live reflector proposal deserialize to byte-identical specs (incl.
    ``handback_mode``)."""
    spec = StrategyReflectionProposal.from_dict({"strategy": dict(d)}).strategy
    if spec is None:
        raise ValueError("strategy dict did not parse into a StrategySpec")
    return spec


def _identity_slug(value: str) -> str:
    """Stable identifier fragment used only by the freeze mechanism."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "unknown"


def _spec_identity_digest(spec: StrategySpec) -> str:
    raw = json.dumps(
        spec_to_dict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def namespace_frozen_strategy_ids(
    specs: Sequence[StrategySpec], *, reserved_ids: Sequence[str] = ()
) -> tuple[StrategySpec, ...]:
    """Resolve reflector identifier collisions without editing proposal files.

    A unique, non-reserved identifier is retained byte-for-byte. When a
    reflector reuses an identifier, the freeze mechanism namespaces every
    member of that collision group by its failure bucket. If distinct payloads
    also share a bucket, a stable digest of the original StrategySpec is added.
    This is deterministic, contains no object-name routing, and leaves every
    field except ``strategy_id`` untouched.
    """
    specs = tuple(specs)
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.strategy_id] = counts.get(spec.strategy_id, 0) + 1

    bucket_counts: dict[tuple[str, str], int] = {}
    for spec in specs:
        bucket = spec.trigger.key()
        key = (spec.strategy_id, bucket)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1

    taken = set(reserved_ids)
    frozen: list[StrategySpec] = []
    for spec in specs:
        original = spec.strategy_id
        collides = counts[original] > 1 or original in taken
        candidate = original
        if collides:
            bucket_slug = "-".join(
                _identity_slug(part)
                for part in (
                    spec.trigger.diagnosis,
                    spec.trigger.geometry,
                    spec.trigger.scene_relation,
                )
            )
            candidate = f"{original}--{bucket_slug}"
            if bucket_counts[(original, spec.trigger.key())] > 1 or candidate in taken:
                candidate = f"{candidate}--{_spec_identity_digest(spec)}"
        if candidate in taken:
            raise ValueError(
                f"frozen strategy identity collision remains for {candidate!r}"
            )
        taken.add(candidate)
        frozen.append(replace(spec, strategy_id=candidate))
    return tuple(frozen)


@dataclass(frozen=True)
class StrategyLibrary:
    """Immutable collection of learned strategies with a content-derived identity."""

    version: str
    parent_version: str | None
    reflection_round: int
    frozen: bool
    strategies: tuple[StrategySpec, ...] = ()
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("library version is required")
        if self.reflection_round < 0:
            raise ValueError("reflection_round must be non-negative")
        object.__setattr__(self, "strategies", tuple(self.strategies))
        seen_ids: set[str] = set()
        for s in self.strategies:
            problems = s.validate()
            if problems:
                raise ValueError(f"invalid strategy {s.strategy_id!r}: {problems}")
            leaked = [p for p in _SCENE_SPECIFIC_PARAMS if p in s.params]
            if leaked:
                raise ValueError(
                    f"frozen strategy {s.strategy_id!r} carries per-scene "
                    f"constant(s) {leaked}; freeze the PLAN, not the coordinate "
                    "(strip via strip_scene_specific_params before freezing)"
                )
            if s.strategy_id in seen_ids:
                raise ValueError(f"duplicate frozen strategy_id {s.strategy_id!r}")
            seen_ids.add(s.strategy_id)
        expected = self.compute_hash()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("strategy library content_hash mismatch")
        object.__setattr__(self, "content_hash", expected)

    def compute_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "parent_version": self.parent_version,
            "reflection_round": self.reflection_round,
            "frozen": self.frozen,
            "strategies": [spec_to_dict(s) for s in self.strategies],
        }
        if include_hash:
            out["content_hash"] = self.content_hash or self.compute_hash()
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyLibrary":
        allowed = {
            "version", "parent_version", "reflection_round", "frozen",
            "content_hash", "strategies",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown strategy-library keys: {sorted(unknown)}")
        return cls(
            version=str(value["version"]),
            parent_version=value.get("parent_version"),
            reflection_round=int(value.get("reflection_round", 0)),
            frozen=_strict_bool(value, "frozen"),
            strategies=tuple(spec_from_dict(s) for s in value.get("strategies", [])),
            content_hash=str(value.get("content_hash", "")),
        )

    @classmethod
    def load(cls, path: str | Path, *, require_frozen: bool = True) -> "StrategyLibrary":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if require_frozen and not raw.get("content_hash"):
            raise ValueError("frozen strategy library must store its content_hash")
        library = cls.from_dict(raw)
        if require_frozen and not library.frozen:
            raise ValueError("runtime requires a frozen strategy library")
        return library


def initial_library(
    specs: Sequence[StrategySpec] = (), *, version: str = "c1.r0"
) -> StrategyLibrary:
    """Create an empty learned library for the initial round."""
    return StrategyLibrary(
        version=version, parent_version=None, reflection_round=0,
        frozen=True, strategies=tuple(strip_scene_specific_params(s) for s in specs),
    )


def freeze_library(
    parent: StrategyLibrary,
    new_specs: Sequence[StrategySpec],
    *,
    version: str,
    reflection_round: int,
) -> StrategyLibrary:
    """Accumulate new specifications onto the parent in the next reflection round.

    Strip per-scene coordinates so perception derives positions from each new scene.
    The same parent and specifications produce the same library identity."""
    if not parent.frozen:
        raise ValueError("reflection parent must itself be frozen")
    if reflection_round != parent.reflection_round + 1:
        raise ValueError("reflection rounds must be sequential")
    cleaned = namespace_frozen_strategy_ids(
        tuple(strip_scene_specific_params(s) for s in new_specs),
        reserved_ids=tuple(s.strategy_id for s in parent.strategies),
    )
    return StrategyLibrary(
        version=version,
        parent_version=parent.version,
        reflection_round=reflection_round,
        frozen=True,
        strategies=tuple(parent.strategies) + cleaned,
    )


def write_frozen_library(library: StrategyLibrary, path: str | Path) -> None:
    if not library.frozen:
        raise ValueError("only frozen artifacts may be written")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(library.to_dict(), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def merge_for_selector(
    library: StrategyLibrary,
) -> tuple[list[StrategySpec], dict[str, str]]:
    """``(strategies, canonical_by_bucket)`` for a :class:`StrategySelector`:
    the ``seed_strategies()`` base plus the library's learned specs, with each
    learned spec made the canonical default for its trigger bucket so it wins
    over the seed default. Bucket-keyed only -- no object-name routing."""
    from .strategy import CANONICAL_STRATEGY_BY_BUCKET, seed_strategies

    specs = list(seed_strategies()) + list(library.strategies)
    canonical = dict(CANONICAL_STRATEGY_BY_BUCKET)
    for s in library.strategies:
        canonical[s.trigger.key()] = s.strategy_id
    return specs, canonical
