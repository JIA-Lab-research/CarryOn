"""Load position-correction memory and select entries by category."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any
SCHEMA_VERSION = 'c2p-positionfix-memory/v1'
GENERATION = 'C2P_POSITIONFIX_G1'
TOP_KEYS = {'schema_version', 'generation', 'frozen', 'bootstrap', 'rules', 'content_hash'}
BOOTSTRAP_KEYS = {'total_failures', 'generation_counts', 'partition_sha256', 'build_rule'}
RULE_KEYS = {'rule_id', 'object_category', 'failure_category', 'source_type', 'component_selection', 'anchor_gate_frac', 'min_component_pixels', 'max_component_area_frac', 'max_height_above_table_m', 'support_failure_rollouts', 'support_source_attempts', 'description'}

def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')

def _content_hash(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop('content_hash', None)
    return hashlib.sha256(_canonical(body)).hexdigest()

def normalize_category(value: str) -> str:
    normalized = re.sub('[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')
    if not normalized:
        raise ValueError('object category must be non-empty')
    return normalized

@dataclass(frozen=True)
class PositionFixRule:
    rule_id: str
    object_category: str
    failure_category: str
    source_type: str
    component_selection: str
    anchor_gate_frac: float
    min_component_pixels: int
    max_component_area_frac: float
    max_height_above_table_m: float
    support_failure_rollouts: int
    support_source_attempts: int
    description: str

@dataclass(frozen=True)
class PositionFixMemory:
    version: str
    generation: str
    content_hash: str
    rules: tuple[PositionFixRule, ...]

    @classmethod
    def load(cls, path: str | Path, *, require_frozen: bool=True) -> 'PositionFixMemory':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
            raise ValueError('positionfix memory top-level schema mismatch')
        if payload['schema_version'] != SCHEMA_VERSION or payload['generation'] != GENERATION:
            raise ValueError('positionfix memory identity mismatch')
        if require_frozen and payload['frozen'] is not True:
            raise ValueError('positionfix memory is not frozen')
        if not isinstance(payload['bootstrap'], dict) or set(payload['bootstrap']) != BOOTSTRAP_KEYS:
            raise ValueError('positionfix bootstrap summary schema mismatch')
        expected = _content_hash(payload)
        if payload['content_hash'] != expected:
            raise ValueError('positionfix memory content hash mismatch')
        raw_rules = payload['rules']
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError('positionfix memory requires at least one rule')
        rules = []
        seen = set()
        for raw in raw_rules:
            if not isinstance(raw, dict) or set(raw) != RULE_KEYS:
                raise ValueError('positionfix rule schema mismatch')
            rule = PositionFixRule(rule_id=str(raw['rule_id']), object_category=normalize_category(raw['object_category']), failure_category=normalize_category(raw['failure_category']), source_type=normalize_category(raw['source_type']), component_selection=str(raw['component_selection']), anchor_gate_frac=float(raw['anchor_gate_frac']), min_component_pixels=int(raw['min_component_pixels']), max_component_area_frac=float(raw['max_component_area_frac']), max_height_above_table_m=float(raw['max_height_above_table_m']), support_failure_rollouts=int(raw['support_failure_rollouts']), support_source_attempts=int(raw['support_source_attempts']), description=str(raw['description']))
            key = (rule.object_category, rule.failure_category, rule.source_type)
            if key in seen:
                raise ValueError(f'duplicate positionfix rule key: {key}')
            seen.add(key)
            if rule.component_selection != 'nearest_current_anchor_boundary':
                raise ValueError('unsupported positionfix component selection')
            if not 0.0 < rule.anchor_gate_frac <= 0.5:
                raise ValueError('positionfix anchor gate outside declared domain')
            if rule.min_component_pixels < 8:
                raise ValueError('positionfix minimum component size too small')
            if not 0.0 < rule.max_component_area_frac <= 0.5:
                raise ValueError('positionfix component area bound outside declared domain')
            if not 0.0 < rule.max_height_above_table_m <= 0.5:
                raise ValueError('positionfix height bound outside declared domain')
            rules.append(rule)
        return cls(version=payload['schema_version'], generation=payload['generation'], content_hash=payload['content_hash'], rules=tuple(rules))

    def select_rule(self, *, object_category: str, failure_category: str, source_type: str) -> PositionFixRule | None:
        key = (normalize_category(object_category), normalize_category(failure_category), normalize_category(source_type))
        for rule in self.rules:
            if (rule.object_category, rule.failure_category, rule.source_type) == key:
                return rule
        return None

def load_positionfix_memory(path: str | Path) -> PositionFixMemory:
    """The only production deserialization entry point for this memory."""
    return PositionFixMemory.load(path, require_frozen=True)
