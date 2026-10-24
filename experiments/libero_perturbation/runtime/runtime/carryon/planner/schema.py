from __future__ import annotations
import json
import re
from typing import Any
from ..endpoints import EndpointRegistry
from ..errors import PlanError
from ..types import Plan, Subtask
PLAN_SCHEMA: dict[str, Any] = {'type': 'object', 'required': ['goal', 'subtasks'], 'additionalProperties': False, 'properties': {'goal': {'type': 'string', 'minLength': 1}, 'subtasks': {'type': 'array', 'minItems': 0, 'items': {'type': 'object', 'required': ['id', 'endpoint', 'instruction'], 'additionalProperties': False, 'properties': {'id': {'type': 'string', 'pattern': '^[A-Za-z0-9_-]+$'}, 'endpoint': {'type': 'string', 'minLength': 1}, 'instruction': {'type': 'string', 'minLength': 1}, 'inputs': {'type': 'object'}, 'depends_on': {'type': 'array', 'items': {'type': 'string'}}}}}}}
_FENCE_RE = re.compile('^```(?:json)?\\s*|\\s*```$', re.MULTILINE)
_REASONING_RE = re.compile('<\\s*reasoning\\s*>.*?<\\s*/\\s*reasoning\\s*>', re.IGNORECASE | re.DOTALL)

def extract_reasoning(text: str) -> str | None:
    """Return the first <reasoning>...</reasoning> body, or None if absent.

    Useful for logging/debugging the model's chain-of-thought without it
    leaking into the parsed Plan.
    """
    m = _REASONING_RE.search(text or '')
    if not m:
        return None
    body = m.group(0)
    body = re.sub('^<\\s*reasoning\\s*>', '', body, flags=re.IGNORECASE)
    body = re.sub('<\\s*/\\s*reasoning\\s*>$', '', body, flags=re.IGNORECASE)
    return body.strip()

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        text = _FENCE_RE.sub('', text).strip()
    return text

def _strip_reasoning(text: str) -> str:
    """Remove all <reasoning>...</reasoning> blocks from text."""
    return _REASONING_RE.sub('', text or '')

def _extract_json(text: str) -> str:
    text = _strip_reasoning(text)
    text = _strip_fences(text)
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end < 0 or end <= start:
        raise PlanError('no JSON object found in planner VLM output')
    return text[start:end + 1]

def _validate_against_schema(obj: Any) -> None:
    try:
        import jsonschema
    except ImportError as e:
        raise PlanError('jsonschema package is required') from e
    try:
        jsonschema.validate(obj, PLAN_SCHEMA)
    except jsonschema.ValidationError as e:
        raise PlanError(f'schema: {e.message} at {list(e.path)}') from e

def _check_graph(subtasks: list[dict], registry: EndpointRegistry) -> None:
    ids = [s['id'] for s in subtasks]
    if len(set(ids)) != len(ids):
        raise PlanError('duplicate subtask ids')
    id_set = set(ids)
    known = set(registry.names())
    for s in subtasks:
        if s['endpoint'] not in known:
            raise PlanError(f"subtask {s['id']}: endpoint '{s['endpoint']}' is not registered. Known: {sorted(known)}")
        for d in s.get('depends_on', []) or []:
            if d not in id_set:
                raise PlanError(f"subtask {s['id']}: depends_on unknown id '{d}'")
    in_deg = {sid: 0 for sid in ids}
    edges: dict[str, list[str]] = {sid: [] for sid in ids}
    for s in subtasks:
        for d in s.get('depends_on', []) or []:
            edges[d].append(s['id'])
            in_deg[s['id']] += 1
    queue = [sid for sid, deg in in_deg.items() if deg == 0]
    visited = 0
    while queue:
        sid = queue.pop()
        visited += 1
        for nxt in edges[sid]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    if visited != len(ids):
        raise PlanError('plan DAG has a cycle')

def parse_and_validate_plan(text: str, registry: EndpointRegistry) -> Plan:
    """Parse an INITIAL plan from planner VLM output. Requires at least one subtask."""
    plan = _parse_plan_common(text, registry)
    if not plan.subtasks:
        raise PlanError('plan has no subtasks (initial planning requires >=1)')
    return plan

def parse_and_validate_reaction_plan(text: str, registry: EndpointRegistry) -> Plan:
    """Parse a REACTION plan from planner VLM output.

    Empty subtasks is allowed and means "no new dispatch needed; keep waiting".
    """
    return _parse_plan_common(text, registry)

def _parse_plan_common(text: str, registry: EndpointRegistry) -> Plan:
    raw = _extract_json(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlanError(f'invalid json: {e}') from e
    _validate_against_schema(obj)
    _check_graph(obj['subtasks'], registry)
    return Plan(id=Plan.new_id(), goal=obj['goal'], subtasks=[Subtask(id=s['id'], endpoint=s['endpoint'], instruction=s['instruction'], inputs=s.get('inputs') or {}, depends_on=list(s.get('depends_on') or [])) for s in obj['subtasks']])
