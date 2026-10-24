from __future__ import annotations
import json
import re
from typing import Any
from ..errors import UnresolvedPlaceholder
from ..types import Plan, Subtask
_PLACEHOLDER_RE = re.compile('\\$\\{\\s*([A-Za-z0-9_-]+)\\.result(\\.[^}]+)?\\s*\\}')

def topo_layers(plan: Plan) -> list[list[Subtask]]:
    """Return subtasks grouped into DAG layers (topological order)."""
    by_id = {s.id: s for s in plan.subtasks}
    remaining = set(by_id.keys())
    layers: list[list[Subtask]] = []
    done: set[str] = set()
    while remaining:
        layer = [by_id[sid] for sid in remaining if all((d in done for d in by_id[sid].depends_on))]
        if not layer:
            raise ValueError('cycle detected in plan DAG')
        layer.sort(key=lambda s: s.id)
        layers.append(layer)
        for s in layer:
            remaining.remove(s.id)
            done.add(s.id)
    return layers

def _get_path(obj: Any, path: str) -> Any:
    tokens = [t for t in path.strip('.').split('.') if t]
    cur = obj
    for t in tokens:
        if isinstance(cur, dict):
            cur = cur.get(t)
        elif isinstance(cur, list):
            try:
                cur = cur[int(t)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur

def _stringify(v: Any) -> str:
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def resolve_placeholders(value: Any, plan: Plan) -> Any:
    """Substitute ${tID.result.path} in strings using finished subtasks in this plan."""
    if isinstance(value, str):

        def repl(m: re.Match) -> str:
            sid = m.group(1)
            path = m.group(2) or ''
            sub = plan.get(sid)
            if sub is None or sub.status != 'success' or sub.result is None:
                raise UnresolvedPlaceholder(f'cannot resolve ${{{sid}.result{path}}}: subtask not finished')
            data = sub.result.data
            resolved = _get_path(data, path) if path else data
            return _stringify(resolved)
        return _PLACEHOLDER_RE.sub(repl, value)
    if isinstance(value, list):
        return [resolve_placeholders(v, plan) for v in value]
    if isinstance(value, dict):
        return {k: resolve_placeholders(v, plan) for k, v in value.items()}
    return value
