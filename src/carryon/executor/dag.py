from __future__ import annotations

import json
import re
from typing import Any

from ..errors import UnresolvedPlaceholder
from ..types import Plan, Subtask

# matches ${t1.result.pose.x} or ${t1.result}
_PLACEHOLDER_RE = re.compile(r"\$\{\s*([A-Za-z0-9_-]+)\.result(\.[^}]+)?\s*\}")
# a string that is EXACTLY one placeholder (nothing around it)
_EXACT_PLACEHOLDER_RE = re.compile(
    r"^\$\{\s*([A-Za-z0-9_-]+)\.result(\.[^}]+)?\s*\}$"
)


def topo_layers(plan: Plan) -> list[list[Subtask]]:
    """Return subtasks grouped into DAG layers (topological order)."""
    by_id = {s.id: s for s in plan.subtasks}
    remaining = set(by_id.keys())
    layers: list[list[Subtask]] = []
    done: set[str] = set()
    while remaining:
        layer = [
            by_id[sid]
            for sid in remaining
            if all(d in done for d in by_id[sid].depends_on)
        ]
        if not layer:
            raise ValueError("cycle detected in plan DAG")
        # stable order by id for determinism
        layer.sort(key=lambda s: s.id)
        layers.append(layer)
        for s in layer:
            remaining.remove(s.id)
            done.add(s.id)
    return layers


def _get_path(obj: Any, path: str) -> Any:
    # path like ".pose.x" -> tokens ["pose", "x"]. Planner VLMs naturally
    # write list indices in bracket form (".observations[0].description");
    # normalize brackets to dots so both spellings resolve identically.
    path = path.replace("[", ".").replace("]", "")
    tokens = [t for t in path.strip(".").split(".") if t]
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


def _resolve_match(m: re.Match, plan: Plan) -> Any:
    sid = m.group(1)
    path = m.group(2) or ""
    sub = plan.get(sid)
    if sub is None or sub.status != "success" or sub.result is None:
        raise UnresolvedPlaceholder(
            f"cannot resolve ${{{sid}.result{path}}}: subtask not finished"
        )
    data = sub.result.data
    return _get_path(data, path) if path else data


def resolve_placeholders(value: Any, plan: Plan) -> Any:
    """Substitute ${tID.result.path} in strings using finished subtasks in this plan.

    A string that is EXACTLY one placeholder resolves to the referenced value
    with its native type (list/dict/number/...), so structured results (e.g.
    search observations) can be passed whole into a downstream subtask's
    inputs. Placeholders embedded in surrounding text are substituted as
    JSON-stringified text, as before.
    """
    if isinstance(value, str):
        exact = _EXACT_PLACEHOLDER_RE.match(value)
        if exact:
            return _resolve_match(exact, plan)

        return _PLACEHOLDER_RE.sub(
            lambda m: _stringify(_resolve_match(m, plan)), value
        )
    if isinstance(value, list):
        return [resolve_placeholders(v, plan) for v in value]
    if isinstance(value, dict):
        return {k: resolve_placeholders(v, plan) for k, v in value.items()}
    return value


def is_single_slot_endpoint(ep) -> bool:
    """A physical endpoint that runs ONE job at a time. Generic capability, not
    keyed on any endpoint name: an ``arm`` tag, or explicit ``single_slot``
    metadata in ``extra``."""
    try:
        if "arm" in (getattr(ep, "tags", None) or []):
            return True
        return bool((getattr(ep, "extra", None) or {}).get("single_slot"))
    except Exception:  # noqa: BLE001
        return False


def arm_subtasks_total_order_ok(plan, registry) -> bool:
    """True iff, for every single-slot endpoint used in the plan, the subtasks
    targeting it form a TOTAL order in the dependency graph -- each later one
    transitively depends on the previous. A same-arm fan-out (t2<-t1, t3<-t1) or
    bare same-arm roots are False: they would run concurrently on a single-slot
    arm. Endpoint single-slot-ness comes from the registry capability, never
    from task strings."""
    subs = list(getattr(plan, "subtasks", []) or [])
    ids = {s.id for s in subs}
    deps = {s.id: [x for x in (s.depends_on or []) if x in ids] for s in subs}

    anc: dict = {}

    def ancestors(sid):
        if sid in anc:
            return anc[sid]
        anc[sid] = set()  # guard against cycles
        acc: set = set()
        for x in deps.get(sid, ()):  # noqa: E741
            acc.add(x)
            acc |= ancestors(x)
        anc[sid] = acc
        return acc

    by_ep: dict = {}
    for s in subs:
        try:
            ep = registry.get(s.endpoint)
        except Exception:  # noqa: BLE001 -- placeholder endpoints etc.
            continue
        if is_single_slot_endpoint(ep):
            by_ep.setdefault(s.endpoint, []).append(s.id)

    for sids in by_ep.values():
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = sids[i], sids[j]
                if b not in ancestors(a) and a not in ancestors(b):
                    return False  # two concurrent same-arm subtasks
    return True
