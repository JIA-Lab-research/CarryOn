from __future__ import annotations

import json
import re
from typing import Any

from ..endpoints import EndpointRegistry
from ..errors import PlanError
from ..types import Plan, Subtask

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["goal", "subtasks"],
    "additionalProperties": False,
    "properties": {
        "goal": {"type": "string", "minLength": 1},
        # Brief message shown to the user at plan-create time. Optional for
        # backward compatibility with scripted/mock plans; the system prompt
        # asks the live planner to always set it.
        "user_reply": {"type": "string"},
        "subtasks": {
            "type": "array",
            # NOTE: minItems=0 -- in REACTION mode the planner VLM may legitimately
            # decide that no new dispatch is needed (informational event,
            # goal already achieved). Initial-planning mode requires either
            # >=1 subtask OR a non-empty user_reply (chat / preference ack)
            # via parse_and_validate_plan() below.
            "minItems": 0,
            "items": {
                "type": "object",
                "required": ["id", "endpoint", "instruction"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]+$"},
                    "endpoint": {"type": "string", "minLength": 1},
                    "instruction": {"type": "string", "minLength": 1},
                    # Judge-only completion criterion (see Subtask.objective).
                    # Optional: omitted/empty means the Judge falls back to the
                    # instruction, so older/scripted plans stay valid.
                    "objective": {"type": "string"},
                    # Why this step is in the plan (see Subtask.purpose). An
                    # enum, so a third value is a parse error rather than a
                    # label the boundary gate below silently ignores.
                    "purpose": {"enum": ["observe", "act"]},
                    "inputs": {"type": "object"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
# <reasoning> ... </reasoning>  (case-insensitive, dot-all so newlines match)
_REASONING_RE = re.compile(
    r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>",
    re.IGNORECASE | re.DOTALL,
)


def extract_reasoning(text: str) -> str | None:
    """Return the first <reasoning>...</reasoning> body, or None if absent.

    Useful for logging/debugging the model's chain-of-thought without it
    leaking into the parsed Plan.
    """
    m = _REASONING_RE.search(text or "")
    if not m:
        return None
    body = m.group(0)
    # strip outer tags
    body = re.sub(r"^<\s*reasoning\s*>", "", body, flags=re.IGNORECASE)
    body = re.sub(r"<\s*/\s*reasoning\s*>$", "", body, flags=re.IGNORECASE)
    return body.strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def _strip_reasoning(text: str) -> str:
    """Remove all <reasoning>...</reasoning> blocks from text."""
    return _REASONING_RE.sub("", text or "")


def _extract_json(text: str) -> str:
    # Drop the chain-of-thought block(s) FIRST so we don't accidentally find a
    # `{` that lives inside the reasoning prose.
    text = _strip_reasoning(text)
    text = _strip_fences(text)
    # find first '{' and matching '}' (best-effort)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise PlanError("no JSON object found in planner VLM output")
    return text[start : end + 1]


def _validate_against_schema(obj: Any) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError as e:
        raise PlanError("jsonschema package is required") from e
    try:
        jsonschema.validate(obj, PLAN_SCHEMA)  # type: ignore[arg-type]
    except jsonschema.ValidationError as e:  # type: ignore[attr-defined]
        raise PlanError(f"schema: {e.message} at {list(e.path)}") from e


def _check_graph(subtasks: list[dict], registry: EndpointRegistry) -> None:
    ids = [s["id"] for s in subtasks]
    if len(set(ids)) != len(ids):
        raise PlanError("duplicate subtask ids")
    id_set = set(ids)
    known = set(registry.names())
    for s in subtasks:
        if "${" in s["endpoint"]:
            # Endpoint discovered at runtime via a ${tN.result...} placeholder
            # (e.g. dispatch to an endpoint an earlier subtask resolved).
            # Name validation is deferred to dispatch time, where the Executor
            # resolves it and fails the subtask if the result is unregistered.
            # It must at least depend on the subtask it references.
            if not (s.get("depends_on") or []):
                raise PlanError(
                    f"subtask {s['id']}: placeholder endpoint "
                    f"'{s['endpoint']}' requires depends_on referencing the "
                    "subtask that produces it"
                )
        elif s["endpoint"] not in known:
            raise PlanError(
                f"subtask {s['id']}: endpoint '{s['endpoint']}' is not registered. "
                f"Known: {sorted(known)}"
            )
        else:
            # Library conformance first: when both are wrong, a minted
            # instruction is the more fundamental error and its message
            # carries the legal entry list.
            _validate_skill_library(s, registry)
            _validate_purpose_declared(s, registry)
        for d in s.get("depends_on", []) or []:
            if d not in id_set:
                raise PlanError(f"subtask {s['id']}: depends_on unknown id '{d}'")
    # acyclicity via topological pass
    in_deg = {sid: 0 for sid in ids}
    edges: dict[str, list[str]] = {sid: [] for sid in ids}
    for s in subtasks:
        for d in s.get("depends_on", []) or []:
            edges[d].append(s["id"])
            in_deg[s["id"]] += 1
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
        raise PlanError("plan DAG has a cycle")
    # Acyclic by here, which is what lets the downstream walk terminate.
    _validate_observation_boundary(subtasks, registry)


def parse_and_validate_plan(text: str, registry: EndpointRegistry) -> Plan:
    """Parse an INITIAL plan from planner VLM output.

    Requires at least one subtask, OR a non-empty ``user_reply`` for
    chat/preference turns that need no robot dispatch.
    """
    plan = _parse_plan_common(text, registry)
    if not plan.subtasks and not (plan.user_reply or "").strip():
        raise PlanError(
            "plan has no subtasks (initial planning requires >=1 subtask, "
            "or a non-empty user_reply for chat-only)"
        )
    return plan


def parse_and_validate_reaction_plan(
    text: str, registry: EndpointRegistry
) -> Plan:
    """Parse a REACTION plan from planner VLM output.

    Empty subtasks is allowed and means "no new dispatch needed; keep waiting".
    """
    return _parse_plan_common(text, registry)


def _parse_plan_common(text: str, registry: EndpointRegistry) -> Plan:
    raw = _extract_json(text)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PlanError(f"invalid json: {e}") from e
    _validate_against_schema(obj)
    _check_graph(obj["subtasks"], registry)
    return Plan(
        id=Plan.new_id(),
        goal=obj["goal"],
        user_reply=str(obj.get("user_reply") or "").strip(),
        subtasks=[
            Subtask(
                id=s["id"],
                endpoint=s["endpoint"],
                instruction=s["instruction"],
                objective=str(s.get("objective") or "").strip(),
                purpose=str(s.get("purpose") or "").strip(),
                inputs=s.get("inputs") or {},
                depends_on=list(s.get("depends_on") or []),
            )
            for s in obj["subtasks"]
        ],
    )


def _endpoint_of(s: dict, registry):
    """The registered endpoint for a subtask dict, or None if unregistered."""
    ep = registry.get(s["endpoint"]) if hasattr(registry, "get") else None
    if ep is None:
        for cand in registry.list():
            if getattr(cand, "name", None) == s["endpoint"]:
                ep = cand
                break
    return ep


def _has_skill_library(s: dict, registry) -> bool:
    """True for endpoints that declare a skill_library -- the physical ones.

    Both per-subtask gates use this as their seam, so mock / scripted plans
    over free-text endpoints keep parsing exactly as they did before.
    """
    lib = getattr(_endpoint_of(s, registry), "skill_library", None)
    return bool(list(lib or []))


def _library_pattern(entry: str):
    """A library entry as a matcher: fixed text, with declared {placeholders}
    consuming arbitrary non-empty text."""
    import re
    parts = re.split(r"\{[^{}]+\}", entry)
    return re.compile("^" + "(.+)".join(re.escape(p) for p in parts) + "$")


def _validate_skill_library(s: dict, registry) -> None:
    """When an endpoint declares a skill_library, the instruction MUST be one
    of its entries (with declared {placeholders} filled). Third strike on the
    same axis made this a gate: the prompt said "copy verbatim" twice, in two
    voices, and the planner still minted "pick up white lid ..." after reading
    a yellow lid as white. Prose asks; the parser refuses. A violation lands
    in PlanError, which the planner retry loop feeds back verbatim -- so the
    error message lists the legal entries, which IS the object-category list
    the planner should have inferred from in the first place."""
    library = list(getattr(_endpoint_of(s, registry), "skill_library", None) or [])
    if not library:
        return                              # free-text endpoint, nothing to hold
    instr = str(s.get("instruction") or "")
    for entry in library:
        if instr == entry or ("{" in entry and _library_pattern(entry).match(instr)):
            return
    raise PlanError(
        f"subtask {s['id']}: instruction {instr!r} is not in endpoint "
        f"'{s['endpoint']}' skill_library. Instructions must be copied VERBATIM "
        f"from one of its entries (only {{placeholders}} may be filled); the "
        f"camera never renames objects -- a colour that reads oddly is still "
        f"the library object. Allowed entries: {library}"
    )


def _validate_purpose_declared(s: dict, registry) -> None:
    """A physical step must say WHY it is in the plan.

    Only skill_library endpoints are held to it, on the same seam
    _validate_skill_library uses. The label is what the boundary gate below
    reads, so an unlabelled physical step would slip that gate in silence --
    which is the whole failure this pair exists to stop.
    """
    if not _has_skill_library(s, registry):
        return
    if str(s.get("purpose") or "").strip():
        return
    raise PlanError(
        f"subtask {s['id']}: missing \"purpose\". Every subtask on an endpoint "
        f"with a skill_library must declare why it is in the plan: \"observe\" "
        f"if the point of the step is to find something out and what you do "
        f"next depends on what it shows, or \"act\" if the need for the step is "
        f"already established."
    )


def _validate_observation_boundary(subtasks: list[dict], registry) -> None:
    """Require at most one observation, with no subtasks depending on it.

    Only skill-library endpoints count as observers. A subsequent planning round
    uses the fresh frame produced by the observation."""
    observers = [s["id"] for s in subtasks
                 if str(s.get("purpose") or "").strip() == "observe"
                 and _has_skill_library(s, registry)]
    if not observers:
        return
    if len(observers) > 1:
        raise PlanError(
            f"plan declares {len(observers)} subtasks with purpose=observe "
            f"({', '.join(observers)}); at most ONE is allowed. Emit the first "
            f"alone -- you are asked again with a fresh camera frame when it "
            f"finishes, and what it shows decides whether the rest are needed "
            f"at all."
        )
    obs = observers[0]
    dependents: dict[str, list[str]] = {}
    for s in subtasks:
        for d in s.get("depends_on", []) or []:
            dependents.setdefault(d, []).append(s["id"])
    downstream: set[str] = set()
    frontier = list(dependents.get(obs, []))
    while frontier:
        sid = frontier.pop()
        if sid in downstream:
            continue
        downstream.add(sid)
        frontier.extend(dependents.get(sid, []))
    if downstream:
        raise PlanError(
            f"subtask(s) {', '.join(sorted(downstream))} depend on {obs}, which "
            f"has purpose=observe. Work whose necessity depends on an "
            f"observation you have not made yet cannot be planned now -- not "
            f"even chained with depends_on. Emit {obs} alone; decide the rest "
            f"next round, from the frame that actually answers it."
        )
