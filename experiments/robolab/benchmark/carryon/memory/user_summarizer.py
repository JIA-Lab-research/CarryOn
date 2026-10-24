"""Distill durable user-profile facts from a completed plan's conversation.

Called by ``Executor.maybe_finish_plan`` whenever a plan reaches a terminal
state. Pulls every ShortTermMemory message tagged with ``plan_id`` (plus the
user-level messages around it) and asks the Planner VLM to distill any
durable facts about the user.

Facts land in the two user-centric namespaces ``profile:*`` (identity, habits,
preferences) and ``entity:*`` (owned-object attributes). The planner may also
have written the same facts mid-plan via ``update_user_memory``; this
summarizer is a backup distillation path. Before writing it:

1. injects the on-disk profile markdown into the VLM excerpt so the model can
   avoid re-emitting known facts, and
2. deterministically drops candidates already covered by the store (exact
   ``(category, key)`` or an equivalent ``profile:avoid`` nesting alias).

A **durability filter** is applied at extraction time: only facts with
durable evidence -- explicitly stated lasting preferences/attributes, or
patterns clearly meant to persist -- are written. One-off, session-scoped
wishes ("今天不想吃肉") and volatile data (weather, time) never enter the
profile.

The VLM is asked to return STRICT JSON shaped like::

    {"facts": [
        {"user_id": "A", "category": "entity:cup", "key": "color",
         "value": "yellow", "confidence": 0.9},
        {"category": "profile:eating_habit", "key": "diet",
         "value": "素食；不吃肉"}
    ]}

Empty or malformed responses are silently dropped — profile distillation is
best-effort.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Protocol

from ..planner.client import PlannerVLMResponse
from ..types import Message, Plan
from .user import (
    AVOID_CATEGORY_PREFIX,
    NAMESPACES,
    UserFact,
    UserMemory,
    _new_fact_id,
    is_avoid_category,
    normalize_category,
    normalize_user_id,
)


_SYSTEM_PROMPT = """\
You are distilling durable USER PROFILE facts from one finished robot task.
Read the conversation excerpts and the EXISTING USER MEMORY block. Return
STRICT JSON only (no markdown fences). Output shape:

{"facts": [
    {"user_id": "<who the fact is about; OMIT the field entirely for the "
                "default speaker (first-person my/me) -- never write "
                "placeholder ids like 'me' or 'user'>",
     "category": "<one of: profile:<aspect>, entity:<object>>",
     "key": "<short snake_case attribute, e.g. diet, city, egg_count, color>",
     "value": <JSON value: number, string, or array>,
     "confidence": <number in [0, 1]>,
     "delete": <optional true -- retract this category/key instead of writing>}
]}

Category namespaces (user-centric):
- profile:* durable facts about the user: identity, habits, preferences.
  e.g. profile:home / city = "Beijing";
  profile:eating_habit / diet = "vegetarian; does not eat meat";
  profile:cooking / boiled_egg_count = 2.
- entity:* owned-object attributes, e.g. entity:cup / color = yellow.
- profile:avoid:* standing prohibitions. Category "profile:avoid:<topic>",
  short snake_case key, plain-language value. Distil from durable negations
  ("never use Z", "don't put my cup in the dishwasher"). Never flatten to
  bare "profile:avoid" with a dotted key (e.g. prefer profile:avoid:running
  / hat, not profile:avoid / running.hat).

VALUE QUALITY:
- Short plain phrases a future planner can act on; prefer the user's wording.
- Do NOT invent coarse enum-like / snake_case labels that drop constraints
  (e.g. do not reduce several dietary constraints to "meat_included").
- Positive habit + standing prohibition -> separate profile:* and
  profile:avoid:* facts, not one vague merged label.

EXISTING MEMORY (do not duplicate):
- The excerpt includes EXISTING USER MEMORY already on disk (often written
  earlier this turn by update_user_memory). Do NOT re-emit a fact that is
  already present with the same meaning, even under a differently nested
  category/key. An empty facts list is correct when nothing new remains.
- Only emit NEW durable facts, SUPERSEDING updates (different value), or
  "delete": true retractions for obsolete keys.

DURABILITY FILTER (the most important rule):
- Only write facts that will still be true in future sessions. Evidence must
  be a lasting preference/attribute explicitly stated, or clearly meant as a
  standing instruction ("always cook 2 eggs from now on").
- Do NOT write one-off / session-scoped wishes ("I don't want meat today",
  "no salt this time") or volatile data (weather, time, robot/task state).
- Generalize durable phrasing into reusable form: "cook 3 eggs for 7 minutes
  from now on" -> profile:cooking / egg_count=3, duration_min=7.
- Do NOT fabricate. If unsure, leave it out; an empty facts list is fine.
- Negatives use the same filter: profile:avoid:* only for a STANDING prohibition,
  never one-off negatives.

CORRECTIONS & VETOES (strong preference signal):
- User corrections / rejections of a prior plan are STRONGER durable evidence
  than passive statements -- prioritise extracting profile:* / entity:* /
  profile:avoid:* from those turns. This does NOT relax the durability filter
  and does NOT change how passive statements are distilled.
- When a new durable preference SUPERSEDES an older one, emit the new write(s)
  AND retract obsolete keys with "delete": true (same category/key that no
  longer applies). Overwrite alone is not enough when the old fact lived under
  a different key (e.g. diet flips to vegetarian -> also delete an obsolete
  avoid:food/greens from a prior meat-lover phase).

IMAGE ATTACHMENTS:
- User messages may be labeled "(attached image imgN: <filename>)".
- For an entity:* fact grounded in such a message, add
  "reference_image": "imgN" on THAT fact. Only when the image depicts the
  entity itself; never on profile:* or invented ids.
"""


class _PlannerVLMLike(Protocol):
    async def plan_chat(
        self, messages: list[dict], *, system: Optional[str] = None,
        temperature: float = 0.2, max_tokens: int = 4096,
    ) -> PlannerVLMResponse: ...


class UserSummarizer:
    def __init__(
        self,
        *,
        planner_vlm: _PlannerVLMLike,
        user_memory: UserMemory,
        user_id: str = "default",
        min_messages: int = 3,
        enabled: bool = True,
    ) -> None:
        self.planner_vlm = planner_vlm
        self.user_memory = user_memory
        self.user_id = normalize_user_id(user_id)
        self.min_messages = max(0, int(min_messages))
        self.enabled = bool(enabled)

    async def maybe_summarize(self, plan: Plan, messages: list[Message]) -> list[UserFact]:
        if not self.enabled:
            return []
        relevant = [m for m in messages if (m.plan_id == plan.id) or m.role in ("user", "assistant")]
        # Need at least a small amount of conversation context.
        if len(relevant) < self.min_messages:
            return []
        # Snapshot AFTER mid-plan update_user_memory writes so the VLM and the
        # deterministic filter both see what is already on disk.
        existing = self.user_memory.snapshot_for_planner()
        excerpt, attachments = self._format_excerpt(
            plan, relevant, existing_memory=self.user_memory.render(),
        )
        try:
            # response_schema_hint=False: plan_chat otherwise appends the
            # PLANNER response schema to the system prompt, which overrides
            # the fact-JSON contract here (same bug family as the reflector /
            # catalog generator) -- facts then parse to [] and are silently
            # dropped by the except/empty guards below.
            # max_tokens must fit the WHOLE facts JSON: the excerpt spans
            # the session's recent user messages, so on the Nth plan of a
            # session the facts list covers every entity mentioned so far.
            # 512 hard-truncated the JSON mid-array on the 4th seeding
            # statement (completion_tokens == 512, parse -> 0 facts,
            # silently dropped); 2048 gives ~4x headroom.
            try:
                resp = await self.planner_vlm.plan_chat(
                    messages=[{"role": "user", "content": excerpt}],
                    system=_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=2048,
                    response_schema_hint=False,
                )
            except TypeError:  # client predates the flag
                resp = await self.planner_vlm.plan_chat(
                    messages=[{"role": "user", "content": excerpt}],
                    system=_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=2048,
                )
        except Exception:
            return []
        facts, deletions, image_refs = _parse_response(
            resp.text,
            source_plan_id=plan.id,
            default_user_id=self.user_id,
            attachments=attachments,
        )
        # Automatic image memory (ADR 0008 addendum): an entity fact grounded
        # in a user message that carried an image attachment also gets that
        # image stored as the entity's reference_image -- same on-disk path
        # as the /remember_image upload channel, no command needed.
        for uid, category, src_path in dict.fromkeys(image_refs):
            try:
                stored = self.user_memory.store_reference_image(uid, src_path)
            except (FileNotFoundError, OSError):
                continue
            facts.append(UserFact(
                id=_new_fact_id(), user_id=uid, category=category,
                key="reference_image", value=stored, source=plan.id,
            ))
        # Drop / rebase candidates already covered by the store (incl. avoid
        # nesting aliases). Deletions always proceed.
        facts = _filter_novel_facts(facts, existing)
        if facts:
            self.user_memory.add_many(facts)
        if deletions:
            self.user_memory.remove_many(deletions)
        return facts

    @staticmethod
    def _format_excerpt(
        plan: Plan,
        messages: list[Message],
        *,
        existing_memory: str = "",
    ) -> tuple[str, dict[str, str]]:
        """Render the conversation excerpt; label image attachments.

        Returns ``(excerpt, attachments)`` where attachments maps the stable
        inline ids (``img1``, ``img2``, ...) shown to the VLM back to the
        attachment file paths from ``Message.metadata["attachments"]``.
        """
        lines = [f"PLAN GOAL: {plan.goal}", ""]
        mem = (existing_memory or "").strip()
        lines.append(
            "EXISTING USER MEMORY (already on disk; do NOT re-emit facts "
            "already present with the same meaning):"
        )
        lines.append(mem if mem else "(empty)")
        lines.append("")
        lines.append("CONVERSATION:")
        attachments: dict[str, str] = {}
        for m in messages[-40:]:
            tag = m.role
            if m.subtask_id:
                tag = f"{tag}:{m.subtask_id}"
            note = ""
            for att in (m.metadata or {}).get("attachments") or []:
                if not isinstance(att, dict):
                    continue
                path = att.get("path")
                if att.get("type") != "image" or not path:
                    continue
                img_id = f"img{len(attachments) + 1}"
                attachments[img_id] = str(path)
                note += f" (attached image {img_id}: {os.path.basename(str(path))})"
            lines.append(f"[{tag}]{note} {m.content[:400]}")
        return "\n".join(lines), attachments


# VLM outputs that mean "the speaker" rather than naming a third party. The
# session user id must absorb these: writing them verbatim fragments one
# person's profile across two files (e.g. robolab.md + me.md), and the
# planner then only ever sees half of it.
_FIRST_PERSON_ALIASES = {
    "me", "myself", "i", "self", "user", "the user", "default speaker",
    "speaker", "default", "我", "本人", "用户",
}


def _resolve_user_id(raw: Any, default_user_id: str) -> str:
    uid = str(raw or "").strip()
    if not uid or uid.lower() in _FIRST_PERSON_ALIASES:
        return normalize_user_id(default_user_id)
    return normalize_user_id(uid)


def _normalize_fact_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _values_equivalent(a: Any, b: Any) -> bool:
    return _normalize_fact_value(a) == _normalize_fact_value(b)


def _avoid_leaf_path(category: str, key: str) -> Optional[str]:
    """Collapse avoid nesting aliases to one leaf path.

    ``profile:avoid:running`` / ``hat`` and ``profile:avoid`` / ``running.hat``
    both become ``running.hat``. Non-avoid categories return None.
    """
    cat = normalize_category(category)
    k = str(key or "").strip()
    if not k or not is_avoid_category(cat):
        return None
    if cat == AVOID_CATEGORY_PREFIX:
        return k
    topic = cat[len(AVOID_CATEGORY_PREFIX) + 1 :]  # after "profile:avoid:"
    return f"{topic}.{k}" if topic else k


def _filter_novel_facts(
    facts: list[UserFact],
    existing: dict[str, dict[str, dict[str, Any]]],
) -> list[UserFact]:
    """Keep only facts that add or supersede something in ``existing``.

    - Exact ``(category, key)`` with the same value -> drop (already stored,
      often by mid-plan ``update_user_memory``).
    - Exact key with a different value -> keep (overwrite).
    - ``profile:avoid`` nesting alias of an existing leaf with the same value
      -> drop (the Alice hat redundancy case).
    - Avoid alias with a different value -> rebase onto the existing
      ``(category, key)`` so we do not create a second section.
    """
    out: list[UserFact] = []
    # Track leaves we accept in this batch so two alias shapes in one VLM
    # response do not both land.
    seen_avoid: dict[str, set[str]] = {}
    for fact in facts:
        uid = normalize_user_id(fact.user_id)
        cat = normalize_category(fact.category)
        key = str(fact.key).strip()
        if not cat or not key:
            continue
        snap = existing.get(uid) or {}
        bucket = snap.get(cat) or {}

        if key in bucket:
            if _values_equivalent(bucket[key], fact.value):
                continue
            out.append(fact)
            leaf = _avoid_leaf_path(cat, key)
            if leaf is not None:
                seen_avoid.setdefault(uid, set()).add(leaf)
            continue

        leaf = _avoid_leaf_path(cat, key)
        if leaf is not None:
            if leaf in seen_avoid.get(uid, set()):
                continue
            rebased = False
            for ec, ekv in snap.items():
                for ek, ev in ekv.items():
                    if _avoid_leaf_path(ec, ek) != leaf:
                        continue
                    if _values_equivalent(ev, fact.value):
                        rebased = True  # drop duplicate alias
                        break
                    # Supersede in place under the key that already exists.
                    out.append(UserFact(
                        id=fact.id or _new_fact_id(),
                        user_id=uid,
                        category=ec,
                        key=ek,
                        value=fact.value,
                        source=fact.source,
                        confidence=fact.confidence,
                    ))
                    seen_avoid.setdefault(uid, set()).add(leaf)
                    rebased = True
                    break
                if rebased:
                    break
            if rebased:
                continue
            seen_avoid.setdefault(uid, set()).add(leaf)

        out.append(UserFact(
            id=fact.id or _new_fact_id(),
            user_id=uid,
            category=cat,
            key=key,
            value=fact.value,
            source=fact.source,
            confidence=fact.confidence,
        ))
    return out


def _parse_response(
    text: str,
    *,
    source_plan_id: str,
    default_user_id: str,
    attachments: Optional[dict[str, str]] = None,
) -> tuple[list[UserFact], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Parse the VLM's fact JSON.

    Returns ``(facts, deletions, image_refs)``. Each deletion is
    ``(user_id, category, key)``. Each image ref is ``(user_id, category,
    attachment_path)`` for an ``entity:*`` fact whose ``reference_image``
    field named one of the excerpt's attachment ids (or its path).
    """
    attachments = attachments or {}
    image_refs: list[tuple[str, str, str]] = []
    deletions: list[tuple[str, str, str]] = []
    text = (text or "").strip()
    if not text:
        return [], deletions, image_refs
    # Tolerate <reasoning>...</reasoning> prefix the planner VLM may emit.
    if text.startswith("<reasoning>") and "</reasoning>" in text:
        text = text.split("</reasoning>", 1)[1].strip()
    if text.startswith("```"):
        # strip ```json fences if present
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except Exception:
        return [], deletions, image_refs
    items = payload.get("facts") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return [], deletions, image_refs
    out: list[UserFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = normalize_category(item.get("category"))
        key = str(item.get("key") or "").strip()
        if not key or not category or category.split(":", 1)[0] not in NAMESPACES:
            continue
        uid = _resolve_user_id(item.get("user_id"), default_user_id)
        if item.get("delete") is True or (
            "value" in item and item.get("value") is None
        ):
            deletions.append((uid, category, key))
            continue
        value: Any = item.get("value")
        conf = _coerce_float(item.get("confidence")) or 0.0
        # The VLM sometimes emits reference_image as a REGULAR fact
        # ({"key": "reference_image", "value": "imgN"}) instead of the
        # documented fact-level field; written verbatim that leaves a bogus
        # "imgN" value in the profile. Route it through the same attachment
        # resolution as the field form, and fail closed on unknown ids.
        if key == "reference_image" and category.startswith("entity:"):
            ref_val = str(value or "").strip()
            path = attachments.get(ref_val) or (
                ref_val if ref_val in attachments.values() else None
            )
            if path:
                image_refs.append((uid, category, path))
            continue
        out.append(UserFact(
            id=_new_fact_id(),
            user_id=uid,
            category=category,
            key=key,
            value=value,
            source=source_plan_id,
            confidence=max(0.0, min(1.0, conf)),
        ))
        # Image-grounded entity fact: reference_image names an attachment id
        # from the excerpt (accept its raw path too). Fail closed on unknown
        # ids and on non-entity categories.
        ref = str(item.get("reference_image") or "").strip()
        if ref and category.startswith("entity:"):
            path = attachments.get(ref) or (
                ref if ref in attachments.values() else None
            )
            if path:
                image_refs.append((uid, category, path))
    return out, deletions, image_refs


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
