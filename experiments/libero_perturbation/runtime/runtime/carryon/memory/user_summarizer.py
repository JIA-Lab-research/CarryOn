"""Distill durable user-profile facts from a completed plan's conversation.

Called by ``Executor.maybe_finish_plan`` whenever a plan reaches a terminal
state. Pulls every ShortTermMemory message tagged with ``plan_id`` (plus the
user-level messages around it) and asks the Planner VLM to distill any
durable facts about the user.

This is the **single distillation writer** for
:class:`carryon.memory.user.UserMemory`: the user never issues memory
commands; the profile grows automatically from ordinary interactions.
(Reference images arrive separately via the ``/remember_image`` upload
channel.) Facts land in the two user-centric namespaces ``profile:*``
(identity, habits, preferences) and ``entity:*`` (owned-object attributes).

A **durability filter** is applied at extraction time: only facts with
durable evidence -- explicitly stated lasting preferences/attributes, or
patterns clearly meant to persist -- are written. One-off, session-scoped
wishes ("今天不想吃肉") and volatile data (weather, time) never enter the
profile.

The VLM is asked to return STRICT JSON shaped like::

    {"facts": [
        {"user_id": "A", "category": "entity:cup", "key": "color",
         "value": "yellow", "confidence": 0.9},
        {"category": "profile:eating_habit", "key": "diet", "value": "素食"}
    ]}

Empty or malformed responses are silently dropped — profile distillation is
best-effort.
"""
from __future__ import annotations
import json
from typing import Any, Optional, Protocol
from ..planner.client import PlannerVLMResponse
from ..types import Message, Plan
from .user import NAMESPACES, UserFact, UserMemory, _new_fact_id, normalize_category, normalize_user_id
_SYSTEM_PROMPT = 'You are distilling durable USER PROFILE facts from one finished robot task.\nRead the conversation excerpts and return STRICT JSON only (no markdown\nfences). Output shape:\n\n{"facts": [\n    {"user_id": "<who the fact is about; omit for the default speaker "\n                "(first-person 我/my)>",\n     "category": "<one of: profile:<aspect>, entity:<object>>",\n     "key": "<short snake_case attribute, e.g. diet, city, egg_count, color>",\n     "value": <JSON value: number, string, or array>,\n     "confidence": <number in [0, 1]>}\n]}\n\nCategory namespaces (user-centric):\n- profile:* durable facts about the user themselves: identity, habits, and\n            preferences. e.g. profile:home / city = 北京;\n            profile:eating_habit / diet = 素食;\n            profile:cooking / boiled_egg_count = 2.\n- entity:*  an attribute of an object the user owns/refers to, e.g.\n            entity:cup / color = yellow.\n\nDURABILITY FILTER (the most important rule):\n- Only write facts that will still be true in future sessions. Evidence must\n  be a lasting preference/attribute explicitly stated ("我是素食主义者",\n  "我的杯子是绿色的", "我煮蛋一般煮3分钟") or clearly meant as a standing\n  instruction ("以后煮蛋都煮2个").\n- Do NOT write one-off, session-scoped wishes ("我今天不想吃肉", "这次少放点\n  盐") — they apply to this task only.\n- Do NOT write volatile data (current weather, time, robot/task state).\n- Generalize durable one-off phrasing into reusable form: "帮我煮3个鸡蛋，\n  煮7分钟，以后都这样" -> profile:cooking / egg_count=3, duration_min=7.\n- Do NOT fabricate. If unsure whether a fact is durable, leave it out; an\n  empty facts list is acceptable and common.\n'

class _PlannerVLMLike(Protocol):

    async def plan_chat(self, messages: list[dict], *, system: Optional[str]=None, temperature: float=0.2, max_tokens: int=4096) -> PlannerVLMResponse:
        ...

class UserSummarizer:

    def __init__(self, *, planner_vlm: _PlannerVLMLike, user_memory: UserMemory, user_id: str='default', min_messages: int=3, enabled: bool=True) -> None:
        self.planner_vlm = planner_vlm
        self.user_memory = user_memory
        self.user_id = normalize_user_id(user_id)
        self.min_messages = max(0, int(min_messages))
        self.enabled = bool(enabled)

    async def maybe_summarize(self, plan: Plan, messages: list[Message]) -> list[UserFact]:
        if not self.enabled:
            return []
        relevant = [m for m in messages if m.plan_id == plan.id or m.role in ('user', 'assistant')]
        if len(relevant) < self.min_messages:
            return []
        excerpt = self._format_excerpt(plan, relevant)
        try:
            try:
                resp = await self.planner_vlm.plan_chat(messages=[{'role': 'user', 'content': excerpt}], system=_SYSTEM_PROMPT, temperature=0.0, max_tokens=512, response_schema_hint=False)
            except TypeError:
                resp = await self.planner_vlm.plan_chat(messages=[{'role': 'user', 'content': excerpt}], system=_SYSTEM_PROMPT, temperature=0.0, max_tokens=512)
        except Exception:
            return []
        facts = _parse_response(resp.text, source_plan_id=plan.id, default_user_id=self.user_id)
        if not facts:
            return []
        self.user_memory.add_many(facts)
        return facts

    @staticmethod
    def _format_excerpt(plan: Plan, messages: list[Message]) -> str:
        lines = [f'PLAN GOAL: {plan.goal}', '']
        for m in messages[-40:]:
            tag = m.role
            if m.subtask_id:
                tag = f'{tag}:{m.subtask_id}'
            lines.append(f'[{tag}] {m.content[:400]}')
        return '\n'.join(lines)

def _parse_response(text: str, *, source_plan_id: str, default_user_id: str) -> list[UserFact]:
    text = (text or '').strip()
    if not text:
        return []
    if text.startswith('<reasoning>') and '</reasoning>' in text:
        text = text.split('</reasoning>', 1)[1].strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except Exception:
        return []
    items = payload.get('facts') if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    out: list[UserFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        category = normalize_category(item.get('category'))
        key = str(item.get('key') or '').strip()
        if not key or not category or category.split(':', 1)[0] not in NAMESPACES:
            continue
        value: Any = item.get('value')
        conf = _coerce_float(item.get('confidence')) or 0.0
        out.append(UserFact(id=_new_fact_id(), user_id=normalize_user_id(item.get('user_id') or default_user_id), category=category, key=key, value=value, source=source_plan_id, confidence=max(0.0, min(1.0, conf))))
    return out

def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
