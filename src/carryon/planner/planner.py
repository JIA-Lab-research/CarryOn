from __future__ import annotations

import json
from typing import Optional

from ..agent.context import (
    ContextBuilder,
    validate_outdoor_packing_weather_gate,
    validate_scene_update_requirement,
)
from ..endpoints import EndpointRegistry
from ..errors import PlanError
from ..logging_util import EventLogger
from .client import PlannerVLMClient
from ..types import Plan

from .schema import (
    extract_reasoning,
    parse_and_validate_plan,
    parse_and_validate_reaction_plan,
)


class Planner:
    def __init__(
        self,
        planner_vlm: PlannerVLMClient,
        context_builder: ContextBuilder,
        registry: EndpointRegistry,
        max_retries: int = 2,
        logger: Optional[EventLogger] = None,
    ) -> None:
        self.planner_vlm = planner_vlm
        self.cb = context_builder
        self.registry = registry
        self.max_retries = max_retries
        self.logger = logger

    async def plan(self, user_task: str, current_plan_id: str | None = None,
                   correction: str | None = None,
                   progress: str | None = None,
                   require_scene_update: str | None = None) -> Plan:
        """Plan this turn. ``current_plan_id`` chains a CONTINUATION round: the
        named plan's own execution results stay in context so the planner knows
        what it already tried, while other plans' results stay excluded


        ``progress`` is the turn's executed-step ledger. It goes AFTER the
        messages the builder produced, because the last of those re-poses the
        user's original instruction and on its own reads as "begin this task".

        ``require_scene_update`` is the reveal text of an observation nothing
        has written down (agent loop's _outstanding_record). When set, a plan
        without an `update_scene` subtask is refused at parse time and the
        refusal is fed back through the repair retry -- the write stays the
        planner's, but whether it happens stops being its memory.
        """
        system, messages = self.cb.build_for_planning(user_task, current_plan_id)
        if progress:
            messages = messages + [{"role": "user", "content": progress}]
        if correction:
            # In-context correction from the continuation loop (e.g. "these
            # steps already executed"); same channel the repair retry uses.
            messages = messages + [{"role": "user", "content": correction}]
        return await self._run_plan_loop(
            system=system,
            messages=messages,
            parser=parse_and_validate_plan,
            user_task=user_task,
            enforce_outdoor_weather_gate=True,
            require_scene_update=require_scene_update,
        )

    async def plan_after_weather(self, user_task: str, weather_plan: Plan) -> Plan:
        """Phase-2 outdoor packing: choose arm subtasks from a weather result."""
        system, messages = self.cb.build_for_packing_after_weather(
            user_task, weather_plan
        )
        return await self._run_plan_loop(
            system=system,
            messages=messages,
            parser=parse_and_validate_plan,
            user_task=user_task,
            enforce_outdoor_weather_gate=False,
        )

    async def react(self, plan: Plan, events: list[dict]) -> Plan:
        """Ask the planner VLM what to dispatch next given plan-state + new events.

        Returned Plan may have zero subtasks, meaning "no action needed".
        """
        system, messages = self.cb.build_for_reaction(plan, events)
        return await self._run_plan_loop(
            system=system,
            messages=messages,
            parser=parse_and_validate_reaction_plan,
            user_task=plan.original_user_request,
            enforce_outdoor_weather_gate=False,
        )

    async def summarize_results(self, plan: Plan) -> str:
        """Use the planner VLM to synthesize a user-facing answer from results.

        This is intentionally not a Plan parse path: after an information-gathering
        endpoint such as ``web_query`` returns raw facts/search results, the model
        should answer the user's actual question in natural language.
        """
        system = (
            "You are CarryOn answering the user's original request after tool "
            "execution. Use ONLY the provided plan and subtask results. Do not "
            "invent facts. If the results are insufficient, say what is missing. "
            "Return concise natural language only; do not return JSON or markdown."
        )
        payload = {
            "goal": plan.goal,
            "subtasks": [
                {
                    "id": s.id,
                    "endpoint": s.endpoint,
                    "instruction": s.instruction,
                    "inputs": s.inputs,
                    "status": s.status,
                    "result": (
                        {
                            "ok": s.result.ok,
                            "data": s.result.data,
                            "error": s.result.error,
                        }
                        if s.result is not None
                        else None
                    ),
                }
                for s in plan.subtasks
            ],
        }
        messages = [
            {
                "role": "user",
                "content": (
                    "Answer the user's request based on these completed "
                    "subtask results:\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            }
        ]
        resp = await self._chat(
            system=system,
            messages=messages,
            response_schema_hint=False,
        )
        answer = _strip_reasoning_text(resp.text).strip()
        if _looks_like_plan_json(answer):
            resp = await self._chat(
                system=system,
                messages=messages + [
                    {"role": "assistant", "content": resp.text},
                    {
                        "role": "user",
                        "content": (
                            "You returned a Plan JSON object. Do not plan more "
                            "subtasks. Answer the user's original question in "
                            "plain natural language based on the provided results."
                        ),
                    },
                ],
                response_schema_hint=False,
            )
            answer = _strip_reasoning_text(resp.text).strip()
        return answer

    async def _run_plan_loop(
        self,
        *,
        system,
        messages,
        parser,
        user_task: str = "",
        enforce_outdoor_weather_gate: bool = False,
        require_scene_update: str | None = None,
    ):
        last_err: Exception | None = None
        attempts = self.max_retries + 1
        for i in range(attempts):
            resp = await self._chat(system=system, messages=messages)
            try:
                plan = parser(resp.text, self.registry)
                if enforce_outdoor_weather_gate:
                    validate_outdoor_packing_weather_gate(
                        plan,
                        user_task=user_task,
                        web_query_available="web_query" in self.registry.names(),
                    )
                if require_scene_update:
                    validate_scene_update_requirement(
                        plan, reveal=require_scene_update)
                # Runtime provenance, not Planner VLM output. This survives
                # initial plans, continuation rounds, weather phase 2, and
                # reactions so the Judge can see the user's original wording.
                plan.original_user_request = user_task
                return plan
            except PlanError as e:
                last_err = e
                # repair-style retry: feed error back
                messages = messages + [
                    {"role": "assistant", "content": resp.text},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed as a "
                            f"valid Plan. Error: {e}\n"
                            f"Please reply with a <reasoning> block followed by "
                            f"STRICT JSON matching the schema."
                        ),
                    },
                ]
        raise PlanError(f"planner failed after {attempts} attempts: {last_err}")

    async def _chat(
        self,
        *,
        system: str,
        messages: list[dict],
        response_schema_hint: bool = True,
    ):
        """Thin wrapper around planner_vlm.plan_chat that also traces I/O.

        We additionally extract the model's <reasoning> block (if any) into
        the trace fields so it shows up in the planner log without polluting the
        Plan parser.
        """
        if self.logger is None:
            return await self.planner_vlm.plan_chat(
                system=system,
                messages=messages,
                response_schema_hint=response_schema_hint,
            )

        pending = self.logger.planner_request(system=system, messages=messages)
        try:
            resp = await self.planner_vlm.plan_chat(
                system=system,
                messages=messages,
                response_schema_hint=response_schema_hint,
            )
        except BaseException as e:
            self.logger.planner_error(pending, e)
            raise
        # best-effort: surface reasoning separately for readers
        reasoning = extract_reasoning(resp.text)
        usage = dict(resp.usage or {})
        if reasoning is not None:
            usage = {**usage, "reasoning_chars": len(reasoning)}
        self.logger.planner_response(pending, text=resp.text, usage=usage)
        return resp


def _strip_reasoning_text(text: str) -> str:
    import re

    text = re.sub(
        r"<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>",
        "",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _looks_like_plan_json(text: str) -> bool:
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(obj, dict) and "goal" in obj and "subtasks" in obj


