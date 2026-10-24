from __future__ import annotations
import json
from typing import Optional
from ..agent.context import ContextBuilder
from ..endpoints import EndpointRegistry
from ..errors import PlanError
from ..logging_util import EventLogger
from .client import PlannerVLMClient
from ..types import Plan
from .schema import extract_reasoning, parse_and_validate_plan, parse_and_validate_reaction_plan

class Planner:

    def __init__(self, planner_vlm: PlannerVLMClient, context_builder: ContextBuilder, registry: EndpointRegistry, max_retries: int=2, logger: Optional[EventLogger]=None) -> None:
        self.planner_vlm = planner_vlm
        self.cb = context_builder
        self.registry = registry
        self.max_retries = max_retries
        self.logger = logger

    async def plan(self, user_task: str) -> Plan:
        system, messages = self.cb.build_for_planning(user_task)
        return await self._run_plan_loop(system=system, messages=messages, parser=parse_and_validate_plan)

    async def react(self, plan: Plan, events: list[dict]) -> Plan:
        """Ask the planner VLM what to dispatch next given plan-state + new events.

        Returned Plan may have zero subtasks, meaning "no action needed".
        """
        system, messages = self.cb.build_for_reaction(plan, events)
        return await self._run_plan_loop(system=system, messages=messages, parser=parse_and_validate_reaction_plan)

    async def summarize_results(self, plan: Plan) -> str:
        """Use the planner VLM to synthesize a user-facing answer from results.

        This is intentionally not a Plan parse path: after an information-gathering
        endpoint such as ``web_query`` returns raw facts/search results, the model
        should answer the user's actual question in natural language.
        """
        system = "You are CarryOn answering the user's original request after tool execution. Use ONLY the provided plan and subtask results. Do not invent facts. If the results are insufficient, say what is missing. Return concise natural language only; do not return JSON or markdown."
        payload = {'goal': plan.goal, 'subtasks': [{'id': s.id, 'endpoint': s.endpoint, 'instruction': s.instruction, 'inputs': s.inputs, 'status': s.status, 'result': {'ok': s.result.ok, 'data': s.result.data, 'error': s.result.error} if s.result is not None else None} for s in plan.subtasks]}
        messages = [{'role': 'user', 'content': "Answer the user's request based on these completed subtask results:\n" + json.dumps(payload, ensure_ascii=False, indent=2)}]
        resp = await self._chat(system=system, messages=messages, response_schema_hint=False)
        answer = _strip_reasoning_text(resp.text).strip()
        if _looks_like_plan_json(answer):
            resp = await self._chat(system=system, messages=messages + [{'role': 'assistant', 'content': resp.text}, {'role': 'user', 'content': "You returned a Plan JSON object. Do not plan more subtasks. Answer the user's original question in plain natural language based on the provided results."}], response_schema_hint=False)
            answer = _strip_reasoning_text(resp.text).strip()
        return answer

    async def _run_plan_loop(self, *, system, messages, parser):
        last_err: Exception | None = None
        attempts = self.max_retries + 1
        for i in range(attempts):
            resp = await self._chat(system=system, messages=messages)
            try:
                return parser(resp.text, self.registry)
            except PlanError as e:
                last_err = e
                messages = messages + [{'role': 'assistant', 'content': resp.text}, {'role': 'user', 'content': f'Your previous response could not be parsed as a valid Plan. Error: {e}\nPlease reply with a <reasoning> block followed by STRICT JSON matching the schema.'}]
        raise PlanError(f'planner failed after {attempts} attempts: {last_err}')

    async def _chat(self, *, system: str, messages: list[dict], response_schema_hint: bool=True):
        """Thin wrapper around planner_vlm.plan_chat that also traces I/O.

        We additionally extract the model's <reasoning> block (if any) into
        the trace fields so it shows up in the planner log without polluting the
        Plan parser.
        """
        if self.logger is None:
            return await self.planner_vlm.plan_chat(system=system, messages=messages, response_schema_hint=response_schema_hint)
        pending = self.logger.planner_request(system=system, messages=messages)
        try:
            resp = await self.planner_vlm.plan_chat(system=system, messages=messages, response_schema_hint=response_schema_hint)
        except BaseException as e:
            self.logger.planner_error(pending, e)
            raise
        reasoning = extract_reasoning(resp.text)
        usage = dict(resp.usage or {})
        if reasoning is not None:
            usage = {**usage, 'reasoning_chars': len(reasoning)}
        self.logger.planner_response(pending, text=resp.text, usage=usage)
        return resp

def _strip_reasoning_text(text: str) -> str:
    import re
    text = re.sub('<\\s*reasoning\\s*>.*?<\\s*/\\s*reasoning\\s*>', '', text or '', flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

def _looks_like_plan_json(text: str) -> bool:
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(obj, dict) and 'goal' in obj and ('subtasks' in obj)
