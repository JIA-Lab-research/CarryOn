from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from typing import Optional
from ..endpoints import EndpointRegistry
from ..memory import Memory, SceneMemory, UserMemory
from ..types import Plan
DEFAULT_SYSTEM_PROMPT = 'You are CarryOn, a high-level robot dispatch agent.\n\nYou will be given:\n  (a) a user instruction, AND/OR\n  (b) one or more "events" describing prior subtask outcomes -- failures,\n      timeouts, or asynchronous status updates pushed by an endpoint\n      (for example: an oven reporting that its indicator turned green,\n      a camera reporting that a target is now visible, etc.).\n\nWhen the configured planner provider is a VLM, you may also receive current\ncamera/image inputs out-of-band in the model request. Use those visual inputs\nas scene context when choosing endpoints, ordering subtasks, and writing\ninstructions.\n\nEach event may or may not require action. Your job is to decide, given the\noverall goal so far, what (if anything) CarryOn should dispatch NEXT.\n\nYou only know an endpoint by its natural-language description and optional\nskill_library. The VLA behind each endpoint is a black box -- do NOT invent\nstructured action parameters or motor commands.\n\n================================================================\nOUTPUT FORMAT  (this is REQUIRED -- the parser depends on it)\n================================================================\nYou MUST first reason out loud, THEN emit a single JSON document.\n\n1. Wrap your reasoning in a <reasoning>...</reasoning> block. In it, walk\n   through:\n     - what the user / event is asking for,\n     - which endpoints look relevant and why,\n     - the order/dependencies you intend,\n     - any uncertainty or fallback you want to flag.\n   This block is for your own thinking; CarryOn logs it but does not\n   parse it.\n\n2. After </reasoning>, emit ONE JSON object matching the schema below.\n   No markdown fences, no extra prose after it.\n\nJSON schema:\n{\n  "goal": "<string -- short restatement of what THIS plan tries to achieve>",\n  "subtasks": [\n    {\n      "id": "<string, e.g. t1>",\n      "endpoint": "<endpoint name from the provided list>",\n      "instruction": "<free-form natural language>",\n      "inputs": { },          // optional extra context, object\n      "depends_on": ["<id>"]  // optional, defaults to []\n    }\n  ]\n}\n\nRules:\n- Use ONLY endpoint names from the provided list.\n- Each subtask.id must be unique; depends_on must reference ids from THIS plan.\n- The DAG must be acyclic.\n- Do NOT include fields other than id/endpoint/instruction/inputs/depends_on.\n- If an endpoint has a skill_library, treat it as the prompt family this model\n  was trained/deployed to support. Write instruction by choosing the closest\n  skill prompt and making only the smallest needed natural-language edits for\n  objects, places, quantities, or placeholders. Do NOT ask an endpoint to do\n  something clearly outside its skill_library.\n- For VLA / learned-policy endpoints, each subtask must be ONE execution of ONE\n  skill_library entry. Do NOT batch repeated objects into plurals such as\n  "two objects" when the closest skill describes one object. Instead, emit\n  repeated subtasks at the skill\'s natural execution granularity, chained in\n  the intended order.\n- NEVER substitute the target object. The instruction\'s object noun must be\n  copied verbatim into the subtask instruction even if it is not listed in the\n  endpoint\'s skill_library, looks absent from the camera view, or seems harder\n  than an alternative. Do NOT plan a different "fallback" object (e.g. do not\n  rewrite "pick up the chocolate pudding" into "pick up the cream cheese as a\n  fallback"): executing the wrong object is a task failure that downstream\n  judges/recovery cannot undo, whereas a not-found object simply fails fast and\n  gets corrected.\n- If the requested sequence contains multiple destinations or phases, decompose\n  it into separate skill executions. Prefer duplicating exact skill_library\n  prompts over inventing new source/destination wording that the endpoint was\n  not trained on.\n- COVER EVERY PHASE the user explicitly mentioned, end to end. If the user\n  chains phases with words like "then", "after that", "and then", "next",\n  emit subtasks for ALL of them in the initial Plan; never stop after only\n  the first phase. The plan\'s "goal" string MUST also restate the full\n  end-to-end objective, not just the first phase.\n- If `web_query` is available and the user asks for current, real-time,\n  external, internet-backed, or weather information, dispatch `web_query`\n  instead of a robot endpoint. Preserve the user\'s FULL question in the\n  `web_query` instruction and inputs.query so the skill can answer the actual\n  question, not only fetch raw facts. For weather, set inputs.kind="weather"\n  and inputs.location when the user explicitly named a place. For general\n  lookups, set inputs.kind="search".\n- If the instruction references results of earlier subtasks IN THIS PLAN, you\n  may use the placeholder ${t1.result.<path>} inside instruction or inputs\n  (it will be substituted at dispatch time).\n- If you decide that NO new dispatch is needed (the event is informational, the\n  goal is already done, or you want to wait for more events), emit a Plan with\n  an EMPTY "subtasks" array, e.g.:\n      {"goal": "no action needed; waiting for next event", "subtasks": []}\n'

class ContextBuilder:

    def __init__(self, memory: Memory, registry: EndpointRegistry, system_prompt: Optional[str]=None, *, now_fn=None, scene_memory: Optional[SceneMemory]=None, scene_top_k: int=20, user_memory: Optional[UserMemory]=None, locations: Optional[dict]=None) -> None:
        self.memory = memory
        self.registry = registry
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._now_fn = now_fn or time.time
        self._task_start_ts: Optional[float] = None
        self.scene_memory = scene_memory
        self.scene_top_k = scene_top_k
        self.user_memory = user_memory
        self.locations = dict(locations or {})

    def mark_task_start(self) -> None:
        """Record the wall-clock time at which the current top-level task
        began. AgentLoop calls this in ``handle()`` so the Planner sees a
        stable ``TASK START TIME`` across replans within the same user turn.
        """
        self._task_start_ts = self._now_fn()

    def _time_header(self) -> str:
        now_ts = self._now_fn()
        lines = [f'CURRENT TIME: {_fmt_iso(now_ts)}']
        if self._task_start_ts is not None:
            lines.append(f'TASK START TIME: {_fmt_iso(self._task_start_ts)}')
            elapsed = max(0.0, now_ts - self._task_start_ts)
            lines.append(f'ELAPSED SINCE TASK START: {elapsed:.1f}s')
        lines.append("Use these timestamps to reason about real-world delays (e.g. 'boil 3 minutes, then turn off heat'). When a delay is required between two subtasks, insert a `wait` subtask with inputs.duration_s and chain via depends_on.")
        return '\n'.join(lines)

    def system_text(self) -> str:
        endpoints = self.registry.snapshot_for_planner_vlm()
        parts: list[str] = [self.system_prompt, self._time_header(), self._user_memory_block(), self._scene_memory_block(), self._locations_block(), 'Available endpoints:\n' + json.dumps(endpoints, ensure_ascii=False, indent=2)]
        return '\n\n'.join((p for p in parts if p))

    def _user_memory_block(self) -> str:
        if self.user_memory is None:
            return ''
        rendered = self.user_memory.render()
        if not rendered.strip():
            return ''
        return 'USER MEMORY (durable per-user facts, as markdown; one `# user: <id>`\nheading per user, then `## <category>` sections):\n' + rendered + "\nCategories are namespaced (all user-centric):\n- profile:* durable facts about the user: identity, habits, and\n            preferences (home city, diet, how they like tasks done,\n            e.g. profile:cooking / boiled_egg_count). INFER task\n            behavior from them: a vegetarian user's sandwich gets no\n            meat; use profile:home to fill a `web_query` location.\n- entity:*  attributes of objects a user owns/refers to; use them to\n            resolve owner references, e.g. rewrite 'A 的杯子' using\n            entity:cup / color for user A. An `entity:*` `reference_image`\n            value is a file path you can pass as `reference_image_path`\n            into `search_locations`.\nA fresh value in the new user message always wins over stored memory\n(a one-off wish like '今天不想吃肉' overrides the profile for this\ntask only)."

    def _scene_memory_block(self) -> str:
        if self.scene_memory is None:
            return ''
        rendered = self.scene_memory.snapshot_for_planner(self.scene_top_k)
        if not rendered or not rendered.strip():
            return 'SCENE MEMORY: (empty)\nIf the task needs an object whose location you do not know, pick\ncandidate waypoints from KNOWN LOCATIONS and dispatch a\n`search_locations` subtask to scan them; then persist what you saw\nwith `update_scene`.'
        return 'SCENE MEMORY (what was last observed at each location, as markdown):\n' + rendered + '\nUse these records to route the robot directly to a known location\nwhen it holds the target object. After any `search_locations` run,\ndispatch `update_scene` (per location) to persist the new observations\n(include coords for shelves, a description for every location).'

    def _locations_block(self) -> str:
        if not self.locations:
            return ''
        from ..locations import render_locations_markdown
        rendered = render_locations_markdown(self.locations)
        if not rendered.strip():
            return ''
        return 'KNOWN LOCATIONS (pre-mapped waypoints the mobile base can reach):\n' + rendered + '\nThese are WHERE the robot can go (the map); SCENE MEMORY is WHAT is\nthere. To scan, build `search_locations.inputs.locations` from these\n(name + reach_endpoint + reach_inputs={x, y, yaw}); YOU decide which\nwaypoints to visit and in what order.'

    def build_for_planning(self, user_task: str) -> tuple[str, list[dict]]:
        """Return (system_text, messages)."""
        msgs: list[dict] = []
        recent_results = [m for m in self.memory.messages() if m.role == 'tool_result'][-10:]
        if recent_results:
            summary = '\n'.join((f'- {m.plan_id}:{m.subtask_id} @ {m.endpoint}: {m.content[:200]}' for m in recent_results))
            msgs.append({'role': 'user', 'content': f'Prior results in this session:\n{summary}'})
            msgs.append({'role': 'assistant', 'content': 'Acknowledged.'})
        msgs.append({'role': 'user', 'content': user_task})
        return (self.system_text(), msgs)

    def build_for_reaction(self, plan: Plan, events: list[dict]) -> tuple[str, list[dict]]:
        """Build messages asking the planner VLM what to dispatch NEXT given events.

        ``events`` is a list of dicts produced by the Reactor, each describing
        one trigger -- a failed/timeout subtask or an asynchronous endpoint
        status update. The planner VLM may respond with an empty-subtasks Plan if no
        action is required.
        """
        sub_lines = []
        for s in plan.subtasks:
            tail = ''
            if s.result is not None:
                if s.result.ok:
                    data = s.result.data
                    brief = data.get('summary') if isinstance(data, dict) and data.get('summary') else str(data)[:120] if data is not None else ''
                    tail = f' -> success: {brief}'
                else:
                    tail = f" -> {s.status}: {s.result.error or ''}"
            elif s.status != 'pending':
                tail = f' -> {s.status}'
            sub_lines.append(f'  - {s.id} @ {s.endpoint} (status={s.status}, attempts={s.attempts}): {s.instruction[:160]}{tail}')
        plan_block = f'Current Plan {plan.id}:\n  goal: {plan.goal}\n  subtasks:\n' + '\n'.join(sub_lines)
        evt_lines = []
        image_parts: list[dict] = []
        for i, ev in enumerate(events, 1):
            kind = ev.get('kind') or 'event'
            ep = ev.get('endpoint') or '?'
            sid = ev.get('subtask_id') or '-'
            detail = ev.get('detail') or ''
            evt_lines.append(f'  [{i}] kind={kind} subtask={sid} endpoint={ep}: {detail}')
            image_parts.extend(_event_image_parts(i, ev))
        events_block = 'Events to react to:\n' + '\n'.join(evt_lines)
        prompt = f"""{plan_block}\n\n{events_block}\n\nDecide what CarryOn should dispatch NEXT.\n- If the events indicate a failure that you want to retry / route to a\n  different endpoint, emit subtasks describing that.\n- If 'before' / 'after' camera images are attached, compare them to\n  diagnose what went wrong and consider an error-recovery subtask.\n- If the events are merely informational (e.g. an oven LED turned green)\n  and trigger a follow-up action, emit that follow-up.\n- If no action is needed (informational only, or original goal already\n  achieved), emit a Plan with an EMPTY "subtasks" array.\n- New subtask ids should be globally unique within this session; prefix\n  them with 'r<n>-' (n = how many reaction rounds so far) to avoid\n  collision with the original plan's ids.\n- depends_on may reference other NEW subtasks in this same response, but\n  NOT subtasks from the original plan."""
        if image_parts:
            content: list[dict] = [{'type': 'text', 'text': prompt}] + image_parts
            return (self.system_text(), [{'role': 'user', 'content': content}])
        return (self.system_text(), [{'role': 'user', 'content': prompt}])

    def build_for_summary(self, plan: Plan, results_text: str) -> tuple[str, list[dict]]:
        return _build_for_summary(plan, results_text)

def _event_image_parts(event_index: int, event: dict) -> list[dict]:
    images = event.get('images')
    if not isinstance(images, dict):
        return []
    parts: list[dict] = []
    for key in ('before', 'after'):
        for snap in images.get(key) or []:
            if not isinstance(snap, dict):
                continue
            b64 = snap.get('image_b64')
            if not b64:
                continue
            mime = snap.get('mime') or 'image/jpeg'
            cam = snap.get('camera') or 'camera'
            parts.append({'type': 'text', 'text': f'event {event_index} {key} ({cam}):'})
            parts.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}})
    return parts

def _fmt_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec='seconds')

def _build_for_summary(plan: Plan, results_text: str) -> tuple[str, list[dict]]:
    sys = 'You are summarizing the result of a completed robot dispatch plan. Be concise, factual, and mention any failures.'
    msg = f'Goal: {plan.goal}\nSubtask outcomes:\n{results_text}\n\nWrite a 1-3 sentence summary for the user.'
    return (sys, [{'role': 'user', 'content': msg}])
