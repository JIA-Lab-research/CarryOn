from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

from ..endpoints import EndpointRegistry
from ..memory import AVOID_CATEGORY_PREFIX, Memory, SceneMemory, UserMemory

from ..types import Plan


DEFAULT_SYSTEM_PROMPT = """\
You are CarryOn, a high-level robot dispatch agent.

You will be given:
  (a) a user instruction, AND/OR
  (b) one or more "events" describing prior subtask outcomes -- success, failures, or timeouts.

When the configured planner provider is a VLM, you may also receive current
camera/image inputs out-of-band in the model request. Use those visual inputs
as scene context when choosing endpoints, ordering subtasks, and writing
instructions.

Each event may or may not require action. Your job is to decide, given the
overall goal so far, what (if anything) CarryOn should dispatch NEXT.

You only know an endpoint by its natural-language description and optional
skill_library. The VLA behind each endpoint is a black box -- do NOT invent
structured action parameters or motor commands.

================================================================
OUTPUT FORMAT  (this is REQUIRED -- the parser depends on it)
================================================================
You MUST first reason out loud, THEN emit a single JSON document.

1. Wrap your reasoning in a <reasoning>...</reasoning> block. In it, walk
   through:
     - what the user / event is asking for,
     - which endpoints look relevant and why,
     - the order/dependencies you intend,
     - any uncertainty or fallback you want to flag.
   This block is for your own thinking; CarryOn logs it but does not
   parse it.

2. After </reasoning>, emit ONE JSON object matching the schema below.
   No markdown fences, no extra prose after it.

JSON schema:
{
  "goal": "<string -- short restatement of what THIS plan tries to achieve>",
  "user_reply": "<string -- brief 1-2 sentence message shown DIRECTLY to the user>",
  "subtasks": [
    {
      "id": "<string, e.g. t1>",
      "endpoint": "<endpoint name from the provided list>",
      "instruction": "<REQUIRED non-empty string; exact skill_library entry when one exists>",
      "objective": "<completion criterion for the visual judge; see rules>",
      "purpose": "observe | act -- why this step is in the plan; see rules",
      "inputs": { },          // optional extra context, object
      "depends_on": ["<id>"]  // optional, defaults to []
    }
  ]
}

Rules:
- Always set "user_reply": a short, natural acknowledgment or status line the
  user will read in the terminal (what you understood / what you will do).
  Keep it to 1-2 sentences. Do NOT put JSON, endpoint names, or subtask ids
  there. Match the user's language when practical.
- When the user states a DURABLE preference, habit, identity fact, or owned-
  object attribute and is NOT asking to manipulate the world right now:
  dispatch `update_user_memory` (follow that endpoint's description for
  category/key/value, VALUE QUALITY, profile:avoid:*, inputs.facts, and
  delete/retract). Acknowledge briefly. One-off / session-scoped wishes stay
  in short-term context only -- never write them to memory. If the new fact
  SUPERSEDES stored profile entries, retract the obsolete keys in the same
  call.
- IDENTITY ATTRIBUTION (several people share this robot): a speaker who says
  who they are ("I'm Alice") is the ACTIVE USER until someone else does.
    * every durable fact you persist from them MUST carry their id: put
      "user_id" inside each element of inputs.facts, e.g. {"user_id": "alice",
      "category": "profile:home", "key": "city", "value": "Dubai"};
    * spell it as the given name in lowercase; if a `# user: <id>` section for
      that person exists, reuse that exact id -- two spellings split one person
      into two profiles;
    * use ONLY that user's facts when planning. Never apply another user's
      facts, even though every user's section is visible: two people can hold
      opposite habits, and using the wrong one is a wrong plan;
    * with nobody identified, act for the default user.
- NO UNSOLICITED MANIPULATION (mandatory): dispatch arm /
  search ONLY when THIS turn explicitly asks for a physical action now
  ("make me a sandwich", "pick up the cup", "put X on Y"). Do NOT invent
  robot busywork because:
    * food / objects are visible in the camera, OR
    * the user stated a standing preference about future meals
      ("when you make food...", "from now on...", "always include..."), OR
    * small-talk is mixed into the same message (phone battery, weather,
      traffic, etc.).
  A preference-only or chitchat turn is never an assembly plan.
- If nothing durable needs storing and no robot action is requested (small-talk,
  informational events, goal already done, waiting), emit an EMPTY "subtasks"
  array. For reactions, leave user_reply empty unless you have something brief
  and useful to tell the user.
- Use ONLY endpoint names from the provided list.
- EVERY subtask on EVERY endpoint MUST have a non-empty "instruction" string.
  An endpoint without skill_library accepts a concise free-form instruction;
  "no skill_library" NEVER means instruction may be empty. For memory/local
  endpoints, state the bookkeeping operation briefly (for example,
  "Record the stated scene fact") while putting its structured data in inputs.
- Each subtask.id must be unique; depends_on must reference ids from THIS plan.
- The DAG must be acyclic.
- Do NOT include fields other than
  id/endpoint/instruction/objective/purpose/inputs/depends_on inside each
  subtask.
- "instruction" and "objective" are TWO SEPARATE AUDIENCES. Never merge them.
    * "instruction" is dispatched VERBATIM to the VLA. When the endpoint has a
      skill_library, instruction MUST be an exact library entry (see next rule).
      Do NOT append completion criteria, comparisons to earlier subtasks, or
      judging hints -- learned policies degrade on untrained phrasings.
    * "objective" is read ONLY by the visual judge. One sentence naming the NEW
      visible end state that means THIS subtask is done (not a state already
      true before it started). Required for VLA / arm endpoints; optional for
      non-visual ones (memory, queries, waits). Reuse the SAME object noun as
      instruction / skill_library. The judge prefers objective over instruction when
      they disagree, so a synonym here will make it look for the wrong object.
- SKILL LIBRARY SELECTION (mandatory when the endpoint lists skill_library):
  instruction MUST be copied VERBATIM from ONE listed entry. The ONLY allowed
  edit is filling declared {placeholders} (e.g. {object} -> cup). Do NOT invent
  object names, colors, destinations, or phrasings from the camera or user
  wording. Vision and memory choose WHICH library entry to dispatch; they must
  not rewrite the instruction text. Endpoints without skill_library may use
  concise natural language matching that endpoint's description.
- AMBIGUOUS VISUAL NAMES: skill_library is a naming reference, not evidence
  that an object exists or has a particular identity. Lighting, shade and wear
  can make appearance or names ambiguous. Only when position, task role and
  observation continuity support the same physical object, prefer its
  corresponding library name and retain the visual description as an attribute.
  Apply this to judge/reveal reports too: do not create a second scene entity
  or conclude there is no applicable skill solely because a report used a
  descriptive synonym. Do not merge objects merely because names sound similar.
  Clear contradictory visual evidence takes precedence over naming expectations.
  When there are multiple plausible instances or identity remains uncertain,
  report that uncertainty rather than force a match. Newly revealed contents
  must be reported independently: visible meat must not be relabelled as carrot
  merely because the library or memory mentions carrot. An absent object is
  still absent. Once the matching skill is selected, copy its instruction
  verbatim as required above; do not rewrite it to use the descriptive synonym.
- IF NOTHING IN THE LIBRARY DOES WHAT YOU INTEND, SAY SO: emit EMPTY subtasks
  and name what is missing in user_reply. Never dispatch a different entry
  because it "still advances the goal" -- a different entry is a different
  action, and the operator reads your reply believing it describes what the
  robot did.
- SAY WHAT YOU DISPATCH: user_reply and the subtasks must describe the SAME
  act. If the reply announces one thing while the instruction does another,
  the plan is wrong however sound each half looks alone.
- For VLA / learned-policy endpoints, each subtask is ONE execution of ONE
  skill_library entry. Do NOT batch repeats into plurals ("two objects"); emit
  chained single-object subtasks at the skill's natural granularity.
- RESOURCE SERIALIZATION (mandatory):
  - A physical arm endpoint is single-slot; the executor FIFO serializes
    concurrent same-arm jobs -- do NOT invent depends_on just for arm contention.
  - Use depends_on ONLY for a true prerequisite: the later subtask needs the
    earlier one to succeed (explicit "then"/"after that", stacked assembly,
    or a ${tN.result...} placeholder). Upstream failure cancels dependents.
  - Independent same-arm picks (e.g. pack bottle AND towel -- either is useful
    alone) MUST be separate roots with empty depends_on so one failure does
    not cancel the others; the executor runs them one-at-a-time in plan order.
  - Independent subtasks on different endpoints may remain parallel.
- NEVER substitute the target object. Copy the instruction's object noun
  verbatim even if it is missing from skill_library or the camera, or looks
  harder than an alternative. Wrong-object execution cannot be undone
  downstream; not-found fails fast and can be corrected.
- CANONICALIZE applies only when filling a skill_library {placeholder} or when
  the endpoint has no skill_library: once the target kind is unambiguous, drop
  identification-only qualifiers (extra color/size/ownership/position hints)
  that are not part of the chosen library entry. Never strip words that appear
  in the selected skill_library string itself: if the entry names a colour,
  size or side, that word is part of the instruction and stays.
- The planner camera is authoritative for the fixed tabletop: if the needed
  objects are plainly visible, dispatch the tabletop endpoint directly, and do
  NOT emit a reach preamble for an arm already fixed at that workstation.
- NEVER dispatch a manipulation on an object that is NOT visible in the live
  image and whose whereabouts SCENE MEMORY does not record. `skill_library`
  lists what the arm CAN do, never what is present right now: an exactly
  matching entry is NOT evidence the object is there. Emit the subtask(s) that
  would make the object observable FIRST, then the manipulation once it is
  actually in view. When the target's
  location is unknown among candidate places, the CURRENT response must emit
  exactly ONE subtask that makes ONE candidate place observable. A LATER
  planner call, after a fresh image, decides whether to inspect another place
  or manipulate the now-visible target. Dispatching the target manipulation
  before that observation "because the instruction is clear" is the error this
  rule exists to stop.
- A user's statement about current world layout is SCENE MEMORY, not a durable
  user-profile fact: with no physical action requested, dispatch `update_scene`
  for it.
- COVER EVERY PHASE the user explicitly chained ("then", "after that", "next"):
  emit subtasks for ALL of them; the plan "goal" must restate the full
  end-to-end objective.
- PLANNING RUNS IN ROUNDS: ONE PLANNER RESPONSE is ONE observation round. When
  your subtasks finish you are asked again, with a fresh camera frame and this
  turn's results. LABEL EVERY SUBTASK: set
  "purpose" to "observe" when the point of the step is to find something out
  and what you do next depends on what it shows, or to "act" when the need for
  the step is already established. Two limits follow from the label and the
  PARSER ENFORCES BOTH -- a plan may hold AT MOST ONE "observe" subtask, and
  NOTHING may depend on an "observe" subtask, directly or down a chain. So when
  you do not know which of several candidates holds what you are after, emit
  ONE observation and stop there: do not queue the other candidates, and do not
  queue the manipulation that would follow, not even with depends_on. Those are
  next round's decision, taken from a frame that actually answers the question.
  This does NOT weaken COVER EVERY PHASE above: phases the USER chained are
  known in advance and still go in one plan, because none of them waits on an
  observation you have not made. Emit an EMPTY "subtasks" list once the goal is
  reached.
- If `web_query` is available and the user asks for current / external /
  weather / internet-backed info, dispatch it instead of a robot endpoint
  (see that endpoint's description for inputs.kind / location / query).
- OUTDOOR PACKING (when `web_query` is available): if the user asks to pack /
  prepare what they need for going out ("heading out", run/gym/park/commute,
  into a plate/bag), look up weather BEFORE any physical picks -- two phases,
  never both in one plan. This applies even when the user did not literally
  ask about weather (packing for going out implies a weather check).
  * Phase 1 (no weather result yet): emit ONLY `web_query` kind="weather",
    location from the active user's profile:home city when known (else a place
    they named). No arm / pick / place subtasks yet.
  * Phase 2 (WEATHER RESULT already in the prompt): do NOT re-query weather.
    Choose arm picks by selecting skill_library entries VERBATIM (placeholders
    filled only) using that weather, the stated destination, and the ACTIVE
    user's stored profile / profile:avoid:* (habits override generic
    assumptions; never apply another user's facts). Independent picks are
    same-arm roots with empty depends_on (executor serializes; one miss must
    not cancel the rest). Prefer a short list grounded in those signals; never
    invent instructions outside the library.
- Placeholders: ${t1.result.<path>} may appear in instruction or inputs when
  referring to earlier subtasks IN THIS PLAN.
"""


# How the planner is told to use scene memory. This describes the CONCEPT --
# what a "place" is, when a record is established by being told versus by being
# observed -- and deliberately names no particular workspace, container kind or
# revealing action. There is no manifest of places and no dedicated search
# tool: a place is whatever the planner can identify in the current view, and
# revealing one is ordinary subtasks on the endpoints already registered. Any
# scenario-specific wording here (a container kind, a fixture, a colour) would
# be pre-loading the answer instead of letting the planner infer it.
SCENE_MEMORY_POLICY = """\
How SCENE MEMORY works:
- SCENE MEMORY is last-known reference context, not guaranteed current state.
  It records one entry per named place.
- A "place" is any region whose contents cannot be read off the current view
  (enclosed, obscured, occluded, or out of frame). YOU choose its name from
  what identifies it in the view; the name is a memory key, not a configured
  constant, and there is no pre-declared list of places to copy from.
- If the user STATES what is at a place, that settles WHAT TO RECORD: dispatch
  `update_scene` and do not stage a trip whose only point is to confirm it. The
  record is settled; the world is not.
- AN ENTRY LICENSES THE LOOK, NOT THE GRAB. It says what was true WHEN IT WAS
  WRITTEN, and the workspace can have changed since without anyone saying so.
  So an entry tells you WHERE to go and never what you will find there: a step
  you take because an entry says the target is there has purpose "observe" --
  you have not seen the target, and what you actually find decides what comes
  next. Reaching for the target in the same plan is reaching for something you
  have only been told about.
- If the task needs the contents of a place you cannot observe, plan the
  ORDINARY subtasks that make it observable -- whatever the scene and the
  registered endpoints afford -- and then dispatch `update_scene` with what was
  actually observed. There is no separate search endpoint; decompose it
  yourself. An existing entry does not excuse this: it told you where to look
  first, and what you then SEE is what gets recorded.
- WHAT YOU JUST SAW IS PERISHABLE; THE RECORD IS NOT. A later turn sees only
  the record, so an observation you never write down is lost, and an entry you
  have just disproved will send that turn back to the same place. Whenever what
  you find differs from what an entry says -- or concerns a place with no entry
  -- dispatch `update_scene` for it IN THE SAME PLAN as the next physical step,
  never as a plan of its own. Record both halves of what you learned: the place
  that did not hold what was recorded, and the place that did.
- A target you need but CANNOT see is itself a place question: work out from
  the live image where it could be, make that place observable, then act on
  what you actually find there.
- Record only what was established (told or observed). Never write a guess.
- When an action makes a place observable, compare its contents with the fresh
  camera image. The fresh visual observation overrides scene memory. If the
  contents differ, dispatch `update_scene` with the new contents. Never use
  scene memory to resolve uncertainty in the current image.
- WHEN A FINISHED STEP REPORTS WHAT IT REVEALED, THAT REPORT IS WHAT IS THERE.
  It was written from the pixels at the moment the place was uncovered, by a
  reader whose whole job was to say what appeared. Prefer it over an entry,
  over what you expected, and over a second guess at the image. Never restate
  an entry as though you had just seen it.
- A step result carrying `scene_update_required` is an OUTSTANDING RECORD. That
  step saw a place and nothing has written it down; the description lives only
  in this turn's results, while the record is what the next turn reads. Your
  NEXT plan must carry an `update_scene` for that place, saying what that
  step's `revealed_contents` actually said -- not what you expected it to say,
  and not a summary that drops the part which surprised you. You are the only
  one who can do this; no other step will. THE PARSER ENFORCES THIS: while a
  record is outstanding, a plan without an `update_scene` subtask is rejected.
- Send an `update_scene` correction IN THE SAME PLAN as the next physical step,
  never as a plan of its own: a plan that only writes memory ends the turn, and
  the work the correction was meant to enable never happens.
"""


class ContextBuilder:
    def __init__(
        self,
        memory: Memory,
        registry: EndpointRegistry,
        system_prompt: Optional[str] = None,
        *,
        now_fn=None,
        scene_memory: Optional[SceneMemory] = None,
        scene_top_k: int = 20,
        user_memory: Optional[UserMemory] = None,
    ) -> None:

        self.memory = memory
        self.registry = registry
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._now_fn = now_fn or time.time
        self._task_start_ts: Optional[float] = None
        self.scene_memory = scene_memory
        self.scene_top_k = scene_top_k
        self.user_memory = user_memory


    # The planner VLM wants:
    #   system: string
    #   messages: [{"role": "user"/"assistant", "content": "..."}]
    # We condense memory into at most a few lines of prior tool_results to keep prompts small.

    def mark_task_start(self) -> None:
        """Record the wall-clock time at which the current top-level task
        began. AgentLoop calls this in ``handle()`` so the Planner sees a
        stable ``TASK START TIME`` across replans within the same user turn.
        """
        self._task_start_ts = self._now_fn()

    def _time_header(self) -> str:
        now_ts = self._now_fn()
        lines = [f"CURRENT TIME: {_fmt_iso(now_ts)}"]
        if self._task_start_ts is not None:
            lines.append(f"TASK START TIME: {_fmt_iso(self._task_start_ts)}")
            elapsed = max(0.0, now_ts - self._task_start_ts)
            lines.append(f"ELAPSED SINCE TASK START: {elapsed:.1f}s")
        lines.append(
            "Use these timestamps to reason about real-world delays "
            "(e.g. 'boil 3 minutes, then turn off heat'). When a delay is "
            "required between two subtasks, insert a `wait` subtask with "
            "inputs.duration_s and chain via depends_on."
        )
        return "\n".join(lines)

    def system_text(self) -> str:
        endpoints = self.registry.snapshot_for_planner_vlm()
        parts: list[str] = [
            self.system_prompt,
            "LIVE VISION AUTHORITY: historical actions may no longer describe the "
            "current scene; determine current object state and goal completion from "
            "the attached live camera image, not from prior results or past verdicts.",
            self._time_header(),
            self._user_memory_block(),
            self._scene_memory_block(),

            "Available endpoints:\n"
            + json.dumps(endpoints, ensure_ascii=False, indent=2),
        ]
        return "\n\n".join(p for p in parts if p)

    def _user_memory_block(self) -> str:
        if self.user_memory is None:
            return ""
        rendered = self.user_memory.render()
        if not rendered.strip():
            return ""
        block = (
            "USER MEMORY (durable per-user facts, as markdown; one `# user: <id>`\n"
            "heading per user, then `## <category>` sections):\n"
            + rendered
            + "\n"
            "How to READ these facts:\n"
            "- profile:* identity / habits / preferences -- INFER task behavior\n"
            "  from them (e.g. lasting dietary constraints shape prepared food;\n"
            "  profile:home can fill a `web_query` location).\n"
            "- entity:* owned-object attributes -- resolve owner references\n"
            "  (e.g. rewrite \"A's cup\" via entity:cup / color). A\n"
            "  `reference_image` value is a path you may pass to any endpoint\n"
            "  whose inputs accept a reference image, to disambiguate which\n"
            "  instance of an object is meant.\n"
        )
        # profile:avoid:* facts are surfaced as explicit AVOID-CONSTRAINTS the
        # planner must respect -- mirrors the positive-fact framing above but as
        # prohibitions. Only appended when such facts are present, so
        # positive-only profiles keep their exact prior prompt block.
        if f"## {AVOID_CATEGORY_PREFIX}" in rendered:
            block += (
                "- profile:avoid:* are AVOID-CONSTRAINTS: hard prohibitions.\n"
                "  INFER behavior that RESPECTS them (prefer an allowed\n"
                "  alternative that still achieves the goal over refusing).\n"
            )
        block += (
            "A fresh value in the new user message always wins over stored\n"
            "memory for this turn (one-off wishes override the profile for the\n"
            "task only). To persist a NEW durable fact -- or to retract keys\n"
            "that a correction has superseded -- dispatch `update_user_memory`\n"
            "(see that endpoint's description)."
        )
        return block

    def _scene_memory_block(self) -> str:
        if self.scene_memory is None:
            return ""

        rendered = self.scene_memory.snapshot_for_planner(self.scene_top_k)
        if not rendered or not rendered.strip():
            head = "SCENE MEMORY: (empty)"
        else:
            head = (
                "SCENE MEMORY (last established content per place, as markdown):\n"
                + rendered
            )
        return head + "\n" + SCENE_MEMORY_POLICY

    def build_for_planning(
        self, user_task: str, current_plan_id: Optional[str] = None
    ) -> tuple[str, list[dict]]:
        """Return (system_text, messages).

        current_plan_id=None means a FRESH top-level instruction: only dialogue
        (the user's words / references / preferences and the planner's
        conversational replies) carries over; durable memory and the live camera
        frame supply the rest. Another plan's EXECUTION RESULTS -- judge verdicts,
        placed_object, visual_evidence, terminal status -- are historical events
        with a plan_id/timestamp, NOT current scene state, so a repeat instruction
        cannot "remember" a placement and skip grounding on the live frame
         Same-plan replan keeps its own plan's
        results (current_plan_id); reaction context is built by build_for_reaction.
        """
        msgs: list[dict] = []
        _DIALOGUE = ("user", "assistant")
        ctx: list = []
        for m in self.memory.messages():
            if m.role in _DIALOGUE:
                ctx.append(m)                                  # dialogue always survives
            elif current_plan_id is not None and m.plan_id == current_plan_id:
                ctx.append(m)                                  # this plan's own results only
        for m in ctx[-10:]:
            role = m.role if m.role in _DIALOGUE else "user"
            msgs.append({"role": role, "content": m.content})
        msgs.append({"role": "user", "content": user_task})
        return self.system_text(), msgs

    def build_for_packing_after_weather(
        self, user_task: str, weather_plan: Plan
    ) -> tuple[str, list[dict]]:
        """Phase-2 outdoor packing: plan arm picks using a finished weather query.

        Injects the weather result explicitly so item selection is grounded in
        that lookup (not a pre-planned list emitted before the query returned).
        """
        weather_text = weather_result_text(weather_plan) or "(no weather result)"
        system, msgs = self.build_for_planning(user_task)
        msgs.append(
            {
                "role": "user",
                "content": (
                    "WEATHER RESULT (already obtained; do NOT call web_query "
                    "again):\n"
                    f"{weather_text}\n\n"
                    "OUTDOOR PACKING phase 2: weather is already known (block "
                    "above). Using that result, the destination in the "
                    "original request, the active user's USER MEMORY, and "
                    "skill_library (copy entries VERBATIM; fill {placeholders} "
                    "only), emit the arm pick/place chain now. Do not "
                    "re-query weather. Do not emit web_query."
                ),
            }
        )
        return system, msgs

    def build_for_reaction(
        self,
        plan: Plan,
        events: list[dict],
    ) -> tuple[str, list[dict]]:
        """Build messages asking the planner VLM what to dispatch NEXT given events.

        ``events`` is a list of dicts produced by the Reactor, each describing
        one trigger -- a failed/timeout subtask or an asynchronous endpoint
        status update. The planner VLM may respond with an empty-subtasks Plan if no
        action is required.
        """
        # plan summary so the planner VLM knows what's already been dispatched
        sub_lines = []
        for s in plan.subtasks:
            tail = ""
            if s.result is not None:
                if s.result.ok:
                    data = s.result.data
                    brief = (
                        data.get("summary") if isinstance(data, dict) and data.get("summary")
                        else (str(data)[:120] if data is not None else "")
                    )
                    tail = f" -> success: {brief}"
                else:
                    tail = f" -> {s.status}: {s.result.error or ''}"
            elif s.status != "pending":
                tail = f" -> {s.status}"
            sub_lines.append(
                f"  - {s.id} @ {s.endpoint} (status={s.status}, attempts={s.attempts}): "
                f"{s.instruction[:160]}{tail}"
            )
        plan_block = (
            f"Current Plan {plan.id}:\n"
            f"  goal: {plan.goal}\n"
            f"  subtasks:\n" + "\n".join(sub_lines)
        )

        evt_lines = []
        image_parts: list[dict] = []
        for i, ev in enumerate(events, 1):
            kind = ev.get("kind") or "event"
            ep = ev.get("endpoint") or "?"
            sid = ev.get("subtask_id") or "-"
            detail = ev.get("detail") or ""
            evt_lines.append(f"  [{i}] kind={kind} subtask={sid} endpoint={ep}: {detail}")
            image_parts.extend(_event_image_parts(i, ev))
        events_block = "Events to react to:\n" + "\n".join(evt_lines)

        prompt = (
            f"{plan_block}\n\n{events_block}\n\n"
            "Decide what CarryOn should dispatch NEXT.\n"
            "- If the events indicate a failure that you want to retry / route to a\n"
            "  different endpoint, emit subtasks describing that.\n"
            "- If 'before' / 'after' camera images are attached, compare them to\n"
            "  diagnose what went wrong and consider an error-recovery subtask.\n"
            "- If the events are merely informational (e.g. an oven LED turned green)\n"
            "  and trigger a follow-up action, emit that follow-up.\n"
            "- If no action is needed (informational only, or original goal already\n"
            "  achieved), emit a Plan with an EMPTY \"subtasks\" array.\n"
            "- New subtask ids should be globally unique within this session; prefix\n"
            "  them with 'r<n>-' (n = how many reaction rounds so far) to avoid\n"
            "  collision with the original plan's ids.\n"
            "- depends_on may reference other NEW subtasks in this same response, but\n"
            "  NOT subtasks from the original plan."
        )
        if image_parts:
            content: list[dict] = [{"type": "text", "text": prompt}] + image_parts
            return self.system_text(), [{"role": "user", "content": content}]
        return self.system_text(), [{"role": "user", "content": prompt}]

    def build_for_summary(self, plan: Plan, results_text: str) -> tuple[str, list[dict]]:  # noqa: E501
        return _build_for_summary(plan, results_text)


def weather_result_text(plan: Plan) -> str:
    """Best-effort weather summary string from a finished ``web_query`` plan."""
    for sub in plan.subtasks:
        if sub.endpoint != "web_query":
            continue
        if sub.status != "success" or sub.result is None or not sub.result.ok:
            continue
        data = sub.result.data
        if isinstance(data, dict):
            text = data.get("summary") or data.get("answer") or ""
            if text:
                return str(text)
            return json.dumps(data, ensure_ascii=False)[:2000]
        if data is not None:
            return str(data)[:2000]
    return ""


def looks_like_outdoor_packing(text: str) -> bool:
    """Heuristic: user wants items packed for going somewhere outdoors/out.

    Kept intentionally light -- used only to enforce weather-before-picks,
    not to encode any task-specific packing rubric.
    """
    t = " ".join((text or "").lower().split())
    if not t:
        return False
    going = any(
        p in t
        for p in (
            "heading out",
            "heading to",
            "going out",
            "for a run",
            "to the gym",
            "to the park",
            "to work",
            "commute",
        )
    )
    pack = any(
        p in t
        for p in (
            "put what i need",
            "what i need into",
            "into the plate",
            "into the bag",
            "pack what",
            "pack my",
        )
    )
    return going and pack


def validate_outdoor_packing_weather_gate(
    plan: Plan,
    *,
    user_task: str,
    web_query_available: bool,
) -> None:
    """Reject phase-1 packing plans that skip weather or mix in arm picks.

    Raises ``PlanError`` so the planner retry loop can ask the VLM again.
    """
    from ..errors import PlanError

    if not web_query_available or not looks_like_outdoor_packing(user_task):
        return
    if not plan.subtasks:
        return
    non_weather = [s for s in plan.subtasks if s.endpoint != "web_query"]
    weather = [s for s in plan.subtasks if s.endpoint == "web_query"]
    if non_weather:
        raise PlanError(
            "outdoor packing requires weather first: emit ONLY a web_query "
            "subtask with kind=weather and location from the active user's "
            "home city (no arm / pick / place subtasks in this plan). "
            "Physical picks are planned after the weather result returns."
        )
    has_weather_kind = False
    for s in weather:
        kind = str((s.inputs or {}).get("kind") or "").strip().lower()
        if kind == "weather" or not kind:
            has_weather_kind = True
            break
    if not has_weather_kind:
        raise PlanError(
            "outdoor packing phase 1 must use web_query with kind=weather"
        )


def validate_scene_update_requirement(plan: Plan, *, reveal: str) -> None:
    """Refuse a continuation plan that leaves an observation unrecorded.

    The judge flags an observe step whose reveal has not reached scene memory
    (result.scene_update_required). The write itself stays with the planner --
    it names the place and authors the entry -- but WHETHER it happens stops
    being its memory. Asked in prose four times and then flagged in the very
    result JSON it was following, the planner acted on revealed_contents
    faithfully every round and never once dispatched `update_scene`; the fifth
    formulation was not going to be the one that worked. Same arc as the
    library gate and the observation boundary: prose asks, the parser refuses,
    and the refusal text teaches through the existing repair retry.

    An `update_scene`-only plan is a legal way to satisfy this on the turn's
    last round -- the memory-only stop then ends the turn with the record
    written, which is the right outcome when the goal is already met.
    """
    from ..errors import PlanError

    if any(s.endpoint == "update_scene" for s in plan.subtasks):
        return
    raise PlanError(
        "an earlier step in this turn revealed something that scene memory "
        f"does not record yet: {reveal!r}. This plan must include an "
        "`update_scene` subtask writing down what that step actually revealed "
        "for the place it uncovered (alongside the next physical step, or "
        "alone if nothing else remains). The observation exists only in this "
        "turn's results; if no plan writes it down, the next turn will read "
        "the stale entry and walk back to the wrong place."
    )


def is_outdoor_weather_gate_plan(plan: Plan) -> bool:
    """True when ``plan`` is phase-1 packing: only weather ``web_query``, succeeded.

    Used by the agent loop to trigger phase-2 packing after the lookup returns.
    Plans that already include arm subtasks are left alone (no double packing).
    """
    if not plan.subtasks:
        return False
    if any(s.endpoint != "web_query" for s in plan.subtasks):
        return False
    for sub in plan.subtasks:
        if sub.status != "success" or sub.result is None or not sub.result.ok:
            continue
        kind = str((sub.inputs or {}).get("kind") or "").strip().lower()
        data = sub.result.data if isinstance(sub.result.data, dict) else {}
        data_kind = str(data.get("kind") or "").strip().lower()
        if kind == "weather" or data_kind == "weather":
            return True
    return False


def _event_image_parts(event_index: int, event: dict) -> list[dict]:
    images = event.get("images")
    if not isinstance(images, dict):
        return []
    parts: list[dict] = []
    for key in ("before", "after"):
        for snap in images.get(key) or []:
            if not isinstance(snap, dict):
                continue
            b64 = snap.get("image_b64")
            if not b64:
                continue
            mime = snap.get("mime") or "image/jpeg"
            cam = snap.get("camera") or "camera"
            parts.append(
                {"type": "text", "text": f"event {event_index} {key} ({cam}):"}
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
    return parts


def _fmt_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )


def _build_for_summary(plan: Plan, results_text: str) -> tuple[str, list[dict]]:
    sys = (
        "You are summarizing the result of a completed robot dispatch plan. "
        "Be concise, factual, and mention any failures."
    )
    msg = (
        f"Goal: {plan.goal}\n"
        f"Subtask outcomes:\n{results_text}\n\n"
        f"Write a 1-3 sentence summary for the user."
    )
    return sys, [{"role": "user", "content": msg}]
