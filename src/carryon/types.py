"""Core data structures shared across modules."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, TypedDict

Role = Literal["system", "user", "assistant", "tool_result", "system_event"]
JobStatus = Literal["pending", "running", "success", "failed", "cancelled", "timeout"]

# ---- Job status vocabulary (single source of truth) ----
# Two distinct concepts that must NOT be conflated:
#
#   JOB_TERMINAL_STATUSES      -- every state in which a Job is "done". Includes
#                                 ``timeout``, which the JobManager tick sets
#                                 *internally*; it is never reported by an
#                                 external callback.
#   CALLBACK_TERMINAL_STATUSES -- the terminal statuses an external endpoint
#                                 callback (or a VLM Judge verdict) may report.
#                                 Excludes ``timeout`` precisely because that is
#                                 an internal deadline decision, not something a
#                                 callback announces.
#
# Consumers (JobManager, Executor, Judge routing/client) reference these instead
# of hard-coding the literal sets, so adding a status is a one-line change.
JOB_TERMINAL_STATUSES = frozenset({"success", "failed", "cancelled", "timeout"})
CALLBACK_TERMINAL_STATUSES = frozenset({"success", "failed", "cancelled"})

# Non-terminal, endpoint-pushed progress signals. They do NOT advance the job
# state machine; the JobManager forwards them to the Reactor.
STATUS_EVENT_KEYWORDS = frozenset(
    {"progress", "status_update", "update", "info", "event"}
)


def is_terminal_status(status: str) -> bool:
    """Whether ``status`` is one of the Job's terminal states (incl. timeout)."""
    return status in JOB_TERMINAL_STATUSES


# ---- JobManager <-> Recovery hook contract ----
# The tri-valued result a recovery hook returns to the JobManager on a
# judge-driven ``failed`` terminal. Each value drives a distinct, safety-relevant
# branch (whether the manager fires another cancel that would move the arm):
#
#   "resumed"        -- recovery succeeded AND handed control back; the same VLA
#                       request resumed. The manager reopens the job (running)
#                       and re-arms the judge instead of finalizing failed.
#   "handled_failed" -- recovery took over and the arm is safely in standby, but
#                       the policy failed. The manager advances the DAG WITHOUT a
#                       second judge-driven cancel (which would move the arm again).
#   False            -- recovery did not take over; the manager takes the normal
#                       judge-driven cancel/release path.
RecoveryHookResult = Literal["resumed", "handled_failed", False]


class CallbackPayload(TypedDict, total=False):
    """The JSON every endpoint POSTs to its callback URL when a Job changes state.

    Single source of truth for the result-channel contract: external VLAs, the
    in-process LocalTransport, the MockTransport, and the ROS callback channel
    all produce this shape; ``JobManager.on_result`` and the VLM-Judge routing
    consume it. It is the most important external boundary in the system, so the
    contract lives here rather than scattered across producer/consumer code.

    All keys are optional (``total=False``): it stays a plain ``dict`` at runtime
    (TypedDict is not enforced), so an unknown VLA sending extra/missing fields
    still works -- the contract is documentation-as-code, not a validator.

    Fields:
      status   -- a ``JobStatus`` terminal value (success/failed/cancelled), OR a
                  non-terminal progress word (running + STATUS_EVENT_KEYWORDS).
                  Anything outside CALLBACK_TERMINAL_STATUSES is treated as a
                  non-terminal status event.
      terminal -- explicit override: ``False`` forces the payload down the
                  non-terminal status-event path regardless of ``status``.
      result   -- endpoint-specific success body (often a dict with ``summary``).
      error    -- failure description (when status=failed/cancelled).
      meta     -- side-band: ``source`` ("vlm_judge" marks a Judge verdict so the
                  manager treats it as authoritative), ``latency_ms``, etc.
    """

    status: str
    terminal: bool
    result: Any
    error: str
    meta: dict


# The one callback-sink signature used across the whole result path: every
# producer (Local/Mock transports, webhook/ROS channels, the VLM-Judge
# supervisor) calls some ``OnResult`` with ``(job_key, CallbackPayload)`` to
# deliver a result or status update; ``JobManager``/``ResultRouter`` provide the
# sinks. Defined once here (next to CallbackPayload) so the many wiring points
# share one name instead of re-declaring identical aliases.
OnResult = Callable[[str, CallbackPayload], Awaitable[None]]


@dataclass
class Message:
    role: Role
    content: str
    endpoint: Optional[str] = None
    subtask_id: Optional[str] = None
    plan_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    raw: Any = None


@dataclass
class SubmitResult:
    accepted: bool
    job_id: Optional[str] = None
    reason: Optional[str] = None
    raw: Any = None


@dataclass
class Subtask:
    id: str
    endpoint: str
    instruction: str
    # The completion criterion for this subtask, written for the VLM Judge only.
    # DUAL-FIELD CONTRACT: ``instruction`` is the exact text dispatched to the
    # VLA and must never be rewritten for the Judge's benefit -- a learned
    # policy is prompt-sensitive and only reliable on its skill_library phrasing.
    # ``objective`` carries the judgeable "what does done look like" statement
    # that would otherwise pollute the instruction. It NEVER reaches an endpoint;
    # the Executor passes it to the Judge supervisor out-of-band. Empty means
    # "no separate criterion", and the Judge falls back to the instruction.
    objective: str = ""
    inputs: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    # Why this step is in the plan, declared by the Planner: "observe" means its
    # point is to find something out and what comes next depends on the result;
    # "act" means the need for it is already established. Empty means undeclared
    # -- free-text endpoints and scripted plans never have to say. The parse gate
    # reads it to refuse plans that pre-commit work behind an observation nobody
    # has made yet; see planner/schema.py::_validate_observation_boundary.
    purpose: str = ""

    # runtime
    job_id: Optional[str] = None
    status: JobStatus = "pending"
    result: Optional[ToolResult] = None
    attempts: int = 0


@dataclass
class Plan:
    id: str
    goal: str
    subtasks: list[Subtask]
    # Brief natural-language message shown to the user when the plan is
    # created (ack / what you'll do). Empty means "no user-facing reply".
    user_reply: str = ""
    # Runtime-only provenance for downstream observers such as the VLM Judge.
    # The Planner VLM does not emit this field: Planner fills it from the
    # original turn text after parsing. Consumers must treat it as read-only
    # background, while Subtask.instruction/objective remain authoritative.
    original_user_request: str = ""

    @staticmethod
    def new_id() -> str:
        return f"plan-{uuid.uuid4().hex[:8]}"

    def get(self, sid: str) -> Optional[Subtask]:
        for s in self.subtasks:
            if s.id == sid:
                return s
        return None

    def format_subtasks(
        self,
        *,
        subs: Optional[list[Subtask]] = None,
        label: str = "subtasks",
    ) -> str:
        """Readable listing of subtasks for the REPL / logs."""
        entries = list(subs) if subs is not None else list(self.subtasks)
        if not entries:
            return f"[{self.id}] {label}: (none)"
        lines = [f"[{self.id}] {label} ({len(entries)}):"]
        for s in entries:
            deps = f" (after {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"  - {s.id} @ {s.endpoint}{deps}: {s.instruction}")
        return "\n".join(lines)
