"""Core data structures shared across modules."""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional, TypedDict
Role = Literal['system', 'user', 'assistant', 'tool_result', 'system_event']
JobStatus = Literal['pending', 'running', 'success', 'failed', 'cancelled', 'timeout']
JOB_TERMINAL_STATUSES = frozenset({'success', 'failed', 'cancelled', 'timeout'})
CALLBACK_TERMINAL_STATUSES = frozenset({'success', 'failed', 'cancelled'})
STATUS_EVENT_KEYWORDS = frozenset({'progress', 'status_update', 'update', 'info', 'event'})

def is_terminal_status(status: str) -> bool:
    """Whether ``status`` is one of the Job's terminal states (incl. timeout)."""
    return status in JOB_TERMINAL_STATUSES
RecoveryHookResult = Literal['resumed', 'handled_failed', False]

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
    inputs: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    job_id: Optional[str] = None
    status: JobStatus = 'pending'
    result: Optional[ToolResult] = None
    attempts: int = 0

@dataclass
class Plan:
    id: str
    goal: str
    subtasks: list[Subtask]

    @staticmethod
    def new_id() -> str:
        return f'plan-{uuid.uuid4().hex[:8]}'

    def get(self, sid: str) -> Optional[Subtask]:
        for s in self.subtasks:
            if s.id == sid:
                return s
        return None
