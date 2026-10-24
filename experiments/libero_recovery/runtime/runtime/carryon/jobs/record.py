from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Literal, Optional
from ..types import JobStatus, Plan, ToolResult

@dataclass
class JobRecord:
    job_key: str
    vla_job_id: Optional[str]
    plan_id: str
    subtask_id: str
    endpoint_name: str
    callback_url: str = ''
    status: JobStatus = 'pending'
    submit_ts: float = 0.0
    deadline_ts: float = 0.0
    result: Optional[ToolResult] = None
    terminal_reason_hint: Optional[str] = None
    attempts: int = 0
    deadline_pause_ts: Optional[float] = None
    release_event: Optional[asyncio.Event] = None

    def ensure_release_event(self) -> asyncio.Event:
        if self.release_event is None:
            self.release_event = asyncio.Event()
        return self.release_event

    def pause_deadline(self, now: float) -> bool:
        """Freeze the wall deadline once; repeated standby calls are idempotent."""
        if self.deadline_pause_ts is not None:
            return False
        self.deadline_pause_ts = float(now)
        return True

    def resume_deadline(self, now: float) -> Optional[float]:
        """Shift the deadline by the frozen interval and return its duration."""
        if self.deadline_pause_ts is None:
            return None
        paused_s = max(0.0, float(now) - self.deadline_pause_ts)
        self.deadline_ts += paused_s
        self.deadline_pause_ts = None
        return paused_s
PlanStatus = Literal['running', 'completed', 'failed', 'cancelled']

@dataclass
class PlanRuntime:
    plan: Plan
    status: PlanStatus = 'running'
    job_keys: set[str] = field(default_factory=set)
