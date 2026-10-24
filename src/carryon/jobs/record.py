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
    callback_url: str = ""
    status: JobStatus = "pending"
    submit_ts: float = 0.0
    deadline_ts: float = 0.0
    result: Optional[ToolResult] = None
    attempts: int = 0
    # Set when the executor has acknowledged a cancel request (or any other
    # late callback arrives after the job has already been closed by the
    # VLM Judge). The "judge declared done" path waits on this event before
    # advancing the DAG, so the next subtask doesn't race the executor's
    # still-running policy and get rejected with reason=busy.
    # Lazily created -- callers should use ``ensure_release_event()``.
    release_event: Optional[asyncio.Event] = None

    def ensure_release_event(self) -> asyncio.Event:
        if self.release_event is None:
            self.release_event = asyncio.Event()
        return self.release_event


PlanStatus = Literal["running", "completed", "failed", "cancelled"]


@dataclass
class PlanRuntime:
    plan: Plan
    status: PlanStatus = "running"
    job_keys: set[str] = field(default_factory=set)
