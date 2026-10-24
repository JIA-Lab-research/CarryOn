"""In-process transport that dispatches to a :class:`SkillRegistry`.

The Planner addresses local skills the same way it addresses external
endpoints: it picks a :class:`RobotEndpoint` whose ``transport: local`` and
the ``address`` names a registered skill. The LocalTransport runs the skill
in a background task and POSTs its :class:`ToolResult` back through the same
``on_result`` callback used by HTTP/ROS callbacks.

Cancel is honored: cancelling a request cancels the underlying asyncio.Task,
so skills with long sleeps (e.g. wait, search_container) shut down promptly.
"""
from __future__ import annotations
import asyncio
import time
import uuid
from typing import Optional
from ..endpoints import RobotEndpoint
from ..logging_util import EventLogger
from ..skills.base import SkillRegistry
from ..types import CallbackPayload, OnResult, SubmitResult, ToolResult

class LocalTransport:
    name = 'local'

    def __init__(self, *, skill_registry: SkillRegistry, on_result: OnResult, logger: Optional[EventLogger]=None) -> None:
        self._skill_registry = skill_registry
        self._on_result = on_result
        self._logger = logger
        self._tasks: dict[str, asyncio.Task] = {}

    def set_logger(self, logger: EventLogger | None) -> None:
        self._logger = logger

    async def send(self, endpoint: RobotEndpoint, instruction: str, inputs: dict, meta: dict) -> SubmitResult:
        skill_name = endpoint.address or endpoint.name
        skill = self._skill_registry.get(skill_name)
        request_id = (meta or {}).get('request_id') or f'local-{uuid.uuid4().hex[:12]}'
        pending = self._logger.transport_submit_request(transport=self.name, endpoint=endpoint.name, address=endpoint.address, instruction=instruction, inputs=inputs, meta=meta) if self._logger is not None else None
        if skill is None:
            sr = SubmitResult(accepted=False, reason=f'no skill named {skill_name!r}', raw={'error': 'skill_not_found'})
            if self._logger is not None and pending is not None:
                self._logger.transport_submit_response(pending, accepted=sr.accepted, job_id=None, reason=sr.reason, raw=sr.raw)
            return sr
        task = asyncio.create_task(self._run_skill(skill, instruction, inputs or {}, request_id), name=f'local-skill:{skill_name}:{request_id}')
        self._tasks[request_id] = task
        sr = SubmitResult(accepted=True, job_id=request_id, reason=f'local skill {skill_name!r} scheduled', raw={'skill': skill_name, 'scheduled_at': time.time()})
        if self._logger is not None and pending is not None:
            self._logger.transport_submit_response(pending, accepted=sr.accepted, job_id=sr.job_id, reason=sr.reason, raw=sr.raw)
        return sr

    async def _run_skill(self, skill, instruction: str, inputs: dict, request_id: str) -> None:
        payload: CallbackPayload
        try:
            result: ToolResult = await skill.run(instruction, inputs)
            if result.ok:
                payload = {'status': 'success', 'result': result.data if result.data is not None else {}}
            else:
                payload = {'status': 'failed', 'error': result.error or 'skill returned ok=False'}
        except asyncio.CancelledError:
            payload = {'status': 'cancelled', 'error': 'skill cancelled'}
        except Exception as e:
            payload = {'status': 'failed', 'error': f'skill {skill.name!r} error: {type(e).__name__}: {e}'}
        finally:
            self._tasks.pop(request_id, None)
        try:
            await self._on_result(request_id, payload)
        except Exception:
            pass

    async def cancel(self, endpoint: RobotEndpoint, request_id: str, *, return_home: bool=False, terminal_reason: str='cancel') -> bool:
        task = self._tasks.get(request_id)
        if task is None:
            return False
        task.cancel()
        if self._logger is not None:
            self._logger.transport_cancel(transport=self.name, endpoint=endpoint.name, request_id=request_id, ok=True)
        return True

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
