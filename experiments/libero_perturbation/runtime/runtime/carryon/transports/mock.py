from __future__ import annotations
import asyncio
import time
from typing import Any, Optional
from ..endpoints import RobotEndpoint
from ..logging_util import EventLogger
from ..types import OnResult, SubmitResult

class MockTransport:
    """Scripted transport for tests.

    Optional ``auto_callback`` + ``auto_callback_after_s`` lets you drive the
    whole pipeline (planner -> executor -> callback -> notifier) with no real
    VLA service. When set, every *accepted* submit schedules a background task
    that sleeps ``auto_callback_after_s`` seconds and then invokes
    ``auto_callback(request_id, {"status":"success", "result": {...}})``.
    """
    name = 'mock'

    def __init__(self, send_responses: dict[str, list[Any]] | None=None, cancel_responses: dict[str, list[Any]] | None=None, *, auto_callback: Optional[OnResult]=None, auto_callback_after_s: float=0.0, logger: Optional[EventLogger]=None) -> None:
        self.send_responses = send_responses or {}
        self.cancel_responses = cancel_responses or {}
        self.record: list[dict] = []
        self._default_send = {'accepted': True, 'job_id': 'mock-job'}
        self._auto_callback = auto_callback
        self._auto_after = float(auto_callback_after_s)
        self._pending: set[asyncio.Task] = set()
        self._logger = logger

    def set_logger(self, logger: EventLogger | None) -> None:
        self._logger = logger

    def set_auto_callback(self, cb: OnResult, after_s: float) -> None:
        """Attach (or replace) the auto-callback hook after construction.

        Useful because Transports are built before JobManager is ready; main.py
        wires this up once both are alive.
        """
        self._auto_callback = cb
        self._auto_after = float(after_s)

    def _pop(self, store: dict, key: str, default):
        seq = store.get(key)
        if not seq:
            return default
        return seq.pop(0)

    async def send(self, endpoint: RobotEndpoint, instruction: str, inputs: dict, meta: dict) -> SubmitResult:
        self.record.append({'kind': 'send', 'endpoint': endpoint.name, 'instruction': instruction, 'inputs': dict(inputs or {}), 'meta': dict(meta or {}), 'ts': time.time()})
        pending = self._logger.transport_submit_request(transport=self.name, endpoint=endpoint.name, address=endpoint.address, instruction=instruction, inputs=inputs, meta=meta) if self._logger is not None else None
        resp = self._pop(self.send_responses, endpoint.name, self._default_send)
        if isinstance(resp, BaseException):
            if self._logger is not None and pending is not None:
                self._logger.transport_submit_error(pending, resp)
            raise resp
        sr = SubmitResult(accepted=bool(resp.get('accepted', False)), job_id=resp.get('job_id'), reason=resp.get('reason'), raw=resp)
        if self._logger is not None and pending is not None:
            self._logger.transport_submit_response(pending, accepted=sr.accepted, job_id=sr.job_id, reason=sr.reason, raw=resp)
        if sr.accepted and self._auto_callback is not None:
            req_id = (meta or {}).get('request_id') or sr.job_id or 'mock-req'
            short = instruction if len(instruction) <= 200 else instruction[:200] + '...'
            brief = f'mock VLA completed: {short}'
            payload = {'status': 'success', 'result': {'summary': brief, 'endpoint': endpoint.name}, 'meta': {'latency_ms': int(self._auto_after * 1000)}}
            task = asyncio.create_task(self._delayed_callback(req_id, payload, self._auto_after))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)
        return sr

    async def _delayed_callback(self, request_id: str, payload: dict, after_s: float) -> None:
        try:
            if after_s > 0:
                await asyncio.sleep(after_s)
            if self._auto_callback is not None:
                await self._auto_callback(request_id, payload)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def cancel(self, endpoint: RobotEndpoint, request_id: str, *, return_home: bool=False, terminal_reason: str='cancel') -> bool:
        self.record.append({'kind': 'cancel', 'endpoint': endpoint.name, 'request_id': request_id, 'return_home': bool(return_home), 'terminal_reason': str(terminal_reason), 'ts': time.time()})
        resp = self._pop(self.cancel_responses, endpoint.name, True)
        if isinstance(resp, BaseException):
            ok = False
        else:
            ok = bool(resp)
        if self._logger is not None:
            self._logger.transport_cancel(transport=self.name, endpoint=endpoint.name, request_id=request_id, ok=ok)
        return ok

    async def aclose(self) -> None:
        for t in list(self._pending):
            t.cancel()
        self._pending.clear()
