"""EndpointInvoker: synchronous "submit + await terminal result" helper.

Used by in-process skills (future composites) that need to
orchestrate calls to real robot endpoints (open_drawer, etc.) without going
through the Executor's DAG machinery.

The skill calls :meth:`EndpointInvoker.call`, which:

1. Generates a unique ``skill-<uuid>`` request id.
2. Submits the instruction via the chosen endpoint's Transport.
3. Awaits a future that the main intake feeds when the matching callback
   arrives at the result channel.

The result router (``jobs/intake.ResultRouter``) checks
``EndpointInvoker.has_pending(job_key)`` before falling through to
``JobManager.on_result``; this gives the invoker first dibs on its own
request ids while leaving normal subtask callbacks untouched.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from ..endpoints import EndpointRegistry
from ..errors import EndpointNotFound
from ..transports.base import Transport


class EndpointInvoker:
    def __init__(
        self,
        *,
        registry: EndpointRegistry,
        transports: dict[str, Transport],
        callback_url: str,
        default_deadline_s: float = 120.0,
    ) -> None:
        self._registry = registry
        self._transports = transports
        self._callback_url = callback_url
        self._default_deadline_s = default_deadline_s
        self._pending: dict[str, asyncio.Future] = {}

    @staticmethod
    def is_skill_request_id(request_id: str) -> bool:
        return isinstance(request_id, str) and request_id.startswith("skill-")

    def has_pending(self, request_id: str) -> bool:
        return request_id in self._pending

    async def feed_result(self, request_id: str, payload: dict) -> None:
        fut = self._pending.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(dict(payload or {}))

    async def call(
        self,
        endpoint_name: str,
        instruction: str,
        inputs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        try:
            endpoint = self._registry.get(endpoint_name)
        except EndpointNotFound:
            return {
                "status": "failed",
                "error": f"endpoint {endpoint_name!r} not registered",
            }
        transport = self._transports.get(endpoint.transport)
        if transport is None:
            return {
                "status": "failed",
                "error": f"no transport for {endpoint.transport!r}",
            }
        request_id = f"skill-{uuid.uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut
        deadline = timeout_s or self._default_deadline_s
        try:
            submit = await transport.send(
                endpoint,
                instruction,
                inputs or {},
                meta={
                    "request_id": request_id,
                    "callback_url": self._callback_url,
                    "deadline_s": deadline,
                },
            )
            if not submit.accepted:
                return {
                    "status": "failed",
                    "error": f"endpoint {endpoint_name!r} rejected: {submit.reason}",
                }
            try:
                return await asyncio.wait_for(fut, timeout=deadline)
            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "error": f"endpoint {endpoint_name!r} did not callback within {deadline}s",
                }
        finally:
            self._pending.pop(request_id, None)
