from __future__ import annotations

from typing import Optional

from aiohttp import web

from ...logging_util import EventLogger
from .base import OnResult


class WebhookChannel:
    """Embeds a tiny aiohttp server listening for VLA callbacks.

    Route: POST /callback/{job_key}
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 9000,
        public_url: str = "http://127.0.0.1:9000",
        auth_token: Optional[str] = None,
        logger: Optional[EventLogger] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.public_url = public_url.rstrip("/")
        self.auth_token = auth_token
        self._logger = logger

        self._on_result: Optional[OnResult] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.BaseSite] = None

    def set_logger(self, logger: EventLogger | None) -> None:
        self._logger = logger

    # -- ResultChannel API --

    async def start(self, on_result: OnResult) -> None:
        self._on_result = on_result
        app = web.Application()
        app.router.add_post("/callback/{job_key}", self._handle_callback)
        app.router.add_get("/healthz", self._handle_health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        self._runner = None
        self._site = None

    def external_url(self, job_key: str) -> str:
        return f"{self.public_url}/callback/{job_key}"

    # -- handlers --

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def _handle_callback(self, request: web.Request) -> web.Response:
        if self.auth_token:
            if request.headers.get("X-CarryOn-Token") != self.auth_token:
                return web.json_response({"ok": False, "error": "unauthorized"}, status=401)
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "invalid json"}, status=400)

        job_key = request.match_info.get("job_key", "")
        if self._logger is not None:
            try:
                self._logger.transport_callback(job_key=job_key, payload=payload or {})
            except Exception:
                pass  # never kill the handler because of a log
        if self._on_result is not None:
            # Do not block the handler with heavy work; on_result is supposed to be fast.
            await self._on_result(job_key, payload or {})
        return web.json_response({"ok": True})
