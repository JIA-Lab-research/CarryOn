from __future__ import annotations

from typing import Any, Optional

import httpx

from ..endpoints import RobotEndpoint
from ..errors import TransportError
from ..logging_util import EventLogger
from ..types import SubmitResult


def _endpoint_base_url(ep: RobotEndpoint) -> str:
    """HTTP base for cancel/reset: ``extra['base_url']`` or address minus last path."""
    base = ep.extra.get("base_url")
    if not base:
        addr = ep.address.rstrip("/")
        # strip the last path component (e.g. "/instruct" or "/execute")
        i = addr.rfind("/")
        base = addr[:i] if i > 0 and "://" not in addr[i:] else addr
    return str(base).rstrip("/")


def _cancel_url(ep: RobotEndpoint, request_id: str) -> str:
    """POST {base}{cancel_path}/{request_id}."""
    return f"{_endpoint_base_url(ep)}{ep.cancel_path}/{request_id}"


def _reset_url(ep: RobotEndpoint) -> str:
    """POST {base}/reset — gello hard-reset / stop+home (best-effort)."""
    return f"{_endpoint_base_url(ep)}/reset"


class HTTPTransport:
    name = "http"

    def __init__(
        self,
        *,
        default_timeout: float = 10.0,
        auth_token: str | None = None,
        client: httpx.AsyncClient | None = None,
        logger: Optional[TransportLogger] = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._auth_token = auth_token
        self._client = client or httpx.AsyncClient(timeout=default_timeout)
        self._owns_client = client is None
        self._logger = logger

    def set_logger(self, logger: TransportLogger | None) -> None:
        self._logger = logger

    def _headers(self, ep: RobotEndpoint) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = ep.extra.get("auth_token") or self._auth_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def send(
        self,
        endpoint: RobotEndpoint,
        instruction: str,
        inputs: dict,
        meta: dict,
    ) -> SubmitResult:
        body: dict[str, Any] = {
            "instruction": instruction,
            "inputs": inputs or {},
            "request_id": meta.get("request_id"),
            "callback_url": meta.get("callback_url"),
            "deadline_s": meta.get("deadline_s"),
        }
        pending = (
            self._logger.transport_submit_request(
                transport=self.name,
                endpoint=endpoint.name,
                address=endpoint.address,
                instruction=instruction,
                inputs=inputs,
                meta=meta,
            )
            if self._logger is not None
            else None
        )
        try:
            r = await self._client.post(
                endpoint.address,
                json=body,
                headers=self._headers(endpoint),
                timeout=meta.get("timeout_s", self._default_timeout),
            )
        except httpx.HTTPError as e:
            if self._logger is not None and pending is not None:
                self._logger.transport_submit_error(pending, e)
            raise TransportError(f"http error to {endpoint.name}: {e}") from e

        if r.status_code // 100 != 2:
            err = TransportError(
                f"http {r.status_code} from {endpoint.name}: {r.text[:200]}"
            )
            if self._logger is not None and pending is not None:
                self._logger.transport_submit_error(pending, err)
            raise err
        try:
            data = r.json()
        except Exception as e:
            if self._logger is not None and pending is not None:
                self._logger.transport_submit_error(pending, e)
            raise TransportError(f"invalid json from {endpoint.name}: {e}") from e
        sr = SubmitResult(
            accepted=bool(data.get("accepted", False)),
            job_id=data.get("job_id"),
            reason=data.get("message") or data.get("reason"),
            raw=data,
        )
        if self._logger is not None and pending is not None:
            self._logger.transport_submit_response(
                pending,
                accepted=sr.accepted,
                job_id=sr.job_id,
                reason=sr.reason,
                raw=data,
            )
        return sr

    async def cancel(
        self,
        endpoint: RobotEndpoint,
        request_id: str,
        *,
        return_home: bool = False,
    ) -> bool:
        url = _cancel_url(endpoint, request_id)
        body: dict[str, Any] = {"return_home": bool(return_home)}
        ok = False
        try:
            r = await self._client.post(
                url,
                json=body,
                headers=self._headers(endpoint),
                timeout=self._default_timeout,
            )
            ok = r.status_code // 100 == 2
        except httpx.HTTPError:
            ok = False
        if self._logger is not None:
            self._logger.transport_cancel(
                transport=self.name,
                endpoint=endpoint.name,
                request_id=request_id,
                ok=ok,
            )
        return ok

    async def reset(self, endpoint: RobotEndpoint) -> bool:
        """Best-effort POST /reset (stop + home). Missing endpoint → False."""
        url = _reset_url(endpoint)
        ok = False
        try:
            r = await self._client.post(
                url,
                json={},
                headers=self._headers(endpoint),
                timeout=self._default_timeout,
            )
            ok = r.status_code // 100 == 2
        except httpx.HTTPError:
            ok = False
        if self._logger is not None:
            self._logger.print_summary(
                "reset", f"{self.name}/{endpoint.name}", f"ok={ok}"
            )
        return ok

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
