"""Multi-camera snapshot helper shared by VLM Judge, Reactor, and skills.

Wraps :func:`fetch_image_bytes` for the common pattern of "fetch every
configured camera in parallel and return a serialisable list of snapshots".
The returned :class:`SceneSnapshot` carries base64-encoded image bytes plus
mime, so callers can either persist them or convert them to OpenAI
multimodal ``image_url`` parts via :meth:`SceneSnapshot.to_image_url_part`.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .images import downscale_image_bytes, fetch_image_bytes


@dataclass
class SceneSnapshot:
    camera: str
    image_b64: str
    mime: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "image_b64": self.image_b64,
            "mime": self.mime,
            "ts": self.ts,
        }

    def to_image_url_part(self, *, max_edge_px: int | None = None) -> dict:
        """OpenAI multimodal ``image_url`` part for this snapshot.

        ``max_edge_px`` opts into downscaling the frame's longer edge (see
        :func:`downscale_image_bytes`); ``None`` (default) sends the stored
        bytes untouched, which is what persisted snapshots and the Judge's own
        history frames want. The stored snapshot is never mutated -- callers
        may render the same snapshot at different caps.
        """
        mime, b64 = self.mime, self.image_b64
        if max_edge_px:
            try:
                body, mime = downscale_image_bytes(
                    base64.b64decode(b64), mime, max_edge_px=max_edge_px
                )
                b64 = base64.b64encode(body).decode("ascii")
            except Exception:  # noqa: BLE001 - undecodable b64: send as stored
                mime, b64 = self.mime, self.image_b64
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneSnapshot":
        return cls(
            camera=str(data.get("camera") or "camera"),
            image_b64=str(data.get("image_b64") or ""),
            mime=str(data.get("mime") or "image/jpeg"),
            ts=float(data.get("ts") or time.time()),
        )


class SceneCapture:
    """Fetch every configured camera's snapshot in parallel.

    Failures from individual cameras are logged inside
    :func:`fetch_image_bytes` and dropped; this method never raises.
    """

    def __init__(
        self,
        cameras: list[dict[str, Any]] | None,
        *,
        http_client: httpx.AsyncClient | None = None,
        image_fetch_timeout_s: float = 5.0,
    ) -> None:
        self.cameras = list(cameras or [])
        self._http = http_client or httpx.AsyncClient(timeout=image_fetch_timeout_s)
        self._owns_http = http_client is None
        self.image_fetch_timeout_s = image_fetch_timeout_s

    async def capture(
        self, camera_names: list[str] | None = None
    ) -> list[SceneSnapshot]:
        """Snapshot the configured cameras.

        ``camera_names`` optionally restricts the capture to the cameras whose
        ``name`` is in the list (order preserved from the configured set) --
        used by callers that only want the view(s) covering one location.
        ``None`` (default) captures every configured camera, unchanged legacy
        behaviour. Names that match no configured camera are ignored; if the
        filter matches nothing, an empty list is returned.
        """
        cams_with_url = [
            c for c in self.cameras
            if isinstance(c, dict) and c.get("snapshot_url")
        ]
        if camera_names is not None:
            wanted = {str(n) for n in camera_names}
            cams_with_url = [
                c for c in cams_with_url if str(c.get("name") or "") in wanted
            ]
        if not cams_with_url:
            return []
        results = await asyncio.gather(
            *(
                fetch_image_bytes(self._http, c, timeout_s=self.image_fetch_timeout_s)
                for c in cams_with_url
            ),
            return_exceptions=False,
        )
        snapshots: list[SceneSnapshot] = []
        for cam, fetched in zip(cams_with_url, results):
            if fetched is None:
                continue
            body, mime = fetched
            b64 = base64.b64encode(body).decode("ascii")
            snapshots.append(
                SceneSnapshot(
                    camera=str(cam.get("name") or "camera"),
                    image_b64=b64,
                    mime=mime,
                )
            )
        return snapshots

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


__all__ = ["SceneCapture", "SceneSnapshot"]
