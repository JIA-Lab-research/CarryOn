"""Helpers for turning camera snapshot URLs into OpenAI multimodal
``image_url`` message parts.

Today only :class:`carryon.judge.OpenRouterVLMJudgeClient` calls these; the
helpers live here (under :mod:`carryon.providers`) so when
:class:`carryon.planner.OpenRouterPlannerVLMClient` grows the same feature it
has the right import already in place.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Optional

import httpx


_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _guess_mime(snapshot_url: str, content_type: Optional[str]) -> str:
    if content_type:
        ct = content_type.split(";", 1)[0].strip().lower()
        if ct.startswith("image/"):
            return ct
    path = snapshot_url.split("?", 1)[0].split("#", 1)[0]
    ext = os.path.splitext(path)[1].lower()
    return _EXT_TO_MIME.get(ext, "image/jpeg")


async def fetch_image_bytes(
    http_client: httpx.AsyncClient,
    camera: dict[str, Any],
    *,
    timeout_s: float,
) -> tuple[bytes, str] | None:
    """Fetch ``camera['snapshot_url']`` and return ``(body_bytes, mime)``.

    Returns ``None`` if the camera has no URL or any step of the fetch
    fails (timeout, non-2xx, empty body). Errors are logged to stderr but
    never raised; callers must keep going even when one camera is
    unreachable for one poll.

    This is the raw-bytes sibling of :func:`fetch_image_url_part`, for
    callers whose downstream API does not speak the OpenAI multimodal
    ``image_url`` data-URL shape (e.g. one that posts a separate base64
    field to a private service).
    """
    url = (camera or {}).get("snapshot_url")
    if not url:
        return None
    local_path = _local_image_path(url)
    if local_path is not None:
        try:
            body = local_path.read_bytes()
        except OSError as e:
            sys.stderr.write(
                f"[vlm] camera {camera.get('name')!r} local snapshot read failed: "
                f"{type(e).__name__}: {e}\n"
            )
            return None
        if not body:
            sys.stderr.write(
                f"[vlm] camera {camera.get('name')!r} local snapshot was empty\n"
            )
            return None
        return body, _guess_mime(str(local_path), None)
    try:
        response = await http_client.get(url, timeout=timeout_s)
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        sys.stderr.write(
            f"[vlm] camera {camera.get('name')!r} snapshot fetch failed: "
            f"{type(e).__name__}: {e}\n"
        )
        return None
    if response.status_code // 100 != 2:
        sys.stderr.write(
            f"[vlm] camera {camera.get('name')!r} snapshot returned "
            f"HTTP {response.status_code}\n"
        )
        return None
    body = response.content
    if not body:
        sys.stderr.write(
            f"[vlm] camera {camera.get('name')!r} snapshot was empty\n"
        )
        return None
    mime = _guess_mime(url, response.headers.get("content-type"))
    return body, mime


def _local_image_path(snapshot_url: str) -> Path | None:
    """Return a local path for prototype/offline snapshot URLs.

    Production camera configs normally use HTTP URLs, but local files make it
    possible to reuse the same judge client against extracted video frames.
    """
    parsed = urlparse(snapshot_url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    path = Path(snapshot_url)
    if path.is_absolute() or path.exists():
        return path
    return None


async def fetch_image_url_part(
    http_client: httpx.AsyncClient,
    camera: dict[str, Any],
    *,
    timeout_s: float,
) -> dict | None:
    """Fetch ``camera['snapshot_url']`` and return an OpenAI multimodal
    ``image_url`` part, or ``None`` if the camera has no URL or the fetch
    failed.

    Errors are logged to stderr but never raised; callers must keep going
    even when one camera is unreachable for one poll.
    """
    fetched = await fetch_image_bytes(http_client, camera, timeout_s=timeout_s)
    if fetched is None:
        return None
    body, mime = fetched
    b64 = base64.b64encode(body).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


# Long-edge cap for VLM image attachments. 768px is the tile size the common
# OpenAI-compatible vision encoders (and OpenRouter's normalisation in front of
# them) downscale to anyway, so sending a 1080p camera frame pays base64 and
# upload cost for pixels the model never sees.
DEFAULT_MAX_IMAGE_EDGE_PX = 768

# Formats we can safely re-encode to. Anything else is normalised to JPEG: the
# result is a data URL whose mime we emit ourselves, so the format need not
# match the camera's.
_RESAVE_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}

_resize_unavailable_logged = False


def downscale_image_bytes(
    body: bytes,
    mime: str,
    *,
    max_edge_px: int = DEFAULT_MAX_IMAGE_EDGE_PX,
) -> tuple[bytes, str]:
    """Shrink ``body`` so its LONGER edge is at most ``max_edge_px``.

    Returns ``(body, mime)``, unchanged when the image is already small enough,
    when ``max_edge_px`` is non-positive (cap disabled), or on ANY failure.

    Fail-open on purpose, in two ways that matter operationally:

    - Pillow is an OPTIONAL dependency. The judge/planner hot path must keep
      working on a robot box that never installed it -- an oversized frame
      costs tokens; a crashed planner costs the episode. Missing Pillow is
      logged once, then silently skipped.
    - A corrupt or exotic frame decodes to an exception rather than an image;
      the original bytes still go to the model, which may well cope.

    Aspect ratio is preserved (``Image.thumbnail`` semantics). Upscaling never
    happens: a frame smaller than the cap is returned byte-identical, so this
    is a no-op for already-small images rather than a lossy re-encode.
    """
    global _resize_unavailable_logged
    if not body or max_edge_px <= 0:
        return body, mime
    try:
        from PIL import Image  # noqa: PLC0415 - optional dep, imported lazily
    except ImportError:
        if not _resize_unavailable_logged:
            _resize_unavailable_logged = True
            print(
                "[images] Pillow not installed: sending camera frames at full "
                "resolution (install Pillow to cap them at "
                f"{max_edge_px}px and cut VLM image cost)",
                file=sys.stderr,
            )
        return body, mime
    try:
        with Image.open(io.BytesIO(body)) as img:
            if max(img.size) <= max_edge_px:
                return body, mime
            fmt = _RESAVE_FORMATS.get((mime or "").lower().split(";", 1)[0].strip())
            out_mime = mime if fmt else "image/jpeg"
            fmt = fmt or "JPEG"
            if fmt == "JPEG" and img.mode not in ("RGB", "L"):
                # JPEG cannot hold alpha/palette; flatten before saving.
                img = img.convert("RGB")
            img.thumbnail((max_edge_px, max_edge_px), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            return buf.getvalue(), out_mime
    except Exception as e:  # noqa: BLE001 - see fail-open rationale above
        print(f"[images] downscale failed ({type(e).__name__}: {e}); "
              "sending original frame", file=sys.stderr)
        return body, mime


__all__ = [
    "DEFAULT_MAX_IMAGE_EDGE_PX",
    "downscale_image_bytes",
    "fetch_image_bytes",
    "fetch_image_url_part",
]
