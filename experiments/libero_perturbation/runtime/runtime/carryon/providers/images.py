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
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any, Optional
import httpx
_EXT_TO_MIME = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.gif': 'image/gif'}

def _guess_mime(snapshot_url: str, content_type: Optional[str]) -> str:
    if content_type:
        ct = content_type.split(';', 1)[0].strip().lower()
        if ct.startswith('image/'):
            return ct
    path = snapshot_url.split('?', 1)[0].split('#', 1)[0]
    ext = os.path.splitext(path)[1].lower()
    return _EXT_TO_MIME.get(ext, 'image/jpeg')

async def fetch_image_bytes(http_client: httpx.AsyncClient, camera: dict[str, Any], *, timeout_s: float) -> tuple[bytes, str] | None:
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
    url = (camera or {}).get('snapshot_url')
    if not url:
        return None
    local_path = _local_image_path(url)
    if local_path is not None:
        try:
            body = local_path.read_bytes()
        except OSError as e:
            sys.stderr.write(f"[vlm] camera {camera.get('name')!r} local snapshot read failed: {type(e).__name__}: {e}\n")
            return None
        if not body:
            sys.stderr.write(f"[vlm] camera {camera.get('name')!r} local snapshot was empty\n")
            return None
        return (body, _guess_mime(str(local_path), None))
    try:
        response = await http_client.get(url, timeout=timeout_s)
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        sys.stderr.write(f"[vlm] camera {camera.get('name')!r} snapshot fetch failed: {type(e).__name__}: {e}\n")
        return None
    if response.status_code // 100 != 2:
        sys.stderr.write(f"[vlm] camera {camera.get('name')!r} snapshot returned HTTP {response.status_code}\n")
        return None
    body = response.content
    if not body:
        sys.stderr.write(f"[vlm] camera {camera.get('name')!r} snapshot was empty\n")
        return None
    mime = _guess_mime(url, response.headers.get('content-type'))
    return (body, mime)

def _local_image_path(snapshot_url: str) -> Path | None:
    """Return a local path for prototype/offline snapshot URLs.

    Production camera configs normally use HTTP URLs, but local files make it
    possible to reuse the same judge client against extracted video frames.
    """
    parsed = urlparse(snapshot_url)
    if parsed.scheme == 'file':
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    path = Path(snapshot_url)
    if path.is_absolute() or path.exists():
        return path
    return None

async def fetch_image_url_part(http_client: httpx.AsyncClient, camera: dict[str, Any], *, timeout_s: float) -> dict | None:
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
    b64 = base64.b64encode(body).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}}

def _producer_metadata_from_headers(headers) -> dict[str, Any]:
    specs = {'producer_frame_seq': ('x-carryon-producer-frame-seq', int), 'producer_frame_ts': ('x-carryon-producer-frame-ts', float), 'simulation_step_id': ('x-carryon-simulation-step-id', int), 'simulation_time_s': ('x-carryon-simulation-time-s', float), 'camera_role': ('x-carryon-camera-role', str), 'camera_source': ('x-carryon-camera-source', str)}
    metadata: dict[str, Any] = {}
    for field, (header, cast) in specs.items():
        raw = headers.get(header)
        try:
            metadata[field] = cast(raw) if raw is not None else None
        except (TypeError, ValueError):
            metadata[field] = None
    metadata['producer_metadata_complete'] = all((metadata.get(field) not in (None, '') for field in specs))
    return metadata

async def fetch_image_url_part_with_metadata(http_client: httpx.AsyncClient, camera: dict[str, Any], *, timeout_s: float) -> tuple[dict, dict[str, Any]] | None:
    """Fetch one image and its atomic producer/simulation identity.

    Producer identity is read from the same HTTP response as the JPEG. Consumer
    clock, file mtime and follow-up metadata requests are deliberately not used
    as freshness evidence.
    """
    url = (camera or {}).get('snapshot_url')
    if not url:
        return None
    local_path = _local_image_path(url)
    if local_path is not None:
        fetched = await fetch_image_bytes(http_client, camera, timeout_s=timeout_s)
        if fetched is None:
            return None
        body, mime = fetched
        metadata = dict((camera or {}).get('producer_metadata') or {})
        metadata['producer_metadata_complete'] = all((metadata.get(field) not in (None, '') for field in ('producer_frame_seq', 'producer_frame_ts', 'simulation_step_id', 'simulation_time_s', 'camera_role', 'camera_source')))
    else:
        try:
            response = await http_client.get(url, timeout=timeout_s)
        except (httpx.HTTPError, asyncio.TimeoutError) as e:
            sys.stderr.write(f"[vlm] camera {camera.get('name')!r} snapshot fetch failed: {type(e).__name__}: {e}\n")
            return None
        if response.status_code // 100 != 2:
            sys.stderr.write(f"[vlm] camera {camera.get('name')!r} snapshot returned HTTP {response.status_code}\n")
            return None
        body = response.content
        if not body:
            sys.stderr.write(f"[vlm] camera {camera.get('name')!r} snapshot was empty\n")
            return None
        mime = _guess_mime(url, response.headers.get('content-type'))
        metadata = _producer_metadata_from_headers(response.headers)
    b64 = base64.b64encode(body).decode('ascii')
    part = {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}}
    return (part, metadata)
__all__ = ['fetch_image_bytes', 'fetch_image_url_part', 'fetch_image_url_part_with_metadata']
