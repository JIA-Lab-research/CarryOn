from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import re
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..providers import (
    OPENROUTER_BASE_URL,
    OpenRouterChatClient,
    OpenRouterHTTPError,
    fetch_image_url_part,
)
from ..types import CALLBACK_TERMINAL_STATUSES, STATUS_EVENT_KEYWORDS
from .config import DEFAULT_SYSTEM_PROMPT, VLMJudgeConfig


# A Judge *verdict*'s terminal set is exactly the callback-reportable terminal
# set (the Judge never emits ``timeout`` -- that is the JobManager's internal
# deadline decision). See carryon.types for the shared vocabulary.
TERMINAL_STATUSES = CALLBACK_TERMINAL_STATUSES
RUNNING_STATUSES = STATUS_EVENT_KEYWORDS | {"running"}

_RESPONSE_SCHEMA_HINT = (
    "Reply with STRICT JSON ONLY (no markdown fence, no prose) matching:\n"
    '{"status":"running|success|failed|cancelled",'
    '"result":{"planner_action":"continue_observing|advance_to_next_task|invoke_correction",'
    '"confidence":<number 0..1>,"visual_evidence":<string>,'
    '"stuck_evidence":<string|null>,"revealed_contents":<string|null>,'
    '"scene_update_required":<boolean>,"recovery_object":<string|null>,'
    '"recovery_point":<[x,y]|null>},'
    '"error":<string|null>}'
)


# Backstop bound on retained watch scopes. The supervisor releases each scope
# when its watch ends (see release_scope), so this only catches watches whose
# release never ran -- a crash, a cancelled task, a caller that is not the
# supervisor. It must stay comfortably above the number of watches alive at
# once (bounded by DAG width, which nothing else caps), because evicting a LIVE
# watch silently re-baselines it: its next poll would pin a mid-episode frame
# as "the scene when this subtask started" and judge every later frame against
# it. Hence 256 rather than a snug fit -- a retained scope costs a few frames.
_MAX_TRACKED_SCOPES = 256


@dataclass
class JudgeDecision:
    status: str
    result: Any = None
    error: str | None = None
    raw: Any = None
    # True when this decision came from a TRANSPORT/API failure (HTTP error,
    # 4xx/5xx from the gateway) rather than from the model. Such a decision is
    # reported as ``running`` -- we learned nothing, so we must not intervene --
    # and this flag is what lets the supervisor count those polls and mark the
    # episode. Without it "the judge was down all along" is indistinguishable
    # from "the judge watched and saw no progress", which is how an outage turns
    # a whole sweep into plausible-looking data.
    api_error: bool = False
    # True when the gateway answered but the reply was unusable (empty,
    # non-JSON, no status, schema-invalid). Distinct from api_error -- the link
    # worked, the model's output did not -- but the CONTROL decision is the
    # same: we learned nothing about the robot, so we must not intervene.
    # Counted separately so an operator can tell "the gateway is down" from
    # "the model is emitting garbage", which have different remedies.
    parse_error: bool = False

    @property
    def inconclusive(self) -> bool:
        """This poll produced no information about the robot."""
        return self.api_error or self.parse_error

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class _Scope:
    """One watch's frame memory: per-camera baseline + rolling history."""

    def __init__(self, history_window_frames: int) -> None:
        self.history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=history_window_frames)
        )
        # First successfully-fetched frame per camera IN THIS WATCH = the
        # baseline the judge compares the current scene against.
        self.reference: dict[str, dict[str, Any]] = {}


class _JudgeDebugRecorder:
    """Best-effort per-query debug dump writer."""

    def __init__(self, *, enabled: bool, output_dir: str, provider: str) -> None:
        self.enabled = bool(enabled)
        self.output_dir = str(output_dir or "").strip()
        self.provider = provider

    def write(
        self,
        *,
        context: dict[str, Any],
        request_payload: dict[str, Any],
        response_payload: Any,
        decision: JudgeDecision | None = None,
        images: list[tuple[str, bytes, str | None]] | None = None,
        error: str | None = None,
        latency: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self.output_dir:
            return
        try:
            poll_idx = int(context.get("poll_index") or 0)
        except Exception:
            poll_idx = 0
        job_key = str(context.get("job_key") or "unknown_job")
        provider_dir = os.path.join(self.output_dir, self.provider)
        job_dir = os.path.join(provider_dir, _safe_name(job_key))
        os.makedirs(job_dir, exist_ok=True)
        stem = f"poll_{poll_idx:04d}" if poll_idx > 0 else f"poll_{int(time.time() * 1000)}"

        image_files: list[dict[str, Any]] = []
        for idx, (label, data, mime) in enumerate(images or [], start=1):
            ext = _mime_to_ext(mime)
            fname = f"{stem}_img_{idx:02d}_{_safe_name(label)}.{ext}"
            fpath = os.path.join(job_dir, fname)
            try:
                with open(fpath, "wb") as f:
                    f.write(data)
                image_files.append(
                    {"label": label, "mime": mime or "application/octet-stream", "path": fname}
                )
            except Exception:
                pass

        record = {
            "ts": int(time.time() * 1000),
            "provider": self.provider,
            "context": context,
            "request": request_payload,
            "response": response_payload,
            "decision": _decision_to_dict(decision) if decision is not None else None,
            "error": error,
            "latency": latency,
            "image_files": image_files,
        }
        with open(os.path.join(job_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)


class VLMJudgeClient(Protocol):
    async def judge(self, context: dict[str, Any]) -> JudgeDecision: ...
    async def aclose(self) -> None: ...


class MockVLMJudgeClient:
    def __init__(self, decisions: list[Any] | None = None) -> None:
        self.decisions = list(decisions or [{"status": "success"}])
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

    async def judge(self, context: dict[str, Any]) -> JudgeDecision:
        self.calls.append(dict(context))
        item = self.decisions[min(self._idx, len(self.decisions) - 1)]
        self._idx += 1
        return _coerce_decision(item)

    async def aclose(self) -> None:
        pass


class OpenRouterVLMJudgeClient:
    """OpenAI-compatible VLM Judge client.

    Each :meth:`judge` call:

    - GETs every camera's ``snapshot_url`` in parallel (via
      :func:`carryon.providers.fetch_image_url_part`) and base64-encodes the
      bodies into ``data:image/...`` URLs.
    - Builds a multimodal user message: a JSON dump of the task context, plus
      one ``image_url`` part per camera that was reachable this poll.
    - Sends it through :class:`carryon.providers.OpenRouterChatClient` with
      ``response_format={"type": "json_object"}`` so the model is forced to
      emit a strict JSON verdict (see ADR-0001 for why Judge uses strict mode
      but Planner does not).

    If every snapshot fetch fails we still send a text-only request so the
    judge can at least decide ``running`` / ``failed`` from context alone.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 30.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        cameras: list[dict[str, Any]] | None = None,
        default_headers: dict[str, str] | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        image_fetch_timeout_s: float = 5.0,
        history_window_frames: int = 3,
        debug_enabled: bool = False,
        debug_output_dir: str = "./logs/vlm_judge_debug",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("VLM judge: api_key is required for provider=openrouter")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.cameras = list(cameras or [])
        self.default_headers = dict(default_headers or {})
        self.http_referer = http_referer
        self.x_title = x_title
        self.image_fetch_timeout_s = image_fetch_timeout_s
        self.history_window_frames = max(1, int(history_window_frames))
        # Keep baseline and rolling frame history separately for each judge watch.
        self._scopes: OrderedDict[str, _Scope] = OrderedDict()
        self._http = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_http = client is None
        self._debug = _JudgeDebugRecorder(
            enabled=debug_enabled, output_dir=debug_output_dir, provider="openrouter"
        )
        self._chat = OpenRouterChatClient(
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            default_headers=default_headers,
            http_referer=http_referer,
            x_title=x_title,
            client=self._http,
        )

    def _latency_telemetry(self, context, *, image_fetch_ms, judge_api_ms,
                           judge_parse_ms, t_start):
        """Build + emit per-invocation wall-clock latency. Additive only: it does
        not touch the decision, prompt, poll interval, or thresholds. Checkpoint
        pause/resume fields are N/A until closed-loop is enabled."""
        def _na(v):
            return v if v is not None else "N/A"
        telem = {
            "request_id": str(context.get("job_key") or context.get("watch_id") or "unknown"),
            "model": getattr(self, "model", None),
            "poll_index": context.get("poll_index"),
            "image_fetch_ms": _na(image_fetch_ms),
            "judge_api_ms": _na(judge_api_ms),
            "judge_parse_ms": _na(judge_parse_ms),
            "judge_total_ms": round((time.monotonic() - t_start) * 1000.0, 1),
            # closed-loop checkpoint pause/resume not enabled on realbot -> N/A.
            "pause_ms": "N/A",
            "resume_ms": "N/A",
            "first_resumed_action_ms": "N/A",
        }
        sys.stderr.write(
            f"[judge latency] req={telem['request_id']} model={telem['model']} "
            f"poll={telem['poll_index']} image_fetch_ms={telem['image_fetch_ms']} "
            f"judge_api_ms={telem['judge_api_ms']} judge_parse_ms={telem['judge_parse_ms']} "
            f"judge_total_ms={telem['judge_total_ms']}\n"
        )
        return telem

    async def judge(self, context: dict[str, Any]) -> JudgeDecision:
        t_start = time.monotonic()
        image_fetch_ms = None
        judge_api_ms = None
        judge_parse_ms = None
        # The supervisor may scope the camera set per job (endpoint-declared
        # `cameras:` list, see supervisor.cameras_for_endpoint); when the
        # context carries such a list, judge only from those views. Fall back
        # to the client's full configured set otherwise (legacy behaviour).
        ctx_cams = context.get("cameras")
        cameras = (
            [c for c in ctx_cams if isinstance(c, dict)]
            if isinstance(ctx_cams, list) and ctx_cams
            else self.cameras
        ) or self.cameras
        cams_with_url = [c for c in cameras if isinstance(c, dict) and c.get("snapshot_url")]
        camera_attempted = bool(cams_with_url)
        scope = self._scope_for(context)
        fetched: list[tuple[dict, list[dict[str, Any]]]] = []
        if cams_with_url:
            _t_fetch = time.monotonic()
            parts = await asyncio.gather(
                *(
                    fetch_image_url_part(self._http, c, timeout_s=self.image_fetch_timeout_s)
                    for c in cams_with_url
                ),
                return_exceptions=False,
            )
            image_fetch_ms = round((time.monotonic() - _t_fetch) * 1000.0, 1)
            for camera, part in zip(cams_with_url, parts):
                if part is None:
                    continue
                key = _camera_history_key(camera)
                scope.reference.setdefault(key, part)
                history = scope.history[key]
                history.append(part)
                fetched.append((camera, list(history)))

        references: list[tuple[dict, dict[str, Any]]] = []
        for camera, _ in fetched:
            ref = scope.reference.get(_camera_history_key(camera))
            if ref is not None:
                references.append((camera, ref))

        messages = [
            {"role": "system", "content": self._system_text(cameras)},
            {
                "role": "user",
                "content": self._user_content(
                    context, fetched, references, camera_attempted=camera_attempted
                ),
            },
        ]

        images_for_debug: list[tuple[str, bytes, str | None]] = []
        for camera, ref_part in references:
            label = str(camera.get("name") or "camera")
            blob, mime = _decode_image_url_part(ref_part)
            if blob is not None:
                images_for_debug.append((f"{label}_baseline", blob, mime))
        for camera, history_parts in fetched:
            label = str(camera.get("name") or "camera")
            for idx, part in enumerate(history_parts, start=1):
                blob, mime = _decode_image_url_part(part)
                if blob is None:
                    continue
                images_for_debug.append((f"{label}_h{idx}", blob, mime))

        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        _t_api = time.monotonic()
        try:
            result = await self._chat.chat(
                model=request_payload["model"],
                messages=request_payload["messages"],
                temperature=request_payload["temperature"],
                max_tokens=request_payload["max_tokens"],
                response_format=request_payload["response_format"],
            )
        except httpx.HTTPError as e:
            judge_api_ms = round((time.monotonic() - _t_api) * 1000.0, 1)
            _telem = self._latency_telemetry(
                context, image_fetch_ms=image_fetch_ms, judge_api_ms=judge_api_ms,
                judge_parse_ms=None, t_start=t_start)
            self._debug.write(
                context=context,
                request_payload=request_payload,
                response_payload=None,
                images=images_for_debug,
                error=f"VLM judge http error: {type(e).__name__}: {e}",
                latency=_telem,
            )
            # Transport failures are inconclusive; keep waiting until the job deadline.
            return JudgeDecision(
                status="running",
                error=f"VLM judge http error: {type(e).__name__}: {e}",
                api_error=True,
            )
        except OpenRouterHTTPError as e:
            err = f"vlm_judge: {e}"
            judge_api_ms = round((time.monotonic() - _t_api) * 1000.0, 1)
            _telem = self._latency_telemetry(
                context, image_fetch_ms=image_fetch_ms, judge_api_ms=judge_api_ms,
                judge_parse_ms=None, t_start=t_start)
            self._debug.write(
                context=context,
                request_payload=request_payload,
                response_payload=None,
                images=images_for_debug,
                error=err,
                latency=_telem,
            )
            # Same reasoning as the httpx branch above: a 403/429/5xx from the
            # gateway is our problem, not the robot's. The 403 that triggered
            # this fix reported "task failed" ~2s into every episode.
            return JudgeDecision(status="running", error=err, api_error=True)

        judge_api_ms = round((time.monotonic() - _t_api) * 1000.0, 1)
        _t_parse = time.monotonic()
        decision = _parse_decision_text(result.text or "")
        judge_parse_ms = round((time.monotonic() - _t_parse) * 1000.0, 1)
        _telem = self._latency_telemetry(
            context, image_fetch_ms=image_fetch_ms, judge_api_ms=judge_api_ms,
            judge_parse_ms=judge_parse_ms, t_start=t_start)
        self._debug.write(
            context=context,
            request_payload=request_payload,
            response_payload={"text": result.text},
            decision=decision,
            images=images_for_debug,
            latency=_telem,
        )
        return decision

    def _scope_for(self, context: dict[str, Any]) -> _Scope:
        """Return frame memory for the current judge watch.

        Use watch_id so retries start with independent baseline frames. If unavailable,
        fall back to job_key, then to the shared client scope."""
        key = str(context.get("watch_id") or context.get("job_key") or "")
        scope = self._scopes.get(key)
        if scope is None:
            scope = _Scope(self.history_window_frames)
            self._scopes[key] = scope
            while len(self._scopes) > _MAX_TRACKED_SCOPES:
                self._scopes.popitem(last=False)
        else:
            self._scopes.move_to_end(key)
        return scope

    def release_scope(self, watch_id: str) -> None:
        """Drop a finished watch's frame memory.

        Called by the supervisor when a watch ends. Idempotent, and safe to
        skip -- _MAX_TRACKED_SCOPES is the backstop for watches that never get
        here.
        """
        self._scopes.pop(str(watch_id or ""), None)

    async def aclose(self) -> None:
        await self._chat.aclose()
        if self._owns_http:
            await self._http.aclose()

    def _system_text(self, cameras: list[dict[str, Any]] | None = None) -> str:
        meta = [
            {
                "name": c.get("name"),
                "type": c.get("type"),
                "description": c.get("description"),
            }
            for c in (self.cameras if cameras is None else cameras)
            if isinstance(c, dict)
        ]
        chunks = [self.system_prompt]
        if meta:
            chunks.append(
                "Camera inputs (metadata only; actual frames are attached as "
                "image_url parts in the user message):\n"
                + json.dumps(meta, ensure_ascii=False)
            )
            chunks.append(
                f"History window per camera: up to {self.history_window_frames} "
                "recent frames, ordered oldest to newest."
            )
        chunks.append(_RESPONSE_SCHEMA_HINT)
        return "\n\n".join(chunks)

    def _user_content(
        self,
        context: dict[str, Any],
        fetched: list[tuple[dict, list[dict[str, Any]]]],
        references: list[tuple[dict, dict[str, Any]]] | None = None,
        *,
        camera_attempted: bool,
    ) -> list[dict]:
        content: list[dict] = [
            {
                "type": "text",
                "text": "Current task context:\n"
                + json.dumps(context, ensure_ascii=False, default=str),
            }
        ]
        for camera, ref_part in references or []:
            label = camera.get("name") or "camera"
            content.append(
                {
                    "type": "text",
                    "text": f"[{label}] BASELINE frame (the scene when THIS "
                    "subtask's watch started; anything already in place here was "
                    "done by an earlier subtask and is not your success. Judge "
                    "the difference between this and the CURRENT frame):",
                }
            )
            content.append(ref_part)
        any_image = False
        for camera, history_parts in fetched:
            if not history_parts:
                continue
            any_image = True
            label = camera.get("name") or "camera"
            total = len(history_parts)
            for idx, part in enumerate(history_parts, start=1):
                content.append(
                    {
                        "type": "text",
                        "text": _window_label(label, idx, total),
                    }
                )
                content.append(part)
        if camera_attempted and not any_image:
            content.append(
                {
                    "type": "text",
                    "text": "(no camera snapshots available this poll)",
                }
            )
        return content


def build_vlm_judge_client(config: VLMJudgeConfig) -> VLMJudgeClient:
    if config.provider == "mock":
        return MockVLMJudgeClient(config.mock_decisions)
    if config.provider == "openrouter":
        if not config.api_key:
            raise ValueError(
                "VLM judge: api_key is required for provider=openrouter"
            )
        return OpenRouterVLMJudgeClient(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            system_prompt=config.system_prompt,
            cameras=config.cameras,
            default_headers=config.default_headers,
            http_referer=config.http_referer,
            x_title=config.x_title,
            image_fetch_timeout_s=config.image_fetch_timeout_s,
            history_window_frames=config.history_window_frames,
            debug_enabled=config.debug_enabled,
            debug_output_dir=config.debug_output_dir,
        )
    raise ValueError(f"unsupported VLM judge provider: {config.provider!r}")


def _coerce_decision(value: Any) -> JudgeDecision:
    if isinstance(value, str):
        status = value.lower().strip()
        return JudgeDecision(status=_normalize_status(status), raw=value)
    if not isinstance(value, dict):
        # Unreadable shape -> inconclusive, not a robot failure (see
        # _parse_decision_text).
        return JudgeDecision(status="running", error=f"invalid judge decision: {value!r}",
                             raw=value, parse_error=True)
    status = _normalize_status(str(value.get("status") or "running").lower().strip())
    return JudgeDecision(
        status=status,
        result=value.get("result"),
        error=value.get("error") or value.get("reason"),
        raw=value,
    )


def _normalize_status(status: str) -> str:
    if status in TERMINAL_STATUSES or status in RUNNING_STATUSES:
        return "running" if status in RUNNING_STATUSES else status
    return "failed"


def _validate_decision_schema(data: dict[str, Any]) -> str | None:
    status = data.get("status")
    if not isinstance(status, str):
        return f"VLM judge response has invalid status: {status!r}"
    normalized_status = status.lower().strip()
    if (
        _normalize_status(normalized_status) == "failed"
        and normalized_status not in TERMINAL_STATUSES
    ):
        return f"VLM judge response has invalid status: {status!r}"

    result = data.get("result")
    if not isinstance(result, dict):
        return "VLM judge response is missing result object."

    required_result_fields = (
        "planner_action",
        "confidence",
        "visual_evidence",
        "stuck_evidence",
    )
    missing = [field for field in required_result_fields if field not in result]
    if missing:
        return f"VLM judge response result is missing required field(s): {missing}"

    if "error" not in data:
        return "VLM judge response is missing top-level error field."

    return None


def _extract_decision(data: Any) -> Any:
    """Best-effort: pull a verdict dict out of the parsed JSON when the
    model wrapped it instead of emitting the verdict directly.
    """
    if isinstance(data, dict):
        if "status" in data:
            return data
        if isinstance(data.get("decision"), dict):
            return data["decision"]
        text = data.get("text")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return data
    return data


def _parse_decision_text(raw_text: str) -> JudgeDecision:
    """Treat a non-JSON judge response as inconclusive until the deadline."""
    if not raw_text.strip():
        return JudgeDecision(
            status="running",
            error="VLM judge returned an empty response.",
            raw={"text": raw_text},
            parse_error=True,
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return JudgeDecision(
            status="running",
            error=f"VLM judge returned invalid JSON: {e}",
            raw={"text": raw_text},
            parse_error=True,
        )

    decision_payload = _extract_decision(payload)
    if not isinstance(decision_payload, dict) or "status" not in decision_payload:
        return JudgeDecision(
            status="running",
            error="VLM judge response did not contain a decision status.",
            raw=payload,
            parse_error=True,
        )
    schema_error = _validate_decision_schema(decision_payload)
    if schema_error:
        return JudgeDecision(status="running", error=schema_error, raw=payload,
                             parse_error=True)
    return _coerce_decision(decision_payload)


def _camera_history_key(camera: dict[str, Any]) -> str:
    return str(
        camera.get("name")
        or camera.get("snapshot_url")
        or camera.get("description")
        or id(camera)
    )


def _window_label(camera_name: str, idx: int, total: int) -> str:
    if idx == total:
        role = "CURRENT"
    elif idx == 1:
        role = "OLDEST"
    else:
        role = "HISTORY"
    return f"Camera {camera_name} frame {idx}/{total} ({role}):"


def _safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return out.strip("_") or "x"


def _mime_to_ext(mime: str | None) -> str:
    if not mime:
        return "bin"
    m = mime.lower().split(";", 1)[0].strip()
    return {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(m, "bin")


def _decode_image_url_part(part: dict[str, Any]) -> tuple[bytes | None, str | None]:
    try:
        url = str(((part or {}).get("image_url") or {}).get("url") or "")
        if not url.startswith("data:") or ";base64," not in url:
            return None, None
        header, b64 = url.split(",", 1)
        mime = header[5:].split(";", 1)[0] if header.startswith("data:") else None
        return base64.b64decode(b64), mime
    except Exception:
        return None, None


def _decision_to_dict(decision: JudgeDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "result": decision.result,
        "error": decision.error,
        "raw": decision.raw,
        "terminal": decision.terminal,
    }
