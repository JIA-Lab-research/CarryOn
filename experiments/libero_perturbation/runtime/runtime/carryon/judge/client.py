from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol
import httpx
from ..providers import OPENROUTER_BASE_URL, OpenRouterChatClient, OpenRouterHTTPError, fetch_image_url_part_with_metadata
from ..types import CALLBACK_TERMINAL_STATUSES, STATUS_EVENT_KEYWORDS
from .config import DEFAULT_SYSTEM_PROMPT, VLMJudgeConfig
TERMINAL_STATUSES = CALLBACK_TERMINAL_STATUSES
RUNNING_STATUSES = STATUS_EVENT_KEYWORDS | {'running'}
_RESPONSE_SCHEMA_HINT = 'Reply with STRICT JSON ONLY (no markdown fence, no prose) matching:\n{"status":"running|success|failed|cancelled","result":{"planner_action":"continue_observing|advance_to_next_task|invoke_correction","confidence":<number 0..1>,"visual_evidence":<string>,"stuck_evidence":<string|null>,"recovery_object":<string|null>,"recovery_point":<[x,y]|null>,"target_released":<boolean>,"target_in_destination":<boolean>,"placement_stable":<boolean>},"error":<string|null>}'

@dataclass
class JudgeDecision:
    status: str
    result: Any = None
    error: str | None = None
    raw: Any = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

class _JudgeDebugRecorder:
    """Best-effort per-query debug dump writer."""

    def __init__(self, *, enabled: bool, output_dir: str, provider: str) -> None:
        self.enabled = bool(enabled)
        self.output_dir = str(output_dir or '').strip()
        self.provider = provider

    def write(self, *, context: dict[str, Any], request_payload: dict[str, Any], response_payload: Any, decision: JudgeDecision | None=None, images: list[tuple[str, bytes, str | None]] | None=None, error: str | None=None) -> None:
        if not self.enabled or not self.output_dir:
            return
        try:
            poll_idx = int(context.get('poll_index') or 0)
        except Exception:
            poll_idx = 0
        job_key = str(context.get('job_key') or 'unknown_job')
        provider_dir = os.path.join(self.output_dir, self.provider)
        job_dir = os.path.join(provider_dir, _safe_name(job_key))
        os.makedirs(job_dir, exist_ok=True)
        stem = f'poll_{poll_idx:04d}' if poll_idx > 0 else f'poll_{int(time.time() * 1000)}'
        image_files: list[dict[str, Any]] = []
        for idx, (label, data, mime) in enumerate(images or [], start=1):
            ext = _mime_to_ext(mime)
            fname = f'{stem}_img_{idx:02d}_{_safe_name(label)}.{ext}'
            fpath = os.path.join(job_dir, fname)
            try:
                with open(fpath, 'wb') as f:
                    f.write(data)
                image_files.append({'label': label, 'mime': mime or 'application/octet-stream', 'path': fname})
            except Exception:
                pass
        record = {'ts': int(time.time() * 1000), 'provider': self.provider, 'context': context, 'request': request_payload, 'response': response_payload, 'decision': _decision_to_dict(decision) if decision is not None else None, 'error': error, 'image_files': image_files}
        with open(os.path.join(job_dir, f'{stem}.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

    def write_telemetry(self, *, context: dict[str, Any], telemetry: dict[str, Any]) -> None:
        """Atomically persist the completed wall-clock ledger for one poll."""
        if not self.enabled or not self.output_dir:
            return
        poll_idx = int(context.get('poll_index') or 0)
        job_key = str(context.get('job_key') or 'unknown_job')
        job_dir = os.path.join(self.output_dir, self.provider, _safe_name(job_key))
        os.makedirs(job_dir, exist_ok=True)
        stem = f'poll_{poll_idx:04d}'
        target = os.path.join(job_dir, f'{stem}_wall_clock.json')
        temp = target + '.tmp'
        with open(temp, 'w', encoding='utf-8') as f:
            json.dump(telemetry, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(temp, target)

class VLMJudgeClient(Protocol):

    async def judge(self, context: dict[str, Any]) -> JudgeDecision:
        ...

    async def aclose(self) -> None:
        ...

class MockVLMJudgeClient:

    def __init__(self, decisions: list[Any] | None=None) -> None:
        self.decisions = list(decisions or [{'status': 'success'}])
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

    def __init__(self, *, api_key: str, model: str, base_url: str=OPENROUTER_BASE_URL, temperature: float=0.0, max_tokens: int=512, timeout_s: float=30.0, system_prompt: str=DEFAULT_SYSTEM_PROMPT, cameras: list[dict[str, Any]] | None=None, default_headers: dict[str, str] | None=None, http_referer: str | None=None, x_title: str | None=None, image_fetch_timeout_s: float=5.0, history_window_frames: int=3, debug_enabled: bool=False, debug_output_dir: str='./logs/vlm_judge_debug', client: httpx.AsyncClient | None=None) -> None:
        if not api_key:
            raise ValueError('VLM judge: api_key is required for provider=openrouter')
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
        self._image_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.history_window_frames))
        self._reference_frame: dict[str, dict[str, Any]] = {}
        self._http = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_http = client is None
        self._debug = _JudgeDebugRecorder(enabled=debug_enabled, output_dir=debug_output_dir, provider='openrouter')
        self._chat = OpenRouterChatClient(api_key=api_key, base_url=base_url, timeout_s=timeout_s, default_headers=default_headers, http_referer=http_referer, x_title=x_title, client=self._http)

    async def judge(self, context: dict[str, Any]) -> JudgeDecision:
        image_fetch_start_wall_ts = time.time()
        image_fetch_t0 = time.perf_counter()
        captured_at_s = time.monotonic()
        cams_with_url = [c for c in self.cameras if isinstance(c, dict) and c.get('snapshot_url')]
        camera_attempted = bool(cams_with_url)
        fetched: list[tuple[dict, list[dict[str, Any]]]] = []
        producer_metadata: dict[str, dict[str, Any]] = {}
        if cams_with_url:
            parts = await asyncio.gather(*(fetch_image_url_part_with_metadata(self._http, c, timeout_s=self.image_fetch_timeout_s) for c in cams_with_url), return_exceptions=False)
            for camera, part in zip(cams_with_url, parts):
                if part is None:
                    continue
                part, metadata = part
                key = _camera_history_key(camera)
                producer_metadata[key] = metadata
                self._reference_frame.setdefault(key, part)
                history = self._image_history[key]
                history.append(part)
                fetched.append((camera, list(history)))
        captured_at_s = time.monotonic()
        image_fetch_end_wall_ts = time.time()
        image_fetch_ms = (time.perf_counter() - image_fetch_t0) * 1000.0
        references: list[tuple[dict, dict[str, Any]]] = []
        for camera, _ in fetched:
            ref = self._reference_frame.get(_camera_history_key(camera))
            if ref is not None:
                references.append((camera, ref))
        messages = [{'role': 'system', 'content': self._system_text()}, {'role': 'user', 'content': self._user_content(context, fetched, references, camera_attempted=camera_attempted)}]
        images_for_debug: list[tuple[str, bytes, str | None]] = []
        for camera, ref_part in references:
            label = str(camera.get('name') or 'camera')
            blob, mime = _decode_image_url_part(ref_part)
            if blob is not None:
                images_for_debug.append((f'{label}_reference', blob, mime))
        for camera, history_parts in fetched:
            label = str(camera.get('name') or 'camera')
            for idx, part in enumerate(history_parts, start=1):
                blob, mime = _decode_image_url_part(part)
                if blob is None:
                    continue
                images_for_debug.append((f'{label}_h{idx}', blob, mime))
        request_payload = {'model': self.model, 'messages': messages, 'temperature': self.temperature, 'max_tokens': self.max_tokens, 'response_format': {'type': 'json_object'}}
        judge_api_request_wall_ts = time.time()
        judge_api_t0 = time.perf_counter()
        try:
            result = await self._chat.chat(model=request_payload['model'], messages=request_payload['messages'], temperature=request_payload['temperature'], max_tokens=request_payload['max_tokens'], response_format=request_payload['response_format'])
        except httpx.HTTPError as e:
            self._debug.write(context=context, request_payload=request_payload, response_payload=None, images=images_for_debug, error=f'VLM judge http error: {type(e).__name__}: {e}')
            return JudgeDecision(status='running', error=f'VLM judge http error (inconclusive): {type(e).__name__}: {e}')
        except OpenRouterHTTPError as e:
            self._debug.write(context=context, request_payload=request_payload, response_payload=None, images=images_for_debug, error=str(e))
            return JudgeDecision(status='running', error=f'judge API error (inconclusive): {e}')
        judge_api_response_wall_ts = time.time()
        judge_api_ms = (time.perf_counter() - judge_api_t0) * 1000.0
        parse_t0 = time.perf_counter()
        decision = _parse_decision_text(result.text or '')
        parse_ms = (time.perf_counter() - parse_t0) * 1000.0
        frame_digests: list[str] = []
        camera_observations: list[dict[str, Any]] = []
        for camera, history_parts in fetched:
            if not history_parts:
                continue
            blob, _mime = _decode_image_url_part(history_parts[-1])
            if blob is None:
                continue
            label = _camera_history_key(camera)
            pixel_sha256 = hashlib.sha256(blob).hexdigest()
            frame_digests.append(f'{label}:{pixel_sha256}')
            camera_observation = dict(producer_metadata.get(label) or {})
            camera_observation.update({'camera_name': str(camera.get('name') or label), 'pixel_sha256': pixel_sha256})
            camera_observations.append(camera_observation)
        frame_identity = ''
        if frame_digests:
            joined = '\n'.join(sorted(frame_digests)).encode('utf-8')
            frame_identity = hashlib.sha256(joined).hexdigest()
        if not isinstance(decision.result, dict):
            decision.result = {}
        observation = {'frame_identity': frame_identity, 'captured_at_s': captured_at_s, 'camera_count': len(frame_digests), 'camera_observations': camera_observations}
        if len(camera_observations) == 1:
            observation.update(camera_observations[0])
        decision.result['_observation'] = observation
        decision.result['_wall_clock_telemetry'] = {'image_fetch_start_wall_ts': image_fetch_start_wall_ts, 'image_fetch_end_wall_ts': image_fetch_end_wall_ts, 'image_fetch_ms': image_fetch_ms, 'judge_api_request_wall_ts': judge_api_request_wall_ts, 'judge_api_response_wall_ts': judge_api_response_wall_ts, 'judge_api_ms': judge_api_ms, 'parse_ms': parse_ms}
        self._debug.write(context=context, request_payload=request_payload, response_payload={'text': result.text}, decision=decision, images=images_for_debug)
        return decision

    def record_poll_telemetry(self, context: dict[str, Any], telemetry: dict[str, Any]) -> None:
        self._debug.write_telemetry(context=context, telemetry=telemetry)

    async def aclose(self) -> None:
        await self._chat.aclose()
        if self._owns_http:
            await self._http.aclose()

    def _system_text(self) -> str:
        meta = [{'name': c.get('name'), 'type': c.get('type'), 'description': c.get('description')} for c in self.cameras if isinstance(c, dict)]
        chunks = [self.system_prompt]
        if meta:
            chunks.append('Camera inputs (metadata only; actual frames are attached as image_url parts in the user message):\n' + json.dumps(meta, ensure_ascii=False))
            chunks.append(f'History window per camera: up to {self.history_window_frames} recent frames, ordered oldest to newest.')
        chunks.append(_RESPONSE_SCHEMA_HINT)
        return '\n\n'.join(chunks)

    def _user_content(self, context: dict[str, Any], fetched: list[tuple[dict, list[dict[str, Any]]]], references: list[tuple[dict, dict[str, Any]]] | None=None, *, camera_attempted: bool) -> list[dict]:
        content: list[dict] = [{'type': 'text', 'text': 'Current task context:\n' + json.dumps(context, ensure_ascii=False, default=str)}]
        for camera, ref_part in references or []:
            label = camera.get('name') or 'camera'
            content.append({'type': 'text', 'text': f'[{label}] reference frame (episode start; compare the current scene against this to judge progress):'})
            content.append(ref_part)
        any_image = False
        for camera, history_parts in fetched:
            if not history_parts:
                continue
            any_image = True
            label = camera.get('name') or 'camera'
            total = len(history_parts)
            for idx, part in enumerate(history_parts, start=1):
                content.append({'type': 'text', 'text': _window_label(label, idx, total)})
                content.append(part)
        if camera_attempted and (not any_image):
            content.append({'type': 'text', 'text': '(no camera snapshots available this poll)'})
        return content

def build_vlm_judge_client(config: VLMJudgeConfig) -> VLMJudgeClient:
    if config.provider == 'mock':
        return MockVLMJudgeClient(config.mock_decisions)
    if config.provider == 'openrouter':
        if not config.api_key:
            raise ValueError('VLM judge: api_key is required for provider=openrouter')
        return OpenRouterVLMJudgeClient(api_key=config.api_key, model=config.model, base_url=config.base_url, temperature=config.temperature, max_tokens=config.max_tokens, timeout_s=config.timeout_s, system_prompt=config.system_prompt, cameras=config.cameras, default_headers=config.default_headers, http_referer=config.http_referer, x_title=config.x_title, image_fetch_timeout_s=config.image_fetch_timeout_s, history_window_frames=config.history_window_frames, debug_enabled=config.debug_enabled, debug_output_dir=config.debug_output_dir)
    raise ValueError(f'unsupported VLM judge provider: {config.provider!r}')

def _coerce_decision(value: Any) -> JudgeDecision:
    if isinstance(value, str):
        status = value.lower().strip()
        return JudgeDecision(status=_normalize_status(status), raw=value)
    if not isinstance(value, dict):
        return JudgeDecision(status='failed', error=f'invalid judge decision: {value!r}', raw=value)
    status = _normalize_status(str(value.get('status') or 'running').lower().strip())
    return JudgeDecision(status=status, result=value.get('result'), error=value.get('error') or value.get('reason'), raw=value)

def _normalize_status(status: str) -> str:
    if status in TERMINAL_STATUSES or status in RUNNING_STATUSES:
        return 'running' if status in RUNNING_STATUSES else status
    return 'failed'

def _validate_decision_schema(data: dict[str, Any]) -> str | None:
    status = data.get('status')
    if not isinstance(status, str):
        return f'VLM judge response has invalid status: {status!r}'
    normalized_status = status.lower().strip()
    if _normalize_status(normalized_status) == 'failed' and normalized_status not in TERMINAL_STATUSES:
        return f'VLM judge response has invalid status: {status!r}'
    result = data.get('result')
    if not isinstance(result, dict):
        return 'VLM judge response is missing result object.'
    required_result_fields = ('planner_action', 'confidence', 'visual_evidence', 'stuck_evidence')
    missing = [field for field in required_result_fields if field not in result]
    if missing:
        return f'VLM judge response result is missing required field(s): {missing}'
    if 'error' not in data:
        return 'VLM judge response is missing top-level error field.'
    return None

def _extract_decision(data: Any) -> Any:
    """Best-effort: pull a verdict dict out of the parsed JSON when the
    model wrapped it instead of emitting the verdict directly.
    """
    if isinstance(data, dict):
        if 'status' in data:
            return data
        if isinstance(data.get('decision'), dict):
            return data['decision']
        text = data.get('text')
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                return data
    return data

def _parse_decision_text(raw_text: str) -> JudgeDecision:
    if not raw_text.strip():
        return JudgeDecision(status='running', error='VLM judge returned an empty response.', raw={'text': raw_text})
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return JudgeDecision(status='running', error=f'VLM judge returned invalid JSON: {e}', raw={'text': raw_text})
    decision_payload = _extract_decision(payload)
    if not isinstance(decision_payload, dict) or 'status' not in decision_payload:
        return JudgeDecision(status='running', error='VLM judge response did not contain a decision status.', raw=payload)
    schema_error = _validate_decision_schema(decision_payload)
    if schema_error:
        return JudgeDecision(status='running', error=schema_error, raw=payload)
    return _coerce_decision(decision_payload)

def _camera_history_key(camera: dict[str, Any]) -> str:
    return str(camera.get('name') or camera.get('snapshot_url') or camera.get('description') or id(camera))

def _window_label(camera_name: str, idx: int, total: int) -> str:
    if idx == total:
        role = 'CURRENT'
    elif idx == 1:
        role = 'OLDEST'
    else:
        role = 'HISTORY'
    return f'Camera {camera_name} frame {idx}/{total} ({role}):'

def _safe_name(text: str) -> str:
    out = re.sub('[^A-Za-z0-9._-]+', '_', str(text or '').strip())
    return out.strip('_') or 'x'

def _mime_to_ext(mime: str | None) -> str:
    if not mime:
        return 'bin'
    m = mime.lower().split(';', 1)[0].strip()
    return {'image/jpeg': 'jpg', 'image/jpg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/gif': 'gif'}.get(m, 'bin')

def _decode_image_url_part(part: dict[str, Any]) -> tuple[bytes | None, str | None]:
    try:
        url = str(((part or {}).get('image_url') or {}).get('url') or '')
        if not url.startswith('data:') or ';base64,' not in url:
            return (None, None)
        header, b64 = url.split(',', 1)
        mime = header[5:].split(';', 1)[0] if header.startswith('data:') else None
        return (base64.b64decode(b64), mime)
    except Exception:
        return (None, None)

def _decision_to_dict(decision: JudgeDecision) -> dict[str, Any]:
    return {'status': decision.status, 'result': decision.result, 'error': decision.error, 'raw': decision.raw, 'terminal': decision.terminal}
