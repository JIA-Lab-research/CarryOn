"""Unified readable event log used across the whole project.

Two concerns, one implementation:

* **File output** — every call appends one pretty-printed JSON record
  (``indent=2``) to a configured ``path``, followed by a ``---`` separator
  line. Long strings inside ``system`` / ``messages[*].content`` / ``text``
  / ``instruction`` have their escaped ``\\n`` expanded back to real
  newlines so multi-thousand-character prompts are readable by eye.

  One record per dump. Example::

      {
        "ts": 1714200000.1,
        "channel": "planner",
        "verb": "planner_request",
        "provider": "openrouter",
        "model": "openai/gpt-5.2",
        "system": "You are CarryOn...
      The registry contains:
        - arm_sim (mock)",
        "messages": [
          { "role": "user", "content": "pick the red cube" }
        ]
      }
      ---

  Because real newlines break strict JSON Lines, the file extension
  defaults to ``.log`` (not ``.jsonl``). To post-process, split on the
  ``---`` separator first, then ``json.loads`` each chunk (after escaping
  the newlines inside strings back to ``\\n`` if you really need strict
  JSON).

* **Stdout summary** — when ``stdout=True`` a compact one-liner of the
  form ``[<channel> <verb>] ...`` is still printed so you can see live
  what the agent is doing without tailing the file.

Failures in this module never propagate -- tracing must never kill the
agent.

Two channels are exposed via convenience methods because their shape is
stable and used in many places:

* ``planner_*``   -- Planner <-> Planner VLM traffic
    ``planner_request`` / ``planner_response`` / ``planner_error``
* ``transport_*`` -- Agent <-> RobotEndpoint traffic (HTTP / Mock / ROS)
    ``transport_submit_request`` / ``transport_submit_response`` /
    ``transport_submit_error`` / ``transport_cancel`` /
    ``transport_callback``

Every method returns quickly. Request/response pairs use a lightweight
``_Pending`` token so latency can be measured without the caller tracking
start times.
"""
from __future__ import annotations
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
_logger = logging.getLogger('carryon.events')
_SEPARATOR = '---'
_REPL_PROMPT: str = ''

def set_repl_prompt(prompt: str) -> None:
    """Register the current REPL prompt so async stdout writers can redraw it."""
    global _REPL_PROMPT
    _REPL_PROMPT = prompt or ''

def get_repl_prompt() -> str:
    return _REPL_PROMPT

def _format_ts(t: float) -> str:
    """Render a Unix timestamp as local-time ``dd-MM-yyyy HH:mm:ss.fff``.

    Example: ``27-04-2026 20:45:03.172``.

    We use local time on purpose -- these logs are intended for reading
    at a REPL session, not for cross-region correlation. Millisecond
    precision is enough to see which events happened in the same tick.
    """
    lt = time.localtime(t)
    ms = int((t - int(t)) * 1000)
    return f"{time.strftime('%d-%m-%Y %H:%M:%S', lt)}.{ms:03d}"

@dataclass
class _Pending:
    started_ts: float
    channel: str
    tag: str

class EventLogger:
    """Shared JSONL + stdout logger. Create one per channel (file)."""

    def __init__(self, path: str | None, stdout: bool=True, *, channel: str='event', default_fields: dict[str, Any] | None=None) -> None:
        self.path = path or ''
        self.stdout = stdout
        self.channel = channel
        self._defaults = dict(default_fields or {})
        self._ensure_dir()

    def emit(self, verb: str, **fields: Any) -> None:
        """Write one JSONL record with ``{ts, channel, verb, ...fields}``.

        Low-level; planner and transport convenience methods are thin wrappers on top.
        """
        rec: dict[str, Any] = {'ts': _format_ts(time.time()), 'channel': self.channel, 'verb': verb}
        rec.update(self._defaults)
        rec.update(fields)
        self._write(rec)

    def print_summary(self, verb: str, tag: str, detail: str) -> None:
        """Print one stdout line in the canonical ``[channel verb] ...`` form."""
        if not self.stdout:
            return
        line = f'[{self.channel} {verb}] {tag} {detail}'
        prompt = _REPL_PROMPT
        try:
            if prompt and sys.stdout.isatty():
                sys.stdout.write('\r\x1b[2K' + line + '\n' + prompt)
            else:
                sys.stdout.write(line + '\n')
            sys.stdout.flush()
        except Exception:
            pass

    def planner_request(self, system: str | None, messages: list[dict]) -> _Pending:
        provider = self._defaults.get('provider', 'unknown')
        model = self._defaults.get('model', 'unknown')
        self.emit('planner_request', system=system, messages=messages)
        n_msgs = len(messages)
        n_chars = sum((len(str(m.get('content', ''))) for m in messages))
        self.print_summary('request', f'{provider}/{model}', f'msgs={n_msgs} chars={n_chars}')
        return _Pending(started_ts=time.time(), channel=self.channel, tag=f'{provider}/{model}')

    def planner_response(self, pending: _Pending, text: str, usage: dict[str, Any] | None) -> None:
        latency_ms = int((time.time() - pending.started_ts) * 1000)
        usage = usage or {}
        self.emit('planner_response', latency_ms=latency_ms, text=text, usage=usage)
        in_tok = usage.get('input_tokens') or usage.get('prompt_tokens')
        out_tok = usage.get('output_tokens') or usage.get('completion_tokens')
        tok_part = f'in={in_tok}tok out={out_tok}tok ' if in_tok is not None or out_tok is not None else ''
        self.print_summary('response', pending.tag, f'{tok_part}latency={latency_ms}ms chars={len(text)}')

    def planner_error(self, pending: _Pending | None, err: BaseException) -> None:
        fields: dict[str, Any] = {'error': f'{type(err).__name__}: {err}'}
        if pending is not None:
            fields['latency_ms'] = int((time.time() - pending.started_ts) * 1000)
        self.emit('planner_error', **fields)
        tag = pending.tag if pending is not None else 'unknown'
        self.print_summary('error', tag, f'{type(err).__name__}: {err}')

    def transport_submit_request(self, *, transport: str, endpoint: str, address: str, instruction: str, inputs: dict | None, meta: dict | None) -> _Pending:
        meta = meta or {}
        self.emit('transport_submit_request', transport=transport, endpoint=endpoint, address=address, instruction=instruction, inputs=inputs or {}, request_id=meta.get('request_id'), callback_url=meta.get('callback_url'), deadline_s=meta.get('deadline_s'))
        rid = meta.get('request_id') or '-'
        full = (instruction or '').replace('\n', ' ')
        preview = full if len(full) <= 200 else full[:200] + '...'
        self.print_summary('submit', f'{transport}/{endpoint}', f'req={rid} instr="{preview}"')
        return _Pending(started_ts=time.time(), channel=self.channel, tag=f'{transport}/{endpoint}')

    def transport_submit_response(self, pending: _Pending, *, accepted: bool, job_id: str | None, reason: str | None=None, raw: Any=None) -> None:
        latency_ms = int((time.time() - pending.started_ts) * 1000)
        self.emit('transport_submit_response', accepted=accepted, job_id=job_id, reason=reason, latency_ms=latency_ms, raw=raw)
        tag_verb = 'accept' if accepted else 'reject'
        extra = f' reason={reason}' if reason else ''
        self.print_summary(tag_verb, pending.tag, f"job={job_id or '-'} latency={latency_ms}ms{extra}")

    def transport_submit_error(self, pending: _Pending, err: BaseException) -> None:
        latency_ms = int((time.time() - pending.started_ts) * 1000)
        self.emit('transport_submit_error', latency_ms=latency_ms, error=f'{type(err).__name__}: {err}')
        self.print_summary('error', pending.tag, f'{type(err).__name__}: {err}')

    def transport_cancel(self, *, transport: str, endpoint: str, request_id: str, ok: bool, latency_ms: int | None=None) -> None:
        fields: dict[str, Any] = {'transport': transport, 'endpoint': endpoint, 'request_id': request_id, 'ok': ok}
        if latency_ms is not None:
            fields['latency_ms'] = latency_ms
        self.emit('transport_cancel', **fields)
        self.print_summary('cancel', f'{transport}/{endpoint}', f'req={request_id} ok={ok}')

    def transport_callback(self, *, job_key: str, payload: dict) -> None:
        status = str((payload or {}).get('status') or 'unknown')
        meta = (payload or {}).get('meta') or {}
        err = (payload or {}).get('error')
        self.emit('transport_callback', job_key=job_key, status=status, error=err, latency_ms=meta.get('latency_ms') if isinstance(meta, dict) else None, payload=payload or {})
        err_part = f' err={err}' if err else ''
        self.print_summary('callback', f'job={job_key}', f'status={status}{err_part}')

    def _ensure_dir(self) -> None:
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        if d and (not os.path.exists(d)):
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                self.path = ''

    def _write(self, rec: dict) -> None:
        if not self.path:
            return
        try:
            line = json.dumps(rec, ensure_ascii=False, default=_safe_default, indent=2)
        except Exception as e:
            _logger.warning('event json serialize failed: %s', e)
            return
        line = _unescape_newlines_in_json(line)
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n' + _SEPARATOR + '\n')
        except OSError as e:
            _logger.warning('event log write failed (%s): %s', self.path, e)
_JSON_STRING_RE = re.compile('"((?:\\\\.|[^"\\\\])*)"', flags=re.DOTALL)

def _unescape_newlines_in_json(text: str) -> str:

    def _fix(match: re.Match) -> str:
        body = match.group(1)
        if '\\n' not in body and '\\t' not in body:
            return match.group(0)
        body = body.replace('\\n', '\n').replace('\\t', '\t')
        return f'"{body}"'
    return _JSON_STRING_RE.sub(_fix, text)

def _safe_default(obj: Any) -> Any:
    if hasattr(obj, 'to_dict'):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, '__dict__'):
        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
    return repr(obj)

def read_event_log(path: str) -> list[dict]:
    """Parse a pretty-printed event log produced by ``EventLogger``.

    The file looks like::

        { ...record A... }
        ---
        { ...record B... }
        ---

    Where each record is pretty JSON with *real* newlines inside long string
    values (for readability). We split on the ``---`` separator and, for
    each chunk, re-escape newlines inside string literals before running
    ``json.loads``.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    out: list[dict] = []
    for chunk in text.split('\n' + _SEPARATOR + '\n'):
        chunk = chunk.strip()
        if not chunk:
            continue
        fixed = _reescape_newlines_in_json(chunk)
        out.append(json.loads(fixed))
    return out

def _reescape_newlines_in_json(text: str) -> str:
    """Inverse of ``_unescape_newlines_in_json``: put ``\\n`` / ``\\t`` back
    inside string literals so the text is strict JSON again.
    """

    def _fix(match: re.Match) -> str:
        body = match.group(1)
        if '\n' not in body and '\t' not in body:
            return match.group(0)
        body = body.replace('\n', '\\n').replace('\t', '\\t')
        return f'"{body}"'
    return _JSON_STRING_RE.sub(_fix, text)
