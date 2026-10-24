"""Collect jam evidence and request bounded grasp-height corrections."""
from __future__ import annotations
import asyncio
import fcntl
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol
from .semantic_memory import FORBIDDEN_DECISION_FIELDS
SIGNAL_FIELDS = frozenset({'object_key', 'planned_mm', 'moved_mm', 'dpos_mm', 'stuck_z_m', 'current_z_m'})
GRASP_NO_HOLD_FIELDS = frozenset({'object_key', 'current_z_m', 'close_gripper_succeeded', 'assert_holding_failed'})
RESPONSE_FIELDS = frozenset({'corrected_z_m', 'rationale'})
OPENROUTER_ENVELOPE = {'model': 'qwen/qwen3-vl-235b-a22b-instruct', 'temperature': 0.0, 'top_p': 0.8, 'max_tokens': 1400, 'provider': {'data_collection': 'deny'}}
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
MIN_Z_M = 0.0
MAX_Z_M = 0.25
MAX_ABS_Z_DELTA_M = 0.2
PERSISTENT_MEMORY_SCHEMA = 'carryon-fixv2-vlm-z-object-memory/v1'
PERSISTENT_MEMORY_FIELDS = frozenset({'schema', 'created_utc', 'object_key', 'corrected_z_m', 'prior_z_m', 'rationale', 'trigger', 'value_source'})

class JamZClient(Protocol):

    def complete(self, prompt: str, envelope: dict[str, Any]) -> str:
        ...

@dataclass(frozen=True)
class JamResidualSignal:
    object_key: str
    planned_mm: float
    moved_mm: float
    dpos_mm: float
    stuck_z_m: float
    current_z_m: float

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        if set(value) != SIGNAL_FIELDS:
            raise ValueError('jam signal schema drift')
        key = str(value['object_key']).strip().lower()
        if not key:
            raise ValueError('object_key is required')
        value['object_key'] = key
        for name in SIGNAL_FIELDS - {'object_key'}:
            number = value[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f'{name} must be numeric')
            number = float(number)
            if not math.isfinite(number):
                raise ValueError(f'{name} must be finite')
            value[name] = number
        if value['planned_mm'] < 0 or value['moved_mm'] < 0 or value['dpos_mm'] < 0:
            raise ValueError('millimetre residuals must be nonnegative')
        if not MIN_Z_M <= value['current_z_m'] <= MAX_Z_M:
            raise ValueError('current_z_m outside physical bounds')
        if not MIN_Z_M <= value['stuck_z_m'] <= MAX_Z_M + 0.2:
            raise ValueError('stuck_z_m outside physical bounds')
        if FORBIDDEN_DECISION_FIELDS.intersection(value):
            raise ValueError('forbidden decision field in jam signal')
        return value

@dataclass(frozen=True)
class GraspNoHoldSignal:
    object_key: str
    current_z_m: float
    close_gripper_succeeded: bool
    assert_holding_failed: bool

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        if set(value) != GRASP_NO_HOLD_FIELDS:
            raise ValueError('grasp-no-hold signal schema drift')
        key = str(value['object_key']).strip().lower()
        if not key:
            raise ValueError('object_key is required')
        value['object_key'] = key
        current = value['current_z_m']
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise ValueError('current_z_m must be numeric')
        current = float(current)
        if not math.isfinite(current) or not MIN_Z_M <= current <= MAX_Z_M:
            raise ValueError('current_z_m outside physical bounds')
        value['current_z_m'] = current
        if value['close_gripper_succeeded'] is not True:
            raise ValueError('close_gripper_succeeded evidence is required')
        if value['assert_holding_failed'] is not True:
            raise ValueError('assert_holding_failed evidence is required')
        if FORBIDDEN_DECISION_FIELDS.intersection(value):
            raise ValueError('forbidden decision field in grasp-no-hold signal')
        return value

@dataclass(frozen=True)
class JamZDecision:
    applied: bool
    corrected_z_m: float
    source: str
    rationale: str = ''
    parse_error: Optional[str] = None
    latch_reused: bool = False
    prior_z_m: Optional[float] = None
    write_succeeded: bool = False
    vlm_call_attempted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

class OpenRouterJamZClient:
    """Synchronous adapter that requires the parameter transport retry handler."""

    def __init__(self, *, api_key: str, base_url: str=OPENROUTER_BASE_URL, timeout_s: float=60.0, client: Any=None) -> None:
        if not api_key:
            raise ValueError('OpenRouter api_key is required')
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout_s = float(timeout_s)
        self._client = client
        self.calls = 0

    async def _complete_async(self, prompt: str, envelope: dict[str, Any]) -> str:
        from ..providers.openrouter import OpenRouterChatClient
        frozen = json.loads(json.dumps(envelope))
        if frozen != OPENROUTER_ENVELOPE:
            raise ValueError('grasp-height reflector OpenRouter envelope mismatch')
        messages = [{'role': 'system', 'content': 'You adjust only the grasp target z behaviour parameter after a classified physical descent jam. Return strict JSON only.'}, {'role': 'user', 'content': prompt}]
        client = self._client or OpenRouterChatClient(api_key=self.api_key, base_url=self.base_url, timeout_s=self.timeout_s)
        owns_client = self._client is None
        try:
            chat_envelope = getattr(client, 'chat_envelope', None)
            if not callable(chat_envelope) or not getattr(chat_envelope, '__carryon_parameter_transport_retry__', False):
                raise RuntimeError('grasp-height parameter transport retry handler is not installed')
            result = await chat_envelope(envelope=frozen, messages=messages)
            text = getattr(result, 'text', None)
            if not isinstance(text, str):
                raise RuntimeError('grasp-height transport returned a non-text result')
            return text
        finally:
            if owns_client:
                await client.aclose()

    def complete(self, prompt: str, envelope: dict[str, Any]) -> str:
        self.calls += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('synchronous grasp-height reflector called from a running event loop')
        return asyncio.run(self._complete_async(prompt, envelope))


class JamZReflector:
    """VLM reflector with a per-run latch and explicit cross-row object memory."""

    def __init__(self, client: JamZClient, *, ledger_path: Optional[Path]=None, persistent_memory_path: Optional[Path]=None, min_z_m: float=MIN_Z_M, max_z_m: float=MAX_Z_M, max_abs_delta_m: float=MAX_ABS_Z_DELTA_M, write_trigger: str='postrow_failure_evidence') -> None:
        self.client = client
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.persistent_memory_path = Path(persistent_memory_path) if persistent_memory_path is not None else None
        self.min_z_m = float(min_z_m)
        self.max_z_m = float(max_z_m)
        self.max_abs_delta_m = float(max_abs_delta_m)
        if write_trigger not in {'postrow_failure_evidence'}:
            raise ValueError('unknown z-reflection write trigger')
        self.write_trigger = write_trigger
        self._vlm_calls_this_row = 0
        self._latch: dict[str, JamZDecision] = {}
        self._persistent_keys: set[str] = set()
        if self.persistent_memory_path is not None:
            if not self.persistent_memory_path.is_absolute():
                raise ValueError('persistent jam-z memory path must be absolute')
            self._load_persistent_memory()
        if self.ledger_path is not None:
            self._initialize_ledger()

    @property
    def latch(self) -> dict[str, JamZDecision]:
        return dict(self._latch)

    @property
    def persistent_keys(self) -> frozenset[str]:
        return frozenset(self._persistent_keys)

    @property
    def persistent_priors_m(self) -> dict[str, float]:
        return {key: float(decision.prior_z_m) for key, decision in self._latch.items() if decision.prior_z_m is not None and key in self._persistent_keys}

    def reflect(self, signal: JamResidualSignal, *, target_before_xyz_m: Optional[list[float]]=None) -> JamZDecision:
        """Earn a future-row z prior from a classified descent jam.

        This method is called only after handback. It never changes the current
        action and returns applied=False even when persistence succeeds.
        """
        if self.write_trigger != 'postrow_failure_evidence':
            return self._fallback(signal.current_z_m, 'postrow_trigger_disabled', event='DESCENT_JAM_WRITE_SKIPPED_TRIGGER_DISABLED', object_key=getattr(signal, 'object_key', None))
        try:
            payload = signal.payload()
        except Exception as exc:
            return self._fallback(signal.current_z_m, f'invalid_signal: {exc}', event='DESCENT_JAM_WRITE_SKIPPED_INVALID_SIGNAL', object_key=getattr(signal, 'object_key', None))
        key = payload['object_key']
        existing = self._latch.get(key)
        if existing is not None and key in self._persistent_keys:
            decision = JamZDecision(applied=False, corrected_z_m=existing.corrected_z_m, prior_z_m=existing.prior_z_m, source='vlm_persistent_memory_existing', rationale=existing.rationale, latch_reused=True)
            self._write_ledger(payload, decision, target_before_xyz_m, raw_response=None, event='DESCENT_JAM_WRITE_SKIPPED_EXISTING')
            return decision
        if self._vlm_calls_this_row >= 1:
            decision = self._fallback(payload['current_z_m'], 'per_row_vlm_call_cap_reached')
            self._write_ledger(payload, decision, target_before_xyz_m, raw_response=None, event='DESCENT_JAM_WRITE_SKIPPED_CALL_CAP')
            return decision
        self._vlm_calls_this_row += 1
        raw = None
        try:
            raw = self.client.complete(self._prompt(payload), json.loads(json.dumps(OPENROUTER_ENVELOPE)))
            parsed = self._parse(raw, current_z_m=payload['current_z_m'])
            candidate = JamZDecision(applied=False, corrected_z_m=parsed.corrected_z_m, prior_z_m=parsed.corrected_z_m - payload['current_z_m'], source='vlm_runtime_earned_memory', rationale=parsed.rationale, vlm_call_attempted=True)
            persisted = self._persist_or_reuse(key, candidate, trigger='descent_jam')
            decision = JamZDecision(applied=False, corrected_z_m=persisted.corrected_z_m, prior_z_m=persisted.prior_z_m, source='vlm_runtime_earned_memory', rationale=persisted.rationale, latch_reused=persisted.latch_reused, write_succeeded=True, vlm_call_attempted=True)
            self._latch[key] = decision
        except Exception as exc:
            decision = JamZDecision(applied=False, corrected_z_m=payload['current_z_m'], source='mechanical_fail_open', parse_error=f'vlm_or_parse_failure: {type(exc).__name__}: {exc}', vlm_call_attempted=True)
        self._write_ledger(payload, decision, target_before_xyz_m, raw_response=raw, event='DESCENT_JAM_MEMORY_WRITE')
        return decision

    def reflect_grasp_no_hold(self, signal: GraspNoHoldSignal) -> JamZDecision:
        if self.write_trigger != 'postrow_failure_evidence':
            return self._fallback(signal.current_z_m, 'grasp_no_hold_trigger_disabled', event='GRASP_NO_HOLD_WRITE_SKIPPED_TRIGGER_DISABLED', object_key=getattr(signal, 'object_key', None))
        try:
            payload = signal.payload()
        except Exception as exc:
            return self._fallback(signal.current_z_m, f'invalid_signal: {exc}', event='GRASP_NO_HOLD_WRITE_SKIPPED_INVALID_SIGNAL', object_key=getattr(signal, 'object_key', None))
        key = payload['object_key']
        existing = self._latch.get(key)
        if existing is not None and key in self._persistent_keys:
            decision = JamZDecision(applied=False, corrected_z_m=existing.corrected_z_m, prior_z_m=existing.prior_z_m, source='vlm_persistent_memory_existing', rationale=existing.rationale, latch_reused=True, write_succeeded=False, vlm_call_attempted=False)
            self._write_ledger(payload, decision, None, raw_response=None, event='GRASP_NO_HOLD_WRITE_SKIPPED_EXISTING')
            return decision
        if self._vlm_calls_this_row >= 1:
            decision = self._fallback(payload['current_z_m'], 'per_row_vlm_call_cap_reached')
            self._write_ledger(payload, decision, None, raw_response=None, event='GRASP_NO_HOLD_WRITE_SKIPPED_CALL_CAP')
            return decision
        self._vlm_calls_this_row += 1
        prompt = self._grasp_no_hold_prompt(payload)
        try:
            raw = self.client.complete(prompt, json.loads(json.dumps(OPENROUTER_ENVELOPE)))
            parsed = self._parse(raw, current_z_m=payload['current_z_m'])
            decision = JamZDecision(applied=False, corrected_z_m=parsed.corrected_z_m, prior_z_m=parsed.corrected_z_m - payload['current_z_m'], source='vlm_runtime_earned_memory', rationale=parsed.rationale, write_succeeded=False, vlm_call_attempted=True)
            persisted = self._persist_or_reuse(key, decision, trigger='grasp_no_hold')
            decision = JamZDecision(applied=False, corrected_z_m=persisted.corrected_z_m, prior_z_m=persisted.prior_z_m, source='vlm_runtime_earned_memory' if persisted.source != 'vlm_persistent_memory' else 'vlm_persistent_memory_race_reuse', rationale=persisted.rationale, latch_reused=persisted.latch_reused, write_succeeded=True, vlm_call_attempted=True)
            self._latch[key] = decision
        except Exception as exc:
            raw = None
            decision = JamZDecision(applied=False, corrected_z_m=payload['current_z_m'], prior_z_m=None, source='mechanical_fail_open', parse_error=f'vlm_or_parse_failure: {type(exc).__name__}: {exc}', write_succeeded=False, vlm_call_attempted=True)
        self._write_ledger(payload, decision, None, raw_response=raw, event='GRASP_NO_HOLD_MEMORY_WRITE')
        return decision

    def _prompt(self, payload: dict[str, Any]) -> str:
        if set(payload) != SIGNAL_FIELDS:
            raise ValueError('prompt payload schema drift')
        if FORBIDDEN_DECISION_FIELDS.intersection(payload):
            raise ValueError('forbidden decision field in prompt payload')
        return 'A z-axis descent jam has already been classified by fixed mechanics. Read the residual telemetry and choose one corrected grasp target z in metres. You may change only z; do not propose thresholds or verification. Return exactly one JSON object with keys corrected_z_m (number) and rationale (short string), with no markdown. SIGNAL=' + json.dumps(payload, sort_keys=True, separators=(',', ':'))

    def _grasp_no_hold_prompt(self, payload: dict[str, Any]) -> str:
        if set(payload) != GRASP_NO_HOLD_FIELDS:
            raise ValueError('grasp-no-hold prompt payload schema drift')
        if FORBIDDEN_DECISION_FIELDS.intersection(payload):
            raise ValueError('forbidden decision field in prompt payload')
        return 'A robot completed gripper closure but the subsequent fixed hold check failed. Reflect on the observed target object and choose one corrected grasp target z in metres for future rows. This call is post-row memory writing only and cannot alter the current action. Return exactly one JSON object with keys corrected_z_m (number) and rationale (short string), with no markdown. SIGNAL=' + json.dumps(payload, sort_keys=True, separators=(',', ':'))

    def _parse(self, raw: str, *, current_z_m: float) -> JamZDecision:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('response must be a JSON object')
        if set(value) != RESPONSE_FIELDS:
            raise ValueError('response has unknown or missing fields')
        corrected = value['corrected_z_m']
        if isinstance(corrected, bool) or not isinstance(corrected, (int, float)):
            raise ValueError('corrected_z_m must be numeric')
        corrected = float(corrected)
        if not math.isfinite(corrected):
            raise ValueError('corrected_z_m must be finite')
        if not self.min_z_m <= corrected <= self.max_z_m:
            raise ValueError('corrected_z_m outside physical bounds')
        if abs(corrected - float(current_z_m)) > self.max_abs_delta_m:
            raise ValueError('corrected_z_m delta outside physical bounds')
        rationale = value['rationale']
        if not isinstance(rationale, str) or len(rationale) > 240:
            raise ValueError('rationale must be a short string')
        return JamZDecision(applied=True, corrected_z_m=corrected, source='vlm_runtime', rationale=rationale)

    def _fallback(self, current_z_m: float, error: str, *, event: Optional[str]=None, object_key: Optional[str]=None) -> JamZDecision:
        if event is not None:
            self._append_ledger_row({'schema': 'carryon-fixv2-parameter-writer-instrument/v2', 'created_utc': datetime.now(timezone.utc).isoformat(), 'event': event, 'object_key': str(object_key).strip().lower() if object_key else None, 'error': str(error), 'parse_error': str(error), 'write_succeeded': False, 'vlm_call_attempted': False, 'current_action_unchanged': True}, exclusive=False)
        return JamZDecision(applied=False, corrected_z_m=float(current_z_m), source='mechanical_fallback', parse_error=error)

    def _persistent_lock_path(self) -> Path:
        if self.persistent_memory_path is None:
            raise RuntimeError('persistent jam-z memory is not configured')
        return self.persistent_memory_path.with_name(self.persistent_memory_path.name + '.lock')

    def _parse_persistent_row(self, value: Any, *, line_number: int) -> tuple[str, JamZDecision]:
        if not isinstance(value, dict) or set(value) != PERSISTENT_MEMORY_FIELDS:
            raise ValueError(f'persistent jam-z memory schema drift at line {line_number}')
        if value['schema'] != PERSISTENT_MEMORY_SCHEMA:
            raise ValueError(f'persistent jam-z memory version drift at line {line_number}')
        key = str(value['object_key']).strip().lower()
        if not key or key != value['object_key']:
            raise ValueError(f'noncanonical object key at line {line_number}')
        corrected = value['corrected_z_m']
        if isinstance(corrected, bool) or not isinstance(corrected, (int, float)):
            raise ValueError(f'persistent corrected_z_m must be numeric at line {line_number}')
        corrected = float(corrected)
        if not math.isfinite(corrected) or not self.min_z_m <= corrected <= self.max_z_m:
            raise ValueError(f'persistent corrected_z_m outside physical bounds at line {line_number}')
        rationale = value['rationale']
        if not isinstance(rationale, str) or len(rationale) > 240:
            raise ValueError(f'persistent rationale invalid at line {line_number}')
        if value['value_source'] != 'vlm_runtime':
            raise ValueError(f'persistent value source invalid at line {line_number}')
        created_utc = value['created_utc']
        if not isinstance(created_utc, str) or not created_utc:
            raise ValueError(f'persistent created_utc invalid at line {line_number}')
        prior = value['prior_z_m']
        if isinstance(prior, bool) or not isinstance(prior, (int, float)):
            raise ValueError(f'persistent prior_z_m must be numeric at line {line_number}')
        prior = float(prior)
        if not math.isfinite(prior) or abs(prior) > self.max_abs_delta_m:
            raise ValueError(f'persistent prior_z_m invalid at line {line_number}')
        if value['trigger'] not in {'grasp_no_hold', 'descent_jam'}:
            raise ValueError(f'persistent trigger invalid at line {line_number}')
        return (key, JamZDecision(applied=True, corrected_z_m=corrected, prior_z_m=prior, source='vlm_persistent_memory', rationale=rationale, latch_reused=True, write_succeeded=True))

    def _read_persistent_rows(self) -> dict[str, JamZDecision]:
        if self.persistent_memory_path is None or not self.persistent_memory_path.exists():
            return {}
        loaded: dict[str, JamZDecision] = {}
        with self.persistent_memory_path.open('r', encoding='utf-8') as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise ValueError(f'blank persistent jam-z memory row at line {line_number}')
                key, decision = self._parse_persistent_row(json.loads(raw), line_number=line_number)
                if key in loaded:
                    raise ValueError(f'duplicate persistent jam-z object key: {key}')
                loaded[key] = decision
        return loaded

    def _load_persistent_memory(self) -> None:
        loaded = self._read_persistent_rows()
        self._latch.update(loaded)
        self._persistent_keys.update(loaded)

    def _persist_or_reuse(self, key: str, decision: JamZDecision, *, trigger: str) -> JamZDecision:
        if trigger not in {'grasp_no_hold', 'descent_jam'}:
            raise ValueError('unknown post-row parameter-memory trigger')
        if self.persistent_memory_path is None:
            return decision
        self.persistent_memory_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._persistent_lock_path()
        with lock_path.open('a', encoding='utf-8') as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                loaded = self._read_persistent_rows()
                existing = loaded.get(key)
                if existing is not None:
                    self._latch.update(loaded)
                    self._persistent_keys.update(loaded)
                    return existing
                row = {'schema': PERSISTENT_MEMORY_SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(), 'object_key': key, 'corrected_z_m': decision.corrected_z_m, 'prior_z_m': decision.prior_z_m, 'rationale': decision.rationale, 'trigger': trigger, 'value_source': 'vlm_runtime'}
                mode = 'a' if self.persistent_memory_path.exists() else 'x'
                with self.persistent_memory_path.open(mode, encoding='utf-8') as stream:
                    stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                self._persistent_keys.add(key)
                return decision
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def record_instrument_event(self, *, event: str, object_key: Optional[str], candidates: Optional[list[str]]=None, resolved_key: Optional[str]=None) -> None:
        """Append a content-blind post-row writer event without changing actions."""
        if not event.startswith(('WRITE_SKIPPED_', 'WRITE_KEY_')):
            raise ValueError('parameter instrumentation event must use declared event vocabulary')
        self._append_ledger_row({'schema': 'carryon-fixv2-parameter-writer-instrument/v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'event': event, 'object_key': object_key, 'candidate_keys': list(candidates or []), 'resolved_key': resolved_key, 'write_succeeded': False, 'vlm_call_attempted': False, 'current_action_unchanged': True}, exclusive=False)

    def _initialize_ledger(self) -> None:
        self._append_ledger_row({'schema': 'carryon-bv2pp-vlm-z-parameter-memory-ledger-init/v4', 'created_utc': datetime.now(timezone.utc).isoformat(), 'event': 'REFLECTOR_INITIALIZED', 'reflector_active': True, 'write_trigger': self.write_trigger, 'goalset_jam_trigger_retired': True, 'episode_local_latch_empty': len(self._latch) == 0, 'persistent_memory_path': str(self.persistent_memory_path) if self.persistent_memory_path is not None else None, 'persistent_loaded_keys': sorted(self._persistent_keys), 'vlm_calls_at_initialization': int(getattr(self.client, 'calls', 0))}, exclusive=True)

    def _append_ledger_row(self, row: dict[str, Any], *, exclusive: bool) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'x' if exclusive else 'a'
        with self.ledger_path.open(mode, encoding='utf-8') as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _write_ledger(self, payload: dict[str, Any], decision: JamZDecision, target_before_xyz_m: Optional[list[float]], *, raw_response: Optional[str], event: str='GOALSET_JAM_REFLECTION') -> None:
        if self.ledger_path is None:
            return
        before = list(target_before_xyz_m) if target_before_xyz_m is not None else [None, None, payload['current_z_m']]
        after = list(before)
        after[2] = decision.corrected_z_m
        row = {'schema': 'carryon-bv2pp-vlm-z-memory-event/v3', 'created_utc': datetime.now(timezone.utc).isoformat(), 'event': event, 'write_trigger': self.write_trigger, 'applied': decision.applied, 'object_key': payload['object_key'], 'prior_z_m': decision.prior_z_m, 'target_before_xyz_m': before, 'target_after_prior_xyz_m': after, 'value_source': decision.source, 'latch_reused': decision.latch_reused, 'write_succeeded': decision.write_succeeded, 'vlm_call_attempted': decision.vlm_call_attempted, 'mechanism_signal': payload, 'vlm_response_raw': raw_response, 'parse_error': decision.parse_error}
        self._append_ledger_row(row, exclusive=False)
__all__ = ['GraspNoHoldSignal', 'GRASP_NO_HOLD_FIELDS', 'JamResidualSignal', 'JamZDecision', 'JamZReflector',  'OpenRouterJamZClient', 'OPENROUTER_ENVELOPE', 'PERSISTENT_MEMORY_FIELDS', 'PERSISTENT_MEMORY_SCHEMA', 'SIGNAL_FIELDS']
