"""Write semantic memory from episode evidence for subsequent recovery rounds."""
from __future__ import annotations
import asyncio
import base64
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol
OPENROUTER_ENVELOPE = {'model': 'qwen/qwen3-vl-235b-a22b-instruct', 'temperature': 0.0, 'top_p': 0.8, 'max_tokens': 700, 'provider': {'data_collection': 'deny'}}
RESPONSE_FIELDS = frozenset({'decision', 'template', 'rationale'})
TEMPLATE = 'REGROUND_V7_ALL_CANDIDATE_V1'
STORE_SCHEMA = 'carryon-fixv2-semantic-memory/v1'
STORE_FIELDS = frozenset({'schema', 'created_utc', 'object_key', 'reference_path', 'reference_sha256', 'registry_sha256', 'template', 'rationale', 'trigger', 'value_source'})

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalize_key(value: str) -> str:
    return ' '.join(str(value or '').strip().lower().replace('_', ' ').split())

def resolve_registry_key(value: str, reference_keys) -> dict[str, Any]:
    """Resolve an instruction key against the object registry."""
    original = normalize_key(value)
    keys = tuple(sorted({normalize_key(item) for item in reference_keys if normalize_key(item)}))
    if original in keys:
        return {'original_key': original, 'resolved_key': original, 'candidate_keys': [original], 'resolution': 'DIRECT'}
    candidates = [item for item in keys if item in original]
    return {'original_key': original, 'resolved_key': candidates[0] if len(candidates) == 1 else None, 'candidate_keys': candidates, 'resolution': 'REGISTRY_SUBSTRING_UNIQUE' if len(candidates) == 1 else 'UNRESOLVED'}

class SemanticClient(Protocol):

    def complete(self, prompt: str, reference_path: Path, envelope: dict[str, Any]) -> str:
        ...

class OpenRouterSemanticClient:

    def __init__(self, *, api_key: str, timeout_s: float=60.0, client=None) -> None:
        if not api_key:
            raise ValueError('OpenRouter api_key is required')
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self._client = client
        self.calls = 0

    async def _complete_async(self, prompt: str, reference_path: Path, envelope: dict[str, Any]) -> str:
        from ..providers.openrouter import OpenRouterChatClient
        frozen = json.loads(json.dumps(envelope))
        if frozen != OPENROUTER_ENVELOPE:
            raise ValueError('Semantic reflection model envelope mismatch')
        payload = base64.b64encode(reference_path.read_bytes()).decode('ascii')
        messages = [{'role': 'system', 'content': 'You are a post-row robot recovery memory reflector. Return strict JSON only. You cannot alter the current action.'}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + payload}}]}]
        client = self._client or OpenRouterChatClient(api_key=self.api_key, timeout_s=self.timeout_s)
        owns_client = self._client is None
        try:
            chat_envelope = getattr(client, 'chat_envelope', None)
            if not callable(chat_envelope) or not getattr(chat_envelope, '__carryon_parameter_transport_retry__', False):
                raise RuntimeError('parameter transport retry handler is not installed')
            result = await chat_envelope(envelope=frozen, messages=messages)
            text = getattr(result, 'text', None)
            if not isinstance(text, str):
                raise RuntimeError('semantic reflector returned non-text')
            return text
        finally:
            if owns_client:
                await client.aclose()

    def complete(self, prompt: str, reference_path: Path, envelope: dict[str, Any]) -> str:
        self.calls += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('semantic reflector called from running event loop')
        return asyncio.run(self._complete_async(prompt, reference_path, envelope))


class SemanticMemoryReflector:

    def __init__(self, client: SemanticClient, *, memory_path: Path, ledger_path: Path, compat_manifest_path: Path, registry_sha256: str) -> None:
        self.client = client
        self.memory_path = Path(memory_path).resolve()
        self.ledger_path = Path(ledger_path).resolve()
        self.compat_manifest_path = Path(compat_manifest_path).resolve()
        self.registry_sha256 = str(registry_sha256)
        self._calls_this_row = 0
        if self.ledger_path.exists():
            raise FileExistsError('semantic row ledger must be fresh')
        manifest = json.loads(self.compat_manifest_path.read_text(encoding='utf-8'))
        if manifest.get('source_registry_sha256') != self.registry_sha256:
            raise ValueError('V7 registry binding drift')
        self.references = {}
        for item in manifest.get('objects', []):
            key = normalize_key(item.get('object_name'))
            path = Path(item.get('canonical_rgb', '')).resolve()
            if not key or not path.is_file():
                raise ValueError('V7 compat object identity drift')
            self.references[key] = (path, item['sha256'])
        if not self.references:
            raise ValueError('V7 compat manifest has no objects')
        self._append_ledger({'schema': 'carryon-fixv2-semantic-ledger-init/v1', 'event': 'REFLECTOR_INITIALIZED', 'memory_path': str(self.memory_path), 'compat_manifest_path': str(self.compat_manifest_path), 'registry_sha256': self.registry_sha256, 'vlm_calls_at_initialization': 0}, exclusive=True)

    def resolve_object_key(self, value: str) -> dict[str, Any]:
        return resolve_registry_key(value, self.references)

    def _read_store(self) -> dict[str, dict]:
        if not self.memory_path.exists():
            return {}
        rows = {}
        for line_number, raw in enumerate(self.memory_path.read_text(encoding='utf-8').splitlines(), 1):
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != STORE_FIELDS:
                raise ValueError(f'semantic store schema drift at line {line_number}')
            if value['schema'] != STORE_SCHEMA:
                raise ValueError(f'semantic store version drift at line {line_number}')
            key = normalize_key(value['object_key'])
            if key != value['object_key'] or key in rows:
                raise ValueError(f'semantic store key drift at line {line_number}')
            ref = Path(value['reference_path']).resolve()
            if value['registry_sha256'] != self.registry_sha256 or not ref.is_file():
                raise ValueError(f'semantic store reference drift at line {line_number}')
            rows[key] = value
        return rows

    def reflect(self, *, object_key: str, failed_steps: list[dict]) -> dict:
        resolution = self.resolve_object_key(object_key)
        key = resolution['resolved_key']
        debug = {'schema': 'carryon-fixv2-semantic-postrow-evidence/v1', 'object_key': key, 'original_object_key': resolution['original_key'] or None, 'candidate_keys': list(resolution['candidate_keys']), 'key_resolution': resolution['resolution'], 'key_resolved_by_registry': resolution['resolution'] == 'REGISTRY_SUBSTRING_UNIQUE', 'failed_step_count': len(failed_steps), 'triggered': False, 'write_succeeded': False, 'vlm_call_attempted': False, 'current_action_unchanged': True}
        if key is None:
            self._append_ledger(dict(debug, event='WRITE_SKIPPED_KEY_UNRESOLVED', key_unresolved=True))
            return debug
        if not failed_steps:
            self._append_ledger(dict(debug, event='WRITE_SKIPPED_NO_EVIDENCE', key_unresolved=False))
            return debug
        existing = self._read_store()
        if key in existing:
            debug.update({'triggered': True, 'skipped_existing': True})
            self._append_ledger(dict(debug, event='WRITE_SKIPPED_EXISTING'))
            return debug
        if self._calls_this_row >= 1:
            debug['fail_open_error'] = 'per_row_semantic_call_cap_reached'
            self._append_ledger(dict(debug, event='WRITE_SKIPPED_CALL_CAP'))
            return debug
        self._calls_this_row += 1
        ref_path, ref_sha = self.references[key]
        compact = [{'op': str(item.get('op', ''))[:80], 'reason': str(item.get('reason', ''))[:240]} for item in failed_steps[:12]]
        prompt = 'A fixed robot recovery just ended with the following failed mechanical steps. Decide whether future rows for the named object should enable the signed REGROUND_V7_ALL_CANDIDATE_V1 identity-comparison template. The attached image is the signed isolated V7 reference and contains no scene location. Return exactly {"decision":"WRITE|SKIP","template":"REGROUND_V7_ALL_CANDIDATE_V1","rationale":"short string"}. ' + json.dumps({'object_key': key, 'failed_steps': compact}, sort_keys=True, separators=(',', ':'))
        try:
            raw = self.client.complete(prompt, ref_path, json.loads(json.dumps(OPENROUTER_ENVELOPE)))
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != RESPONSE_FIELDS:
                raise ValueError('semantic response schema drift')
            if value['decision'] not in {'WRITE', 'SKIP'}:
                raise ValueError('semantic response decision drift')
            if value['template'] != TEMPLATE:
                raise ValueError('semantic response template drift')
            rationale = value['rationale']
            if not isinstance(rationale, str) or not rationale or len(rationale) > 240:
                raise ValueError('semantic rationale invalid')
            debug.update({'triggered': True, 'vlm_call_attempted': True, 'decision': value['decision'], 'reference_sha256': ref_sha})
            if value['decision'] == 'WRITE':
                row = {'schema': STORE_SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(), 'object_key': key, 'reference_path': str(ref_path), 'reference_sha256': ref_sha, 'registry_sha256': self.registry_sha256, 'template': TEMPLATE, 'rationale': rationale, 'trigger': 'postrow_recovery_failure', 'value_source': 'vlm_runtime'}
                debug['write_succeeded'] = self._append_if_absent(key, row)
        except Exception as exc:
            debug.update({'triggered': True, 'vlm_call_attempted': True, 'fail_open_error': f'{type(exc).__name__}: {exc}'})
        self._append_ledger(dict(debug, event='SEMANTIC_MEMORY_PROPOSAL'))
        return debug

    def _append_if_absent(self, key: str, row: dict) -> bool:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.memory_path.with_name(self.memory_path.name + '.lock')
        with lock.open('a', encoding='utf-8') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                if key in self._read_store():
                    return False
                mode = 'a' if self.memory_path.exists() else 'x'
                with self.memory_path.open(mode, encoding='utf-8') as output:
                    output.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                return True
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _append_ledger(self, row: dict, *, exclusive: bool=False) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'x' if exclusive else 'a'
        with self.ledger_path.open(mode, encoding='utf-8') as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
