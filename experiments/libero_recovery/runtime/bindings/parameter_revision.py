"""Bind parameter-reflection proposals to active grasp-height memory entries."""
from runtime_paths import bound_path, resource
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import tempfile
from carryon.recovery import jam_z_reflection as frozen
FrozenReflector = frozen.JamZReflector
Decision = frozen.JamZDecision

def digest(data):
    return hashlib.sha256(data).hexdigest()

class RevisableJamZReflector(FrozenReflector):

    def _parse(self, raw, *, current_z_m):
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {'action', 'corrected_z_m', 'rationale'}:
            raise ValueError('explicit action response schema mismatch')
        rationale = value['rationale']
        if not isinstance(rationale, str) or len(rationale) > 240:
            raise ValueError('rationale must be a short string')
        if value['action'] == 'NO_UPDATE':
            if value['corrected_z_m'] is not None:
                raise ValueError('NO_UPDATE requires null corrected_z_m')
            return Decision(applied=False, corrected_z_m=current_z_m, source='vlm_no_update', rationale=rationale)
        if value['action'] != 'UPDATE_Z':
            raise ValueError('unknown action')
        return super()._parse(json.dumps({'corrected_z_m': value['corrected_z_m'], 'rationale': rationale}), current_z_m=current_z_m)

    def _decision_prompt(self, payload, trigger):
        expected = frozen.SIGNAL_FIELDS if trigger == 'descent_jam' else frozen.GRASP_NO_HOLD_FIELDS
        if set(payload) != expected or frozen.FORBIDDEN_DECISION_FIELDS.intersection(payload):
            raise ValueError('frozen signal schema drift')
        existing = self._read_persistent_rows().get(payload['object_key'])
        active = None if existing is None else {'corrected_z_m': existing.corrected_z_m, 'prior_z_m': existing.prior_z_m, 'rationale': existing.rationale, 'source': existing.source}
        return 'This is post-row reflection on an unresolved ' + trigger + '. Choose NO_UPDATE when the supplied evidence does not support a z change. NO_UPDATE retains the current valid configuration and any existing memory; it does not reset z. UPDATE_Z proposes one absolute grasp contact-target z in metres for later rows, not a hand/flange z or a lift endpoint. A failed hold alone does not identify a z error. An existing object entry may be revised if the evidence supports a different value. Do not infer that copying a failed height makes it effective. Do not change other parameters. Return exactly {"action":"NO_UPDATE","corrected_z_m":null,"rationale":"..."} or {"action":"UPDATE_Z","corrected_z_m":number,"rationale":"..."}, rationale at most 240 characters, no markdown. SIGNAL=' + json.dumps(payload, sort_keys=True, separators=(',', ':')) + ' ACTIVE_MEMORY=' + json.dumps(active, sort_keys=True, separators=(',', ':'))

    def reflect(self, signal, *, target_before_xyz_m=None):
        return self._reflect_terminal(signal, 'descent_jam', target_before_xyz_m)

    def reflect_grasp_no_hold(self, signal):
        return self._reflect_terminal(signal, 'grasp_no_hold', None)

    def _reflect_terminal(self, signal, trigger, before):
        prefix = trigger.upper()
        if self.write_trigger != 'postrow_failure_evidence':
            return self._fallback(signal.current_z_m, 'postrow_trigger_disabled', event=prefix + '_WRITE_SKIPPED_TRIGGER_DISABLED', object_key=getattr(signal, 'object_key', None))
        try:
            payload = signal.payload()
        except Exception as exc:
            return self._fallback(signal.current_z_m, 'invalid_signal: ' + str(exc), event=prefix + '_WRITE_SKIPPED_INVALID_SIGNAL', object_key=getattr(signal, 'object_key', None))
        if self._vlm_calls_this_row >= 1:
            decision = self._fallback(payload['current_z_m'], 'per_row_vlm_call_cap_reached')
            self._write_ledger(payload, decision, before, raw_response=None, event=prefix + '_WRITE_SKIPPED_CALL_CAP')
            return decision
        raw = None
        self._vlm_calls_this_row += 1
        try:
            prompt = self._decision_prompt(payload, trigger)
            raw = self.client.complete(prompt, json.loads(json.dumps(frozen.OPENROUTER_ENVELOPE)))
            parsed = self._parse(raw, current_z_m=payload['current_z_m'])
            if parsed.source == 'vlm_no_update':
                path = self.persistent_memory_path
                if path is None:
                    existing = self._latch.get(payload['object_key'])
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with self._persistent_lock_path().open('a', encoding='utf-8') as lock:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                        try:
                            loaded = self._read_persistent_rows()
                            existing = loaded.get(payload['object_key'])
                            self._latch.update(loaded)
                            self._persistent_keys.update(loaded)
                        finally:
                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                decision = Decision(applied=False, corrected_z_m=existing.corrected_z_m if existing else payload['current_z_m'], prior_z_m=existing.prior_z_m if existing else None, source='vlm_no_update', rationale=parsed.rationale, latch_reused=existing is not None, write_succeeded=False, vlm_call_attempted=True)
                self._write_ledger(payload, decision, before, raw_response=raw, event=prefix + '_WRITE_SKIPPED_NO_UPDATE')
                return decision
            candidate = Decision(applied=False, corrected_z_m=parsed.corrected_z_m, prior_z_m=parsed.corrected_z_m - payload['current_z_m'], source='vlm_runtime_earned_memory', rationale=parsed.rationale, vlm_call_attempted=True)
            decision = self._persist_or_reuse(payload['object_key'], candidate, trigger=trigger)
            if decision.write_succeeded:
                self._latch[payload['object_key']] = decision
        except Exception as exc:
            decision = Decision(applied=False, corrected_z_m=payload['current_z_m'], source='mechanical_fail_open', parse_error=f'vlm_or_parse_failure: {type(exc).__name__}: {exc}', vlm_call_attempted=True)
        self._write_ledger(payload, decision, before, raw_response=raw, event=prefix + '_MEMORY_WRITE')
        return decision

    def _persist_or_reuse(self, key, decision, *, trigger):
        if trigger not in {'grasp_no_hold', 'descent_jam'}:
            raise ValueError('unknown post-row parameter-memory trigger')
        path = self.persistent_memory_path
        if path is None:
            raise ValueError('persistent store required; no successful-write claim without disk')
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._persistent_lock_path().open('a', encoding='utf-8') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._read_persistent_rows()
                old_bytes = path.read_bytes() if path.exists() else b''
                rows = [json.loads(s) for s in old_bytes.splitlines()]
                prior_row = next((r for r in rows if r['object_key'] == key), None)
                row = {'schema': frozen.PERSISTENT_MEMORY_SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(), 'object_key': key, 'corrected_z_m': decision.corrected_z_m, 'prior_z_m': decision.prior_z_m, 'rationale': decision.rationale, 'trigger': trigger, 'value_source': 'vlm_runtime'}
                self._parse_persistent_row(row, line_number=1)
                if prior_row is None:
                    lines = list(old_bytes.splitlines(keepends=True))
                    if lines and (not lines[-1].endswith(b'\n')):
                        lines[-1] += b'\n'
                    lines.append((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
                else:
                    lines = [(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode() if r['object_key'] == key else s for r, s in zip(rows, old_bytes.splitlines(keepends=True))]
                new_bytes = b''.join(lines)
                audit = {'schema': 'carryon-parameter-active-row-revision/v1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'object_key': key, 'action': 'UPDATE_Z', 'operation': 'REVISE' if prior_row else 'INSERT', 'before_row': prior_row, 'after_row': row, 'store_before_sha256': digest(old_bytes), 'store_after_sha256': digest(new_bytes)}
                self._append_ledger_row(dict(audit, event='PARAMETER_UPDATE_PROPOSED', write_succeeded=False), exclusive=False)
                with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.tmp.', delete=False) as stream:
                    tmp = Path(stream.name)
                    stream.write(new_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.replace(tmp, path)
                    fd = os.open(str(path.parent), os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                finally:
                    if tmp.exists():
                        tmp.unlink()
                assert path.read_bytes() == new_bytes
                self._append_ledger_row(dict(audit, event='PARAMETER_UPDATE_COMMITTED', write_succeeded=True), exclusive=False)
                self._persistent_keys.add(key)
                return Decision(applied=False, corrected_z_m=decision.corrected_z_m, prior_z_m=decision.prior_z_m, source='vlm_runtime_earned_memory', rationale=decision.rationale, write_succeeded=True, vlm_call_attempted=True)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

def install(expected_candidate_sha):
    assert frozen.JamZReflector is FrozenReflector, 'unexpected existing class binding'
    frozen.JamZReflector = RevisableJamZReflector
    return {'original_module': str(frozen.__file__), 'effective_class_source': __file__, 'scope': 'isolated CPU process only; not deployed', 'trigger_unchanged': True}
