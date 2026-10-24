"""Isolated f1b2b369 amendment: rationale length is not action validity.

Numeric and persistent validators below are copied from the SHA-bound accepted
module. Only the rationale-length clause is removed. No trigger/robot change.
"""
import json, math
from carryon.recovery import jam_z_reflection as frozen
from parameter_evidence import EvidenceReflector

class LengthPreferenceReflector(EvidenceReflector):

    def _parse(self, raw, *, current_z_m):
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {'action', 'corrected_z_m', 'rationale'}:
            raise ValueError('explicit action response schema mismatch')
        rationale = value['rationale']
        if not isinstance(rationale, str):
            raise ValueError('rationale must be a string')
        if value['action'] == 'NO_UPDATE':
            if value['corrected_z_m'] is not None:
                raise ValueError('NO_UPDATE requires null corrected_z_m')
            return frozen.JamZDecision(applied=False, corrected_z_m=current_z_m, source='vlm_no_update', rationale=rationale)
        if value['action'] != 'UPDATE_Z':
            raise ValueError('unknown action')
        return self._numeric_parse(json.dumps({'corrected_z_m': value['corrected_z_m'], 'rationale': rationale}), current_z_m=current_z_m)

    def _numeric_parse(self, raw, *, current_z_m):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('response must be a JSON object')
        if set(value) != frozen.RESPONSE_FIELDS:
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
        if not isinstance(rationale, str):
            raise ValueError('rationale must be a string')
        return frozen.JamZDecision(applied=True, corrected_z_m=corrected, source='vlm_runtime', rationale=rationale)

    def _parse_persistent_row(self, value, *, line_number):
        if not isinstance(value, dict) or set(value) != frozen.PERSISTENT_MEMORY_FIELDS:
            raise ValueError(f'persistent jam-z memory schema drift at line {line_number}')
        if value['schema'] != frozen.PERSISTENT_MEMORY_SCHEMA:
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
        if not isinstance(rationale, str):
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
        return (key, frozen.JamZDecision(applied=True, corrected_z_m=corrected, prior_z_m=prior, source='vlm_persistent_memory', rationale=rationale, latch_reused=True, write_succeeded=True))

    def _decision_prompt(self, payload, trigger):
        prompt = super()._decision_prompt(payload, trigger)
        old = 'rationale at most 240 characters, no markdown. '
        assert prompt.count(old) == 1, 'Parameter prompt anchor changed'
        return prompt.replace(old, 'Prefer a concise rationale (about 240 characters); length is not an action-validity condition. No markdown. ')

def configure_inner(inner, selector_arm, row_root):
    import parameter_evidence as evidence
    from parameter_evidence_binding import configure_inner as accepted
    assert evidence.EvidenceReflector is EvidenceReflector
    evidence.EvidenceReflector = LengthPreferenceReflector
    accepted(inner, selector_arm, row_root)
    assert frozen.JamZReflector is LengthPreferenceReflector
