"""Parse and validate parameter and semantic reflection responses."""
import json,math,types
from decimal import Decimal
frozen=types.SimpleNamespace(RESPONSE_FIELDS={'corrected_z_m','rationale'},PERSISTENT_MEMORY_SCHEMA='carryon-fixv2-vlm-z-object-memory/v1',PERSISTENT_MEMORY_FIELDS={'schema','created_utc','object_key','corrected_z_m','prior_z_m','rationale','trigger','value_source'},JamZDecision=lambda **k:types.SimpleNamespace(**k))
RESPONSE_FIELDS=frozenset({'decision','template','rationale'})
TEMPLATE='REGROUND_V7_ALL_CANDIDATE_V1'

class PinnedParser:

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

def to_runtime_meters(raw):
    value = json.loads(raw, parse_float=Decimal, parse_int=Decimal, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('non-finite JSON number')))
    if not isinstance(value, dict) or set(value) != {'action', 'corrected_z_mm', 'rationale'}:
        raise ValueError('millimetre response schema mismatch')
    if not isinstance(value['rationale'], str):
        raise ValueError('rationale must be text')
    action = value['action']
    z = value['corrected_z_mm']
    if action == 'NO_UPDATE':
        if z is not None:
            raise ValueError('NO_UPDATE requires null')
        meters = None
    elif action == 'UPDATE_Z':
        if isinstance(z, bool) or not isinstance(z, Decimal) or (not z.is_finite()):
            raise ValueError('millimetres must be finite numeric')
        meters = float(z / Decimal(1000))
    else:
        raise ValueError('unknown action')
    return json.dumps({'action': action, 'corrected_z_m': meters, 'rationale': value['rationale']}, allow_nan=False, separators=(',', ':'))

def parse_mm_with_existing(raw, existing_parser, current_z_m):
    translated = to_runtime_meters(raw)
    return (existing_parser._parse(translated, current_z_m=current_z_m), translated)

def parse_p(raw,anchor_m):
    if isinstance(anchor_m,bool) or not isinstance(anchor_m,(int,float)) or not math.isfinite(anchor_m):
        raise ValueError('missing finite frozen anchor')
    parser=PinnedParser()
    parser.min_z_m=0.0
    parser.max_z_m=0.25
    parser.max_abs_delta_m=0.20
    decision,translated=parse_mm_with_existing(raw,parser,anchor_m)
    return dict(vars(decision),translated_runtime_json=translated)

def parse_s(raw):
    debug={}
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != RESPONSE_FIELDS:
        raise ValueError('semantic response schema drift')
    if value['decision'] not in {'WRITE', 'SKIP'}:
        raise ValueError('semantic response decision drift')
    if value['template'] != TEMPLATE:
        raise ValueError('semantic response template drift')
    rationale = value['rationale']
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('semantic rationale invalid')
    if len(rationale) > 240:
        debug['rationale_quality_flags'] = ['EXCEEDS_PREFERRED_LENGTH']
        debug['rationale_character_count'] = len(rationale)
        debug['rationale_preferred_max_chars'] = 240
    return dict(value,quality=debug)
