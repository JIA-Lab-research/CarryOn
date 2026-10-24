'Bind semantic object memory and parameter memory to the recovery executor.'
from __future__ import annotations
from runtime_paths import bound_path, resource
import hashlib
import importlib
import json
import os
import re
import time
from pathlib import Path
BUILD = Path(__file__).resolve().parent
R7 = BUILD / 'semantic_memory_config.json'
SOM_PROMPT = BUILD / 'candidate_selection_prompt.txt'
JAM_Z_REFLECTION = BUILD / '../runtime' / 'carryon/recovery/jam_z_reflection.py'

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

EXPECTED = {'v7_registry': sha256(bound_path('references/FROZEN_REFERENCE_REGISTRY.json'))}

def _load_jam_z_reflection():
    if not JAM_Z_REFLECTION.is_file():
        raise RuntimeError('Parameter reflection module file is missing')
    module = importlib.import_module('carryon.recovery.jam_z_reflection')
    loaded_path = Path(module.__file__).resolve()
    if loaded_path != JAM_Z_REFLECTION.resolve():
        raise RuntimeError(f'jam_z_reflection import path drift: {loaded_path}')
    return module
JAM_Z_REFLECTION_MODULE = _load_jam_z_reflection()
MIN_Z_M = JAM_Z_REFLECTION_MODULE.MIN_Z_M
MAX_Z_M = JAM_Z_REFLECTION_MODULE.MAX_Z_M
MAX_ABS_Z_DELTA_M = JAM_Z_REFLECTION_MODULE.MAX_ABS_Z_DELTA_M

def normalize_key(value: str) -> str:
    value = str(value or '').strip().lower().replace('_', ' ')
    value = re.sub('\\s+', ' ', value)
    return re.sub('\\s+1$', '', value)

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}.{time.time_ns()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

def load_frozen_inputs() -> str:
    for path in (R7, SOM_PROMPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    return SOM_PROMPT.read_text(encoding='utf-8')


def _store_from_env(name: str) -> Path:
    text = os.environ.get(name, '').strip()
    if not text:
        raise RuntimeError(f'{name} is required')
    return Path(text).resolve()

def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, text in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not text.strip():
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise RuntimeError(f'non-object memory row at {path}:{line_number}')
        rows.append(value)
    return rows

def load_semantic_entries() -> dict[str, dict]:
    entries = {}
    for line_number, item in enumerate(_jsonl_rows(_store_from_env('CARRYON_RECOVERY_SEMANTIC_MEMORY_PATH')), 1):
        required = {'schema', 'created_utc', 'object_key', 'reference_path', 'reference_sha256', 'registry_sha256', 'template', 'rationale', 'trigger', 'value_source'}
        if set(item) != required or item['schema'] != 'carryon-fixv2-semantic-memory/v1':
            raise RuntimeError(f'semantic memory schema drift at line {line_number}')
        key = normalize_key(item['object_key'])
        if not key or key != item['object_key'] or key in entries:
            raise RuntimeError(f'semantic memory key drift at line {line_number}')
        ref = Path(item['reference_path']).resolve()
        if item['registry_sha256'] != EXPECTED['v7_registry']:
            raise RuntimeError(f'semantic registry SHA drift at line {line_number}')
        if not ref.is_file():
            raise RuntimeError(f'Semantic reference file is missing at line {line_number}')
        entries[key] = dict(item)
    return entries

def load_parameter_memory() -> dict[str, dict[str, float]]:
    rows = {}
    for line_number, item in enumerate(_jsonl_rows(_store_from_env('CARRYON_RECOVERY_PARAMETER_MEMORY_PATH')), 1):
        if item.get('schema') != 'carryon-fixv2-vlm-z-object-memory/v1':
            raise RuntimeError(f'parameter memory schema drift at line {line_number}')
        key = normalize_key(item.get('object_key'))
        prior = item.get('prior_z_m')
        corrected = item.get('corrected_z_m')
        if not key or key != item.get('object_key') or key in rows:
            raise RuntimeError(f'parameter memory key drift at line {line_number}')
        if isinstance(prior, bool) or not isinstance(prior, (int, float)):
            raise RuntimeError(f'parameter prior type drift at line {line_number}')
        if isinstance(corrected, bool) or not isinstance(corrected, (int, float)):
            raise RuntimeError(f'parameter corrected-z type drift at line {line_number}')
        prior = float(prior)
        corrected = float(corrected)
        if not -MAX_ABS_Z_DELTA_M <= prior <= MAX_ABS_Z_DELTA_M:
            raise RuntimeError(f'parameter prior outside signed bounds at line {line_number}')
        if not MIN_Z_M <= corrected <= MAX_Z_M:
            raise RuntimeError(f'parameter corrected-z outside physical bounds at line {line_number}')
        rows[key] = {'prior_z_m': prior, 'corrected_z_m': corrected}
    return rows

def load_parameter_priors() -> dict[str, float]:
    return {key: row['prior_z_m'] for key, row in load_parameter_memory().items()}

def load_parameter_corrected_z() -> dict[str, float]:
    return {key: row['corrected_z_m'] for key, row in load_parameter_memory().items()}

class ReloadingMap:
    """Mapping facade whose every read reloads the append-only batch store."""

    def __init__(self, loader):
        self._loader = loader

    def get(self, key, default=None):
        return self._loader().get(key, default)

    def __contains__(self, key):
        return key in self._loader()

    def __getitem__(self, key):
        return self._loader()[key]

    def __iter__(self):
        return iter(self._loader())

    def __len__(self):
        return len(self._loader())
PROMPT_BYTES = load_frozen_inputs()
ENTRIES = ReloadingMap(load_semantic_entries)
Z_PRIORS_M = ReloadingMap(load_parameter_priors)
Z_CORRECTED_M = ReloadingMap(load_parameter_corrected_z)

def _selector_for_backend(backend):
    for detector in list(getattr(backend, 'detector_sources', {}).values()) + [getattr(backend, 'detector', None)]:
        if detector is None:
            continue
        selector = getattr(detector, 'visual_object_selector', None)
        if selector is not None:
            return selector
        fallback = getattr(detector, 'fallback', None)
        selector = getattr(fallback, 'visual_object_selector', None)
        if selector is not None:
            return selector
    return None
