"""Load the selected memory version and collect failure evidence between rounds."""
from runtime_paths import bound_path, resource
from pathlib import Path
import os, functools
from memory_loading import *

def resolve_manifest_row(manifest, row_id):
    normalized = []
    for item in manifest['rows']:
        row = item['row'] if isinstance(item, dict) and 'row' in item else item
        require(isinstance(row, dict) and isinstance(row.get('row_id'), str), 'fixed consumer invalid row shape')
        normalized.append(row)
    ids = [row['row_id'] for row in normalized]
    require(len(ids) == len(set(ids)), 'fixed consumer duplicate row_id')
    matches = [row for row in normalized if row['row_id'] == row_id]
    require(len(matches) == 1, 'fixed S consumer row absent or ambiguous')
    return matches[0]

def install(inner, row_root):
    from collection_binding import install as collect
    pointer = Path(os.environ['CARRYON_MEMORY_SNAPSHOT_POINTER'])
    cp = Path(os.environ['CARRYON_MEMORY_CONTRACT_PATH'])
    ch = os.environ['CARRYON_MEMORY_CONTRACT_SHA256']
    root, report = load_snapshot(pointer, cp, ch)
    manifest = read(Path(resource('row_manifest')))
    row = resolve_manifest_row(manifest, Path(row_root).name)
    require(sha(root / 'PARAMETER.jsonl') == row['initial_memory']['PARAMETER']['sha256'], 'P exact frozen experiment snapshot mismatch')
    require(sha(root / 'SEMANTIC.jsonl') == row['initial_memory']['SEMANTIC']['sha256'], 'S exact admitted snapshot mismatch')
    require(row['snapshot_contract'] == {'path': str(cp), 'sha256': ch}, 'row contract mismatch')
    require(row['reflection_and_writeback'] is False, 'writeback enabled')
    installed = collect(inner, pointer, cp, ch, Path(row_root) / 'OBJECT_BATCH_COLLECTION.jsonl')
    B = installed['backend_class']
    old = B.__init__

    @functools.wraps(old)
    def init(self, *a, **kw):
        old(self, *a, **kw)
        installed['checked']()
        require(isinstance(self.jam_z_reflector, installed['P']) and isinstance(self.semantic_reflector, installed['S']), 'collector factory mismatch')
    B.__init__ = init
    inner._memory_consumer_install = installed
    return installed
