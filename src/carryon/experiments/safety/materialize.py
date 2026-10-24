"""Assemble the distributed Safety runtime in an independent directory."""
from pathlib import Path
import shutil
from ..files import bound_file, new_directory, read_json, write_json


def materialize(source, output):
    source = Path(source).resolve()
    records = read_json(source / 'SOURCE_INDEX.json')['files']
    if len({r['path'] for r in records}) != len(records):
        raise ValueError('Duplicate Safety source path')
    with new_directory(output, inputs=[source]) as root:
        for record in records:
            origin = bound_file(record, source)
            target = root / record['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
        write_json(root / 'RUNTIME.json', {'schema': 'carryon-safety-runtime/v1',
                                         'required_files': [r['path'] for r in records]})
    return runtime_files(output)


def runtime_files(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'RUNTIME.json')
    if manifest['schema'] != 'carryon-safety-runtime/v1':
        raise ValueError('Invalid Safety runtime')
    for relative in manifest['required_files']:
        bound_file({'path': relative}, root)
    return {'runtime': str(root), 'file_count': len(manifest['required_files'])}
