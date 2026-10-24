"""Explicit external-resource configuration for relocated experiment runners."""
from __future__ import annotations
import json,os
from pathlib import Path

def path(name: str) -> Path:
    env_name='CARRYON_RESOURCE_'+name.upper()
    direct=os.environ.get(env_name)
    if direct:return Path(direct).expanduser()
    config=os.environ.get('CARRYON_RESOURCES')
    if config:
        config_path=Path(config).expanduser().resolve()
        values=json.loads(config_path.read_text(encoding='utf-8-sig'))
        if name in values and values[name]:
            p=Path(values[name]).expanduser()
            return p if p.is_absolute() else config_path.parent/p
    raise RuntimeError(f'External resource {name!r} is not configured. Set {env_name} or CARRYON_RESOURCES.')
