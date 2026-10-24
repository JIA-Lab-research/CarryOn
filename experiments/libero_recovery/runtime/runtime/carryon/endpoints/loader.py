from __future__ import annotations
import os
from typing import Any
from .model import RobotEndpoint
from .registry import EndpointRegistry
_REQUIRED = ('name', 'transport', 'address')

def _skill_library(item: dict[str, Any]) -> list[str]:
    raw = item.get('skill_library') or []
    if not isinstance(raw, list) or not all((isinstance(s, str) for s in raw)):
        name = item.get('name') or '<unknown>'
        raise ValueError(f'endpoint {name}: skill_library must be a list of strings')
    return list(raw)

def _build_endpoint(item: dict[str, Any]) -> RobotEndpoint:
    for k in _REQUIRED:
        if k not in item or not item[k]:
            raise ValueError(f'endpoint entry missing required field: {k}: {item!r}')
    known = {'name', 'transport', 'address', 'description', 'tags', 'skill_library', 'cancel_path', 'return_home_on_cancel'}
    extra = {k: v for k, v in item.items() if k not in known}
    raw_home = item.get('return_home_on_cancel', True)
    return RobotEndpoint(name=item['name'], transport=item['transport'], address=item['address'], description=item.get('description', '') or '', tags=list(item.get('tags') or []), skill_library=_skill_library(item), cancel_path=item.get('cancel_path', '/cancel') or '/cancel', return_home_on_cancel=bool(raw_home) if raw_home is not None else True, extra=extra)

def load_endpoints_yaml(path: str) -> EndpointRegistry:
    """Build an EndpointRegistry from a YAML file.

    File shape:
        endpoints:
          - name: ...
            transport: http
            address: ...
            description: ...
            tags: [...]
            skill_library: [...]
    """
    if not os.path.exists(path):
        return EndpointRegistry()
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    items = data.get('endpoints') or []
    reg = EndpointRegistry()
    for it in items:
        reg.register(_build_endpoint(it))
    return reg
