"""Static waypoint manifest loader.

The manifest is the pre-mapped set of named poses the mobile base can reach
(shelves, the user/home spot, ...). It is environment *map* data -- "where each
shelf physically is" -- and is kept separate from SceneMemory, which records
"what was observed at each location". The Planner sees both: the manifest tells
it where it *can* go; scene memory tells it what is *there*.

YAML shape (``locations.yaml``)::

    locations:
      shelf_1: {x: 3.0, y: -1.5, yaw: 0.0, reach_endpoint: go2_navigator,
                description: "metal shelf by the window"}
      user:    {x: 0.0, y: 0.0, yaw: 0.0, reach_endpoint: go2_navigator}

Everything is optional; a missing file yields an empty manifest.
"""
from __future__ import annotations
import os
from typing import Any

def load_locations(path: str | None) -> dict[str, dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    raw = data.get('locations', data)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, fields in raw.items():
        if isinstance(fields, dict):
            out[str(name)] = dict(fields)
    return out

def render_locations_markdown(manifest: dict[str, dict[str, Any]]) -> str:
    """Compact markdown table-ish view for the Planner system prompt."""
    if not manifest:
        return ''
    lines: list[str] = []
    for name, fields in manifest.items():
        x = fields.get('x')
        y = fields.get('y')
        yaw = fields.get('yaw', 0.0)
        reach = fields.get('reach_endpoint', '')
        desc = fields.get('description', '')
        parts = [f'- {name}: (x={x}, y={y}, yaw={yaw})']
        if reach:
            parts.append(f'reach_endpoint={reach}')
        if desc:
            parts.append(f'-- {desc}')
        lines.append(' '.join(parts))
    return '\n'.join(lines)
__all__ = ['load_locations', 'render_locations_markdown']
