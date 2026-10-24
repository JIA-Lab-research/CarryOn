"""Map object hints to open-vocabulary segmentation phrases.

The appearance-catalog approach follows RATs libero_catalog. Phrase priority
is reflector sam_text_prompts, then the catalog, then the raw object hint."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _norm(hint: str) -> str:
    return " ".join(str(hint or "").strip().lower().replace("_", " ").split())


class ObjectCatalog:
    """Read-only phrase lookup; missing file or hint fails open (empty list)."""

    def __init__(self, mapping: dict[str, list[str]]):
        self._map = {_norm(k): [str(v) for v in vs] for k, vs in mapping.items()}

    def prompts_for(self, object_hint: str) -> list[str]:
        return list(self._map.get(_norm(object_hint), []))

    def __len__(self) -> int:
        return len(self._map)


def load_object_catalog(path: str) -> Optional[ObjectCatalog]:
    if not str(path or "").strip():
        return None
    f = Path(path)
    if not f.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - catalog is an enhancement, fail open
        return None
    mapping = {
        k: [str(x) for x in v if str(x).strip()]
        for k, v in data.items()
        if isinstance(v, (list, tuple))
    }
    return ObjectCatalog(mapping) if mapping else None
