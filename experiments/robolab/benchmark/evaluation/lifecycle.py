"""Reset episode memory and record lifecycle events."""
from __future__ import annotations
import json
import shutil
import time
from pathlib import Path


def reset_memory(mem_dir: Path, *, arm: str, batch: str, repeat: int,
                 cell: str, reason: str) -> dict:
    """Start an episode with an empty memory store."""
    if mem_dir.exists():
        shutil.rmtree(mem_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    if any(mem_dir.iterdir()):
        raise RuntimeError("Memory directory is not empty after reset")
    return {"event": "memory_reset", "ts": time.time(), "cell": cell,
            "arm": arm, "batch": batch, "repeat": repeat, "reason": reason,
            "memory_dir": str(mem_dir)}


def append_event(ledger: Path, event: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
