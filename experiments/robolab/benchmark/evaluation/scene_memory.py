"""Read the scene-memory inventory used by RoboLab episodes."""
from pathlib import Path


def scene_entries(scene_md: Path) -> dict:
    """`## place` -> its objects line, as the planner would read it."""
    if not scene_md.is_file():
        return {}
    out, cur = {}, None
    for ln in scene_md.read_text(encoding="utf-8").splitlines():
        if ln.startswith("## "):
            cur = ln[3:].strip()
            out[cur] = ""
        elif cur and ln.strip().startswith("- objects:"):
            out[cur] = ln.split(":", 1)[1].strip()
    return out
