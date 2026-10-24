#!/usr/bin/env python3
'Shared image-query, object-identity and memory-condition helpers.'
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RoboLab = REPO / "evaluation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RoboLab))

from episode import m_manifest_text, serve_cell, window  # noqa: E402
from lifecycle import append_event, reset_memory  # noqa: E402

CANON = RoboLab / "reference_canonical"
ARMS = ("P", "PM")
# One sentence for every cell: the text names nothing.
QUERY_P = "go pick up this"
QUERY_PM = "go pick up mine"
TURN1_PM = "this is mine"                   # semantics; issued as the command below
ENTITY = "mine"
NEUTRAL = "reference.png"                   # the basename the prompt will echo
REMEMBERED_RE = re.compile(r"\[remember_image\] stored:")
DONE_RE = re.compile(r"\[plan-[0-9a-f]+ done\]")


# Detect upstream service failures in the transcript so a request with no
# dispatch is not classified as the planner asking for clarification.
VLM_ERROR_RE = re.compile(
    r"\[error\].*?(planner VLM returned (?P<code>\d{3})|PlannerVLMError"
    r"|RateLimit|Timeout|ConnectionError)", re.IGNORECASE)


def vlm_error_in(transcript: str) -> "str | None":
    m = VLM_ERROR_RE.search(transcript or "")
    if not m:
        return None
    return m.group("code") or m.group(1)


def canonical_for(spec: dict) -> Path:
    return CANON / Path(spec["reference_canonical_path"]).name


def serve_neutral(spec: dict, ref_dir: Path) -> tuple[Path, str]:
    """Copy this cell's reference image to a fixed, meaningless filename.

    The CLI replaces an @image token with "[attached image: <basename>]", and
    that marker is text the planner reads. Serving bagel.png would hand the
    model the word the picture exists to replace.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    dst = ref_dir / NEUTRAL
    src = canonical_for(spec)
    shutil.copyfile(src, dst)
    return dst


def _spawn(cfg: Path):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "carryon",
         "--config", str(cfg)],
        cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        start_new_session=True, close_fds=True)
    q: "queue.Queue" = queue.Queue()

    def _rd():
        for ln in proc.stdout:
            q.put(ln)
        q.put(None)

    threading.Thread(target=_rd, daemon=True).start()
    return proc, q


def _drain(q, lines, deadline, quiet_s, stop_re=None):
    """Read until stop_re matches, the stream ends, or it goes quiet."""
    hit = False
    while time.time() < deadline:
        try:
            ln = q.get(timeout=quiet_s)
        except queue.Empty:
            break
        if ln is None:
            break
        lines.append(ln.rstrip("\n"))
        if stop_re is not None and stop_re.search(ln):
            hit = True
            break
        if stop_re is None and DONE_RE.search(ln):
            hit = True
    return hit


def run_episode_img(cfg: Path, arm: str, turns: list, tag: str, out: Path,
                    timeout_s: float, quiet_s: float) -> dict:
    """One episode; retried once if the agent dies before printing anything."""
    ep = None
    for attempt in (1, 2, 3):
        ep = _run_episode_img_once(cfg, arm, turns, tag, out, timeout_s, quiet_s)
        ep["spawn_retries"] = attempt - 1
        if not (ep["agent_died_early"] and ep["transcript_empty"]):
            return ep
    return ep


def _run_episode_img_once(cfg: Path, arm: str, turns: list, tag: str, out: Path,
                          timeout_s: float, quiet_s: float) -> dict:
    """One episode. For PM the clock starts at TURN 2.

    PM's turn 1 only files an image; timing it alongside M's single turn would
    compare a two-turn wall clock against a one-turn one and call the
    difference a property of memory.
    """
    proc, q = _spawn(cfg)
    lines: list[str] = []
    t0 = time.time()
    deadline = t0 + timeout_s
    turn1_ok = None
    done = False
    try:
        if arm == "PM":
            proc.stdin.write(turns[0] + "\n")
            proc.stdin.flush()
            # A definite signal, not a silence: the command echoes on success.
            turn1_ok = _drain(q, lines, min(deadline, time.time() + 60.0),
                              10.0, stop_re=REMEMBERED_RE)
        t_measure = time.time()
        proc.stdin.write(turns[-1] + "\n")
        proc.stdin.flush()
        done = _drain(q, lines, deadline, quiet_s)
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
        proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
    t1 = time.time()
    (out / "transcripts" / f"{tag}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return {"plan_done": done, "wall_s": round(t1 - t0, 3),
            "t_start": t_measure, "t_end": t1,
            "t_process_start": t0, "turn1_ok": turn1_ok,
            "agent_returncode": proc.returncode,
            "transcript_empty": not any(l.strip() for l in lines),
            "agent_died_early": (t1 - t_measure) < 10.0 and not done}


def profile_has_reference(mem: Path) -> tuple[bool, str | None]:
    """Did turn 1 actually put a readable reference image in the profile?"""
    for md in sorted((mem / "user").glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        if f"## entity:{ENTITY}" not in text:
            continue
        for line in text.splitlines():
            if line.strip().startswith("- reference_image:"):
                p = line.split(":", 1)[1].strip()
                # the store writes a path relative to the agent's cwd; resolve
                # it the same way rather than against whoever runs this check
                full = p if os.path.isabs(p) else os.path.join(str(REPO), p)
                return os.path.isfile(full), p
    return False, None


def planner_saw_image(plog: Path, offset: int) -> tuple[bool, int]:
    """Return whether requests recorded since offset contain an image part."""
    if not plog.is_file():
        return False, offset
    size = plog.stat().st_size
    if size <= offset:
        return False, size
    with plog.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        chunk = fh.read()
    return ('"image_url"' in chunk), size
