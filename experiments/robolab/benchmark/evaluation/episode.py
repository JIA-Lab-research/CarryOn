#!/usr/bin/env python3
'Shared endpoint selection episodes, scene setup and request windows.'
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RoboLab = REPO / "evaluation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RoboLab))

from carryon.logging_util import read_event_log  # noqa: E402

import cells  # noqa: E402
from lifecycle import append_event, reset_memory
from scene_memory import scene_entries  # noqa: E402
from targets import noun  # noqa: E402


STATION_ENDPOINT = {k: v["endpoint"] for k, v in cells.STATIONS.items()}
STATIONS = tuple(sorted(STATION_ENDPOINT.values()))
DONE_RE = re.compile(r"\[(plan-[0-9a-f]+) done\]")
PORT = 18121


def _ts(s: str) -> float:
    head, _, ms = s.partition(".")
    return time.mktime(time.strptime(head, "%d-%m-%Y %H:%M:%S")) + (
        int(ms) / 1000.0 if ms else 0.0)


def serve_cell(cell: str, log: Path, port: int = PORT):
    from portable import serve_cell_owned
    return serve_cell_owned(cell, log, port)


def run_episode(cfg: Path, instruction: str, tag: str, out: Path,
                timeout_s: float, quiet_s: float) -> dict:
    """One episode; retried once if the agent dies before printing anything."""
    ep = None
    for attempt in (1, 2, 3):
        ep = _run_episode_once(cfg, instruction, tag, out, timeout_s, quiet_s)
        ep["spawn_retries"] = attempt - 1
        if not (ep["agent_died_early"] and ep["transcript_empty"]):
            return ep
    return ep


def _run_episode_once(cfg: Path, instruction: str, tag: str, out: Path,
                      timeout_s: float, quiet_s: float) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
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
    lines, done = [], False
    try:
        proc.stdin.write(instruction + chr(10))
        proc.stdin.flush()
        while time.time() - t0 < timeout_s:
            try:
                ln = q.get(timeout=quiet_s)
            except queue.Empty:
                break
            if ln is None:
                break
            lines.append(ln.rstrip(chr(10)))
            if DONE_RE.search(ln):
                done = True
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
        chr(10).join(lines) + chr(10), encoding="utf-8")
    return {"plan_done": done, "wall_s": round(t1 - t0, 3),
            "t_start": t0, "t_end": t1,
            "agent_returncode": proc.returncode,
            "transcript_empty": not any(l.strip() for l in lines),
            "agent_died_early": (t1 - t0) < 10.0 and not done}


def window(tlog: Path, t0: float, t1: float) -> dict:
    if not tlog.is_file():
        return {"dispatched_arm_endpoint": None, "time_to_dispatch_s": None,
                "search_invocations": None, "checked_endpoints": None,
                "writes_agent": None}
    recs = read_event_log(str(tlog))
    for r in recs:
        r["_t"] = _ts(r["ts"])
    recs = sorted((r for r in recs if t0 <= r["_t"] <= t1 + 5.0),
                  key=lambda r: r["_t"])
    checks, dispatch, writes = [], None, 0
    for r in recs:
        if r.get("verb") != "transport_submit_request":
            continue
        ep = r.get("endpoint")
        if ep == "check_endpoint":
            checks.append(r)
        elif ep == "update_scene":
            writes += 1
        elif ep in STATIONS and dispatch is None:
            dispatch = r
    return {
        "dispatched_arm_endpoint": dispatch.get("endpoint") if dispatch else None,
        "time_to_dispatch_s": (round(dispatch["_t"] - t0, 3) if dispatch else None),
        "search_invocations": sum(1 for c in checks
                                  if dispatch is None or c["_t"] <= dispatch["_t"]),
        "checked_endpoints": [(c.get("inputs") or {}).get("endpoint") for c in checks],
        "writes_agent": writes,
    }


def m_manifest_text(spec: dict) -> str:
    out = []
    for station, prims in sorted(spec["m_manifest"].items()):
        out.append(f"## {STATION_ENDPOINT[station]}")
        out.append("- objects: " + ", ".join(noun(p) for p in prims))
    return chr(10).join(out) + chr(10)


def stale_text(spec: dict, target_noun: str) -> tuple:
    """An entry naming a station that does NOT hold the target."""
    host = spec["host_endpoint"]
    wrong = sorted(s for s in STATIONS if s != host)[0]
    return wrong, f"## {wrong}{chr(10)}- objects: {target_noun}{chr(10)}"


TARGET_NOUN = {"hammer": "red-handled hammer", "bagel": "bagel",
               "measuring_cup": "measuring cup", "keyboard": "keyboard",
               "banana": "banana", "rubiks": "rubiks cube",
               "birdhouse": "birdhouse", "mustard": "mustard"}
