#!/usr/bin/env python3
'Evaluate text requests without scene memory.'
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
from lifecycle import append_event, reset_memory  # noqa: E402
from scene_memory import scene_entries  # noqa: E402
import lane as lane_module  # noqa: E402


STATION_ENDPOINT = {k: v["endpoint"] for k, v in cells.STATIONS.items()}
STATIONS = tuple(sorted(STATION_ENDPOINT.values()))
DONE_RE = re.compile(r"\[(plan-[0-9a-f]+) done\]")
PORT = 18121
ARMS = ("T",)


def _ts(s: str) -> float:
    head, _, ms = s.partition(".")
    return time.mktime(time.strptime(head, "%d-%m-%Y %H:%M:%S")) + (
        int(ms) / 1000.0 if ms else 0.0)


def build_manifest(specs: dict, repeats: int) -> list:
    """Deterministic row order, fixed before anything runs."""
    rows = []
    for cell in sorted(specs):
        for arm in ARMS:
            for rep in range(1, repeats + 1):
                rows.append({"batch": "main", "cell": cell, "arm": arm,
                             "repeat": rep, "k": None})
    for i, r in enumerate(rows, 1):
        r["ordinal"] = i
        r["tag"] = (f"{r['cell']}_{r['arm']}_r{r['repeat']}" if r["k"] is None
                    else f"{r['cell']}_Mstale_k{r['k']}")
    return rows


def serve_cell(cell: str, log: Path, port: int = PORT):
    from portable import serve_cell_owned
    return serve_cell_owned(cell, log, port)


def run_episode(cfg: Path, instruction: str, tag: str, out: Path,
                timeout_s: float, quiet_s: float) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "carryon",
         "--config", str(cfg)],
        cwd=str(REPO), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
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
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
    t1 = time.time()
    (out / "transcripts" / f"{tag}.txt").write_text(
        chr(10).join(lines) + chr(10), encoding="utf-8")
    return {"plan_done": done, "wall_s": round(t1 - t0, 3),
            "t_start": t0, "t_end": t1,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(RoboLab / "targets.json"))
    ap.add_argument("--config", default=str(RoboLab / "planner.yaml"))
    ap.add_argument("--arms", choices=("T",), default="T", help="Text without memory; other conditions have separate runners")
    ap.add_argument("--lane", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "text"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--resume-from", type=int, default=1)
    a = ap.parse_args()
    global ARMS
    ARMS = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    lane = lane_module.setup(a.lane, Path(a.config))
    a.config = str(lane.config)

    specs = json.loads(Path(a.spec).read_text())["specs"]
    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    rows = build_manifest(specs, a.repeats)
    manifest = {
                   'lane': lane.describe(),
                   'arms': list(ARMS),
                   'schema': 'robolab-formal-manifest/v1',
                   'namespace': 'carryon-robolab-text',
                   'n_episodes': len(rows),
                   'rows': rows
               }
    mpath = out / "RUN_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10),
                     encoding="utf-8")
    print(f"namespace        : {manifest['namespace']}")
    print(f"episodes         : {len(rows)}")
    if a.manifest_only:
        return

    ledger = out / "reset_ledger.jsonl"
    results = out / "episodes.jsonl"
    mem = lane.mem
    tlog = lane.tlog

    served, errs, resets, ran = None, 0, 0, 0
    for r in rows:
        if r["ordinal"] < a.resume_from:
            continue
        cell, arm = r["cell"], r["arm"]
        spec = specs[cell]
        if served != cell:
            serve_cell(cell, out / 'frame_server.log', port=lane.port)
            served = cell
        ev = reset_memory(mem, arm=arm, batch=r["batch"], repeat=r["repeat"],
                          cell=cell, reason=f"ordinal {r['ordinal']}")
        append_event(ledger, ev)
        resets += 1
        inherit = False
        seeded_at = None
        instruction = spec["query"]

        if tlog.exists():
            tlog.unlink()
        ep = run_episode(Path(a.config), instruction, r["tag"], out,
                         a.timeout_s, a.quiet_s)
        ep.update(window(tlog, ep["t_start"], ep["t_end"]))


        row = dict(r)
        row.update(ep)
        row.update({
                       'instruction': instruction,
                       'expected_endpoint': spec['host_endpoint'],
                       'kind': spec['kind'],
                       'memory_seeded_at': seeded_at,
                       'inherited_memory': inherit,
                       'scene_after': scene_entries(mem / 'scene.md')
                   })
        contract = ("cell", "arm", "repeat", "instruction",
                    "dispatched_arm_endpoint", "time_to_dispatch_s",
                    "search_invocations", "checked_endpoints", "writes_agent",

                    "plan_done")
        missing = [f for f in contract if f not in row]
        # time_to_dispatch_s is legitimately null when nothing was dispatched
        if row["dispatched_arm_endpoint"] is not None and row["time_to_dispatch_s"] is None:
            missing.append("time_to_dispatch_s")
        if row["search_invocations"] is None:
            missing.append("search_invocations")
        row["contract_missing"] = missing
        row["plan_error"] = "[plan error]" in (out / "transcripts" / f"{r['tag']}.txt").read_text(encoding="utf-8", errors="replace")
        row["instrument_error"] = bool(missing) or ep["agent_died_early"]
        if row["instrument_error"]:
            errs += 1
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + chr(10))
        ran += 1

        if ran % 40 == 0 or r["ordinal"] == len(rows):
            # counts only -- blind until unblinding
            print(f"[progress] ran={ran}/{len(rows)} ordinal={r['ordinal']} "
                  f"instrument_errors={errs} resets_witnessed={resets}",
                  flush=True)

    print(f"[done] ran={ran} instrument_errors={errs} resets_witnessed={resets}")


TARGET_NOUN = {"hammer": "red-handled hammer", "bagel": "bagel",
               "measuring_cup": "measuring cup", "keyboard": "keyboard",
               "banana": "banana", "rubiks": "rubiks cube",
               "birdhouse": "birdhouse", "mustard": "mustard"}

if __name__ == "__main__":
    main()
