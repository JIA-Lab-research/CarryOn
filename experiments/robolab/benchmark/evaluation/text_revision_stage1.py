#!/usr/bin/env python3
'Run the first allocation stage of text scene-memory revision.'
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RoboLab = REPO / "evaluation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RoboLab))

from episode import run_episode, serve_cell, window  # noqa: E402
from lifecycle import append_event, reset_memory  # noqa: E402
import lane as lane_module  # noqa: E402
from episode import _ts  # noqa: E402
from carryon.logging_util import read_event_log  # noqa: E402
from scene_memory import scene_entries  # noqa: E402


PHRASE = {"bagel": "a bagel", "banana": "the banana",
          "birdhouse": "the birdhouse", "hammer": "the red-handled hammer",
          "keyboard": "the keyboard", "measuring_cup": "the measuring cup",
          "mustard": "the mustard", "rubiks": "the rubiks cube"}
HINT = ("go pick up {p}. If what you find there is not {p}, "
        "keep looking at the other stations.")
TARGET_NOUN = {"hammer": "red-handled hammer", "bagel": "bagel",
               "measuring_cup": "measuring cup", "keyboard": "keyboard",
               "banana": "banana", "rubiks": "rubiks cube",
               "birdhouse": "birdhouse", "mustard": "mustard"}
ARMS = ("N", "H")


def guard_records(tlog: Path, t0: float, t1: float) -> list:
    if not tlog.is_file():
        return []
    return [r for r in read_event_log(str(tlog))
            if r.get("verb") == "lookguard" and t0 <= _ts(r["ts"]) <= t1 + 5.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(RoboLab / "targets.json"))
    ap.add_argument("--config", default=str(RoboLab / "planner.yaml"))
    ap.add_argument("--seed-table", default=str(RoboLab / "protocol" / "revision_stage1.json"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "mstale_v4"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--configs", default=None, help="comma-separated configuration subset")
    ap.add_argument("--lane", type=int, default=None,
                    help="run in an isolated lane: own port, memory, logs, config")
    a = ap.parse_args()
    lane = lane_module.setup(a.lane, Path(a.config))
    a.config = str(lane.config)

    seed_path = Path(a.seed_table)
    table = json.loads(seed_path.read_text())
    configs = table["configs"]
    if a.configs:
        want = {c.strip() for c in a.configs.split(",") if c.strip()}
        configs = [c for c in configs if c["config_id"] in want]
        if not configs:
            raise SystemExit(f"no configs matched {sorted(want)}")
    specs = json.loads(Path(a.spec).read_text())["specs"]

    # config-major, arms alternating, repeat fixed in its slot
    rows, o = [], 0
    for cfg in configs:
        for rep in range(1, a.repeats + 1):
            for arm in ARMS:
                for k in (1, 2):
                    o += 1
                    rows.append({"batch": "M-stale-v5", "config_id": cfg["config_id"],
                                 "cell": cfg["cell"], "arm": arm, "repeat": rep,
                                 "k": k, "ordinal": o,
                                 "stale_station": cfg["stale_station"],
                                 "expected_endpoint": cfg["expected_endpoint"],
                                 "tag": f"{cfg['config_id']}_{arm}_r{rep}_k{k}"})

    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = {
                   'lane': lane.describe(),
                   'schema': 'robolab-mstale-v4-manifest/v1',
                   'namespace': 'carryon-robolab-text-revision-stage1',
                   'seed_table': table,
                   'n_configs': len(configs),
                   'n_pairs': len(rows) // 2,
                   'n_episodes': len(rows),
                   'arms': list(ARMS),
                   'rows': rows
               }
    mpath = out / "RUN_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10),
                     encoding="utf-8")
    print(f"namespace       : {manifest['namespace']}")
    print(f"configs/pairs/eps: {len(configs)}/{len(rows)//2}/{len(rows)}")
    if a.manifest_only:
        return

    ledger = out / "reset_ledger.jsonl"
    results = out / "episodes.jsonl"
    mem = lane.mem
    tlog = lane.tlog

    served, errs, resets, ran = None, 0, 0, 0
    for r in rows:
        cell, k, arm = r["cell"], r["k"], r["arm"]
        spec = specs[cell]
        if served != cell:
            serve_cell(cell, out / 'frame_server.log', port=lane.port)
            served = cell

        if k == 1:
            ev = reset_memory(mem, arm=arm, batch="M-stale-v5", repeat=r["repeat"],
                              cell=cell, reason=f"pair {r['config_id']} {arm} r{r['repeat']}")
            append_event(ledger, ev)
            resets += 1
            (mem / "scene.md").write_text(
                f"## {r['stale_station']}{chr(10)}"
                f"- objects: {TARGET_NOUN[cell]}{chr(10)}", encoding="utf-8")
            instruction = (spec["query"] if arm == "N"
                           else HINT.format(p=PHRASE[cell]))
        else:
            instruction = spec["query"]

        if tlog.exists():
            tlog.unlink()
        ep = run_episode(Path(a.config), instruction, r["tag"], out,
                         a.timeout_s, a.quiet_s)
        ep.update(window(tlog, ep["t_start"], ep["t_end"]))
        guard = guard_records(tlog, ep["t_start"], ep["t_end"])

        transcript = (out / "transcripts" / f"{r['tag']}.txt").read_text(
            encoding="utf-8", errors="replace")
        row = dict(r)
        row.update(ep)
        checked = row.get("checked_endpoints") or []
        row.update({
                       'instruction': instruction,
                       'kind': spec['kind'],
                       'memory_seeded_at': r['stale_station'] if k == 1 else None,
                       'inherited_memory': k == 2,
                       'scene_after': scene_entries(mem / 'scene.md'),
                       'looked': bool(checked),
                       'first_checked_is_stale': bool(checked) and checked[0] == r['stale_station'],
                       'reached_without_looking': not checked and row.get('dispatched_arm_endpoint') is not None,
                       'lookguard_fired': sum((1 for g in guard if g.get('lookguard_fired'))),
                       'lookguard_corrected_replan_ok': sum((1 for g in guard if g.get('lookguard_corrected_replan_ok'))),
                       'lookguard_trigger_station': sorted({g.get('lookguard_trigger_station') for g in guard} - {None})
                   })
        contract = ("config_id", "cell", "arm", "repeat", "k", "instruction",
                    "dispatched_arm_endpoint", "time_to_dispatch_s",
                    "search_invocations", "checked_endpoints", "writes_agent",
                    "memory_seeded_at", "inherited_memory",
                    "scene_after",  "plan_done",
                    "lookguard_fired")
        missing = [f for f in contract if f not in row]
        if row["dispatched_arm_endpoint"] is not None and row["time_to_dispatch_s"] is None:
            missing.append("time_to_dispatch_s")
        if row["search_invocations"] is None:
            missing.append("search_invocations")
        # Combine the termination indicators to distinguish a truncated request from a healthy long turn.
        truncated = bool(
            not ep["plan_done"] and not transcript.strip()
            and a.quiet_s + 20 <= ep["wall_s"] <= a.quiet_s + 40)
        row["contract_missing"] = missing
        row["truncation_suspect"] = truncated
        row["plan_error"] = "[plan error]" in transcript
        row["instrument_error"] = (bool(missing) or ep["agent_died_early"]
                                   or truncated)
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + chr(10))
        ran += 1
        if row["instrument_error"]:
            errs += 1
            why = (missing or ("truncated turn" if truncated
                               else "agent died early"))
            print(f"[STOP] instrument error at {r['tag']}: {why}", flush=True)
            break
        if ran % 24 == 0:
            print(f"[progress] ran={ran}/{len(rows)} instrument_errors={errs} "
                  f"resets_witnessed={resets}", flush=True)

    # the final clear, deliberately not logged: it closes the last pair and the
    # ledger holds one event per pair, not one per clear
    if mem.exists():
        shutil.rmtree(mem)
    mem.mkdir(parents=True, exist_ok=True)
    print(f"[done] ran={ran} instrument_errors={errs} resets_witnessed={resets}")


if __name__ == "__main__":
    main()
