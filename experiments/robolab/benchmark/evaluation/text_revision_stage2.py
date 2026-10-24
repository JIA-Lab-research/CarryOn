#!/usr/bin/env python3
'Run the second allocation stage of text scene-memory revision.'
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import shutil
import subprocess
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

STAGE = 2
ORDINAL_BASE = 192          # Continues the first-stage repeat sequence.
OTHER_RUNNERS = ("run_text.py", "run_text_memory.py", "run_image.py",
                 "text_revision_stage1.py", "text_revision_stage2.py",
                 "image_revision_stage1.py", "image_revision_stage2.py")


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


def _other_batch_running(lane=None) -> list:
    """Find other runners in the same lane, excluding this process and its ancestors."""
    mine, cur = set(), os.getpid()
    while cur and cur not in mine:
        mine.add(cur)
        try:
            stat = pathlib.Path(f"/proc/{cur}/stat").read_text()
            cur = int(stat[stat.rindex(")") + 2:].split()[1])
        except Exception:
            break
    hits = []
    for d in pathlib.Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in mine:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode(
                "utf-8", "replace").replace(chr(0), " ").strip()
        except Exception:
            continue
        if any(n in cmd for n in OTHER_RUNNERS):
            mine_lane = f"--lane {lane}" in cmd if lane is not None else "--lane" not in cmd
            if not mine_lane:
                continue
            hits.append(f"{d.name}: {cmd[:120]}")
    return hits


_LANE = None


def preflight(alloc: list, v4_rows: list) -> dict:
    """Check pair counts, allocation balance, configuration IDs and lane availability."""
    pairs = [(a, arm) for a in alloc for arm in ARMS]
    per_arm = collections.Counter(arm for _, arm in pairs)
    per_cell_arm = collections.Counter((a["cell"], arm) for a, arm in pairs)

    # Merge with the first-stage design, counted per arm.
    merged = collections.Counter()
    for r in v4_rows:
        if r["k"] == 1:                  # one row per pair
            merged[(r["cell"], r["stale_station"], r["arm"])] += 1
    for a, arm in pairs:
        merged[(a["cell"], a["stale_station"], arm)] += 1
    layers, bad_layers = {}, []
    cells = sorted({a["cell"] for a in alloc})
    for cell in cells:
        for arm in ARMS:
            got = sorted((v for (c, s, ar), v in merged.items()
                          if c == cell and ar == arm), reverse=True)
            layers[f"{cell}|{arm}"] = got
            if got != [4, 4, 3]:
                bad_layers.append((cell, arm, got))
    # Check the prescribed rotation for each repeat.
    rot_bad = [a for a in alloc
               if a["rotation_index"] != {3: 0, 4: 1, 5: 2, 6: 0, 7: 1}[a["repeat"]]]

    v4_ids = {r["config_id"] for r in v4_rows}
    stray = sorted({a["config_id"] for a in alloc} - v4_ids)
    others = _other_batch_running(_LANE)

    checks = {
        "1_pair_counts": {
            "new_pairs": len(pairs), "expected": 80,
            "per_arm": dict(per_arm), "expected_per_arm": 40,
            "per_cell_per_arm": sorted(set(per_cell_arm.values())),
            "expected_per_cell_per_arm": 5,
            "ok": (len(pairs) == 80 and set(per_arm.values()) == {40}
                   and set(per_cell_arm.values()) == {5})},
        "2_merged_layers_443": {
            "layers": layers, "violations": bad_layers,
            "rotation_violations": rot_bad,
            "ok": not bad_layers and not rot_bad},
        "3_config_ids_subset_of_v4": {
            "v4_config_ids": len(v4_ids), "stray": stray,
            "ok": not stray and len(v4_ids) == 24},
        "4_no_other_batch": {"found": others, "ok": not others},
    }
    checks["all_green"] = all(v["ok"] for k, v in checks.items()
                              if k != "all_green")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(RoboLab / "targets.json"))
    ap.add_argument("--config", default=str(RoboLab / "planner.yaml"))
    ap.add_argument("--seed-table", default=str(RoboLab / "protocol" / "revision_stage1.json"))
    ap.add_argument("--alloc-table", default=str(RoboLab / "protocol" / "revision_stage2.json"))
    ap.add_argument("--v4-manifest", default=str(REPO / "logs" / "robolab" /
                                                "text_revision_stage1" / "RUN_MANIFEST.json"))
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "text_revision_stage2"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--lane", type=int, default=None,
                    help="run in an isolated lane: own port, memory, logs, config")
    a = ap.parse_args()
    lane = lane_module.setup(a.lane, Path(a.config))
    global _LANE
    _LANE = a.lane
    a.config = str(lane.config)

    seed_path = Path(a.seed_table)
    table = json.loads(seed_path.read_text())
    configs = table["configs"]
    specs = json.loads(Path(a.spec).read_text())["specs"]

    alloc_path = Path(a.alloc_table)
    alloc_doc = json.loads(alloc_path.read_text())
    alloc = alloc_doc["allocations"]

    v4_path = Path(a.v4_manifest)
    v4_rows = json.loads(v4_path.read_text())["rows"]

    gate = preflight(alloc, v4_rows)
    print("Checking revision allocation:")
    for name in sorted(k for k in gate if k != "all_green"):
        print(f"  {name:28} {'PASS' if gate[name]['ok'] else 'RED'}")
    if not gate["all_green"]:
        (Path(a.out)).mkdir(parents=True, exist_ok=True)
        red = Path(a.out) / "PREFLIGHT_RED.json"
        red.write_text(json.dumps(gate, indent=2, sort_keys=True, default=str)
                       + chr(10), encoding="utf-8")
        raise SystemExit(f"PREFLIGHT_RED: nothing ran; report at {red}")

    # cell-major, then repeat, arms alternating -- one frame-server repoint
    # Per cell, matching the first stage.
    rows, o = [], ORDINAL_BASE
    for al in sorted(alloc, key=lambda x: (x["cell"], x["repeat"])):
        for arm in ARMS:
            for k in (1, 2):
                o += 1
                rows.append({"batch": "M-stale-v5", "stage": STAGE,
                             "config_id": al["config_id"],
                             "cell": al["cell"], "arm": arm,
                             "repeat": al["repeat"],
                             "k": k, "ordinal": o,
                             "stale_station": al["stale_station"],
                             "expected_endpoint": al["expected_endpoint"],
                             "tag": f"{al['config_id']}_{arm}_r{al['repeat']}_k{k}"})

    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = {
                   'lane': lane.describe(),
                   'schema': 'robolab-mstale-stage2-manifest/v1',
                   'namespace': 'carryon-robolab-text-revision-stage2',
                   'stage': STAGE,
                   'alloc_table': alloc_doc,
                   'preflight_section4': gate,
                   'ordinal_scheme': f'continues stage 1: this batch is {ORDINAL_BASE + 1}-{ORDINAL_BASE + 160}',
                   'pairs_this_stage': len(rows) // 2,
                   'pairs_pooled_per_arm': 88,
                   'pairs_pooled_both_arms': 176,
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
    print(f"cells/pairs/eps : {len(set(al['cell'] for al in alloc))}/"
          f"{len(rows)//2}/{len(rows)}")
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
