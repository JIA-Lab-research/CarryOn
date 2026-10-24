#!/usr/bin/env python3
'Evaluate text requests with scene memory.'
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RoboLab = REPO / "evaluation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RoboLab))

from episode import (TARGET_NOUN, _ts, m_manifest_text, run_episode,  # noqa: E402
                       serve_cell, window)
from lifecycle import append_event, reset_memory  # noqa: E402
import lane as lane_module  # noqa: E402
from scene_memory import scene_entries  # noqa: E402
from carryon.logging_util import read_event_log  # noqa: E402


def guard_records(tlog: Path, t0: float, t1: float) -> list[dict]:
    if not tlog.is_file():
        return []
    return [r for r in read_event_log(str(tlog))
            if r.get("verb") == "lookguard" and t0 <= _ts(r["ts"]) <= t1 + 5.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(RoboLab / "targets.json"))
    ap.add_argument("--config", default=str(RoboLab / "planner.yaml"))
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "text_memory"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--lane", type=int, default=None,
                    help="run in an isolated lane: own port, memory, logs, config")
    a = ap.parse_args()
    lane = lane_module.setup(a.lane, Path(a.config))
    a.config = str(lane.config)

    specs = json.loads(Path(a.spec).read_text())["specs"]
    rows, o = [], 0
    for cell in sorted(specs):
        for rep in range(1, a.repeats + 1):
            o += 1
            rows.append({"batch": "main", "cell": cell, "arm": "M",
                         "generation": "guarded", "repeat": rep, "ordinal": o,
                         "tag": f"{cell}_Mg_r{rep}"})

    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = {
                   'lane': lane.describe(),
                   'schema': 'robolab-m88-manifest/v1',
                   'namespace': 'carryon-robolab-text-memory',
                   'n_cells': len(specs),
                   'repeats': a.repeats,
                   'n_episodes': len(rows),
                   'rows': rows
               }
    mpath = out / "RUN_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10),
                     encoding="utf-8")
    print(f"namespace       : {manifest['namespace']}")
    print(f"cells/repeats/eps: {len(specs)}/{a.repeats}/{len(rows)}")
    if a.manifest_only:
        return

    ledger = out / "reset_ledger.jsonl"
    results = out / "episodes.jsonl"
    mem = lane.mem
    tlog = lane.tlog

    served, errs, resets, ran = None, 0, 0, 0
    for r in rows:
        cell = r["cell"]
        spec = specs[cell]
        if served != cell:
            serve_cell(cell, out / 'frame_server.log', port=lane.port)
            served = cell

        ev = reset_memory(mem, arm="M", batch="main", repeat=r["repeat"],
                          cell=cell, reason=f"ordinal {r['ordinal']}")
        append_event(ledger, ev)
        resets += 1
        (mem / "scene.md").write_text(m_manifest_text(spec), encoding="utf-8")

        if tlog.exists():
            tlog.unlink()
        ep = run_episode(Path(a.config), spec["query"], r["tag"], out,
                         a.timeout_s, a.quiet_s)
        ep.update(window(tlog, ep["t_start"], ep["t_end"]))
        guard = guard_records(tlog, ep["t_start"], ep["t_end"])


        transcript = (out / "transcripts" / f"{r['tag']}.txt").read_text(
            encoding="utf-8", errors="replace")
        row = dict(r)
        row.update(ep)
        row.update({
                       'instruction': spec['query'],
                       'expected_endpoint': spec['host_endpoint'],
                       'kind': spec['kind'],
                       'scene_after': scene_entries(mem / 'scene.md'),
                       'lookguard_fired': sum((1 for g in guard if g.get('lookguard_fired'))),
                       'lookguard_corrected_replan_ok': sum((1 for g in guard if g.get('lookguard_corrected_replan_ok'))),
                       'lookguard_trigger_station': sorted({g.get('lookguard_trigger_station') for g in guard} - {None}),
                       'truncation_suspect': bool(not ep['plan_done'] and (not transcript.strip()) and (a.quiet_s + 20 <= ep['wall_s'] <= a.quiet_s + 40))
                   })
        contract = ("cell", "arm", "repeat", "instruction",
                    "dispatched_arm_endpoint", "time_to_dispatch_s",
                    "search_invocations", "checked_endpoints", "writes_agent",

                    "plan_done", "lookguard_fired", "scene_after")
        missing = [f for f in contract if f not in row]
        if row["dispatched_arm_endpoint"] is not None and row["time_to_dispatch_s"] is None:
            missing.append("time_to_dispatch_s")
        if row["search_invocations"] is None:
            missing.append("search_invocations")
        row["contract_missing"] = missing
        row["plan_error"] = "[plan error]" in transcript
        row["instrument_error"] = (bool(missing) or ep["agent_died_early"]
                                   or bool(row.get("truncation_suspect")))
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + chr(10))
        ran += 1
        if row["instrument_error"]:
            errs += 1
            print(f"[STOP] instrument error at {r['tag']}: "
                  f"{missing or 'agent died early'}", flush=True)
            break
        if ran % 11 == 0:
            print(f"[progress] ran={ran}/{len(rows)} instrument_errors={errs} "
                  f"resets_witnessed={resets}", flush=True)

    print(f"[done] ran={ran} instrument_errors={errs} resets_witnessed={resets}")


if __name__ == "__main__":
    main()
