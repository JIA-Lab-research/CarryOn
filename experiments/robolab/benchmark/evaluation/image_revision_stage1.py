#!/usr/bin/env python3
'Run the first allocation stage of image scene-memory revision.'
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RoboLab = REPO / "evaluation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RoboLab))

from episode import _ts, serve_cell, window  # noqa: E402
from lifecycle import append_event, reset_memory  # noqa: E402
import lane as lane_module  # noqa: E402
from scene_memory import scene_entries  # noqa: E402
from carryon.logging_util import read_event_log  # noqa: E402
from image_requests import (ENTITY, NEUTRAL, QUERY_PM, REMEMBERED_RE,  # noqa: E402
                    canonical_for, planner_saw_image, profile_has_reference,
                    run_episode_img, serve_neutral)


ARMS = ("N", "H")
# The correction hint with its noun phrase replaced by the same pronoun the query
# uses -- otherwise the hint hands back the word the picture is there to give.
HINT_PM = ("go pick up mine. If what you find there is not mine, "
           "keep looking at the other stations.")
TARGET_NOUN = {"hammer": "red-handled hammer", "bagel": "bagel",
               "measuring_cup": "measuring cup", "keyboard": "keyboard",
               "banana": "banana", "rubiks": "rubiks cube",
               "birdhouse": "birdhouse", "mustard": "mustard"}


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
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "img_stale"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--configs", default=None, help="comma-separated configuration subset")
    ap.add_argument("--manifest-only", action="store_true")
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

    missing = [c["cell"] for c in configs
               if not canonical_for(specs[c["cell"]]).is_file()]
    if missing:
        raise SystemExit(f"REFERENCE_IMAGES_MISSING on this host: {sorted(set(missing))}")

    rows, o = [], 0
    for cfg in configs:
        for rep in range(1, a.repeats + 1):
            for arm in ARMS:
                for k in (1, 2):
                    o += 1
                    rows.append({"batch": "image-revision", "config_id": cfg["config_id"],
                                 "cell": cfg["cell"], "arm": arm, "repeat": rep,
                                 "k": k, "ordinal": o,
                                 "stale_station": cfg["stale_station"],
                                 "expected_endpoint": cfg["expected_endpoint"],
                                 "tag": f"{cfg['config_id']}_{arm}_r{rep}_k{k}"})

    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = {
                   'lane': lane.describe(),
                   'schema': 'robolab-img-stale-manifest/v1',
                   'namespace': 'carryon-robolab-image-revision-stage1',
                   'seed_table': table,
                   'what_is_stale': 'the remembered LOCATION; the reference image is always correct and says what, never where',
                   'arms': {'N': {'k1': QUERY_PM}, 'H': {'k1': HINT_PM}},
                   'k2_query': QUERY_PM,
                   'text_never_names_the_target': True,
                   'neutral_attachment_basename': NEUTRAL,
                   'guard_note': 'the guard matches memory nouns against the SUBTASK instruction the planner writes from the image; its firing rate here is measured, not assumed',
                   'n_configs': len(configs),
                   'n_pairs': len(rows) // 2,
                   'n_episodes': len(rows),
                   'rows': rows
               }
    mpath = out / "RUN_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    print(f"namespace       : {manifest['namespace']}")
    print(f"configs/pairs/eps: {len(configs)}/{len(rows)//2}/{len(rows)}")
    print(f"N k1            : {QUERY_PM!r}")
    print(f"H k1            : {HINT_PM!r}")
    if a.manifest_only:
        return

    ledger = out / "reset_ledger.jsonl"
    results = out / "episodes.jsonl"
    mem = lane.mem
    tlog = lane.tlog
    plog = lane.plog
    ref_dir = out / "ref"

    served, errs, resets, ran = None, 0, 0, 0
    for r in rows:
        cell, k, arm = r["cell"], r["k"], r["arm"]
        spec = specs[cell]
        if served != cell:
            serve_cell(cell, out / 'frame_server.log', port=lane.port)
            served = cell

        if k == 1:
            ev = reset_memory(mem, arm=arm, batch="image-revision", repeat=r["repeat"],
                              cell=cell,
                              reason=f"pair {r['config_id']} {arm} r{r['repeat']}")
            append_event(ledger, ev)
            resets += 1
            ref_path = serve_neutral(spec, ref_dir)
            # the stale part: a LOCATION pointing at the wrong station
            (mem / "scene.md").write_text(
                f"## {r['stale_station']}\n"
                f"- objects: {TARGET_NOUN[cell]}\n", encoding="utf-8")
            turns = [f'/remember_image "{ref_path}" {ENTITY}',
                     QUERY_PM if arm == "N" else HINT_PM]
            ep_arm = "PM"                       # two-turn: file, then ask
        else:
            turns = [QUERY_PM]                  # inherits profile AND scene
            ep_arm = "P"                        # single-turn shape, no filing

        if tlog.exists():
            tlog.unlink()
        p_off = plog.stat().st_size if plog.is_file() else 0
        ep = run_episode_img(Path(a.config), ep_arm, turns, r["tag"], out,
                             a.timeout_s, a.quiet_s)
        ep.update(window(tlog, ep["t_start"], ep["t_end"]))
        guard = guard_records(tlog, ep["t_start"], ep["t_end"])
        saw_image, _ = planner_saw_image(plog, p_off)
        ref_ok, ref_stored = profile_has_reference(mem)

        transcript = (out / "transcripts" / f"{r['tag']}.txt").read_text(
            encoding="utf-8", errors="replace")
        row = dict(r)
        row.update(ep)
        checked = row.get("checked_endpoints") or []
        row.update({
                       'instruction': turns[-1],
                       'turns': turns,
                       'kind': spec['kind'],
                       'memory_seeded_at': r['stale_station'] if k == 1 else None,
                       'inherited_memory': k == 2,
                       'planner_saw_image': saw_image,
                       'profile_reference_ok': ref_ok,
                       'profile_reference_path': ref_stored,
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
                    "lookguard_fired", "planner_saw_image")
        missing_f = [f for f in contract if f not in row]
        if row["dispatched_arm_endpoint"] is not None and row["time_to_dispatch_s"] is None:
            missing_f.append("time_to_dispatch_s")
        if row["search_invocations"] is None:
            missing_f.append("search_invocations")
        truncated = bool(
            not ep["plan_done"] and not transcript.strip()
            and a.quiet_s + 20 <= ep["wall_s"] <= a.quiet_s + 40)
        row["vlm_error"] = vlm_error_in(transcript)
        row["contract_missing"] = missing_f
        row["truncation_suspect"] = truncated
        row["plan_error"] = "[plan error]" in transcript
        # An image arm without an image is not a weak arm, it is not the arm.
        row["image_gate_failed"] = (not saw_image) or (not ref_ok)
        row["instrument_error"] = (bool(missing_f) or ep["agent_died_early"]
                                   or row["vlm_error"] is not None
                                   or truncated or row["image_gate_failed"])
        with results.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        ran += 1
        if row["instrument_error"]:
            errs += 1
            why = (missing_f or (
                f"planner VLM error {row['vlm_error']}"
                if row["vlm_error"] else "no image reached the planner" if not saw_image
                                 else "no readable reference in the profile"
                                 if not ref_ok else
                                 "truncated turn" if truncated
                                 else "agent died early"))
            print(f"[STOP] instrument error at {r['tag']}: {why}", flush=True)
            break
        if ran % 24 == 0:
            print(f"[progress] ran={ran}/{len(rows)} instrument_errors={errs} "
                  f"resets_witnessed={resets}", flush=True)

    if mem.exists():
        shutil.rmtree(mem)
    mem.mkdir(parents=True, exist_ok=True)
    print(f"[done] ran={ran} instrument_errors={errs} resets_witnessed={resets}")


if __name__ == "__main__":
    main()
