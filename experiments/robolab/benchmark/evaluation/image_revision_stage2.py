#!/usr/bin/env python3
'Run the second allocation stage of image scene-memory revision.'
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
import collections  # noqa: E402
from image_requests import (ENTITY, NEUTRAL, QUERY_PM, REMEMBERED_RE,  # noqa: E402
                    canonical_for, planner_saw_image, profile_has_reference,
                    run_episode_img, serve_neutral)


STAGE = 2
ORDINAL_BASE = 192
ROTATION = {3: 0, 4: 1, 5: 2, 6: 0, 7: 1}
OTHER_RUNNERS = ("run_text.py", "run_text_memory.py", "run_image.py",
                 "text_revision_stage1.py", "text_revision_stage2.py",
                 "image_revision_stage1.py", "image_revision_stage2.py")
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


def _other_batch_running(lane=None) -> list:
    """Any other RoboLab runner alive; reads /proc and skips this process's whole
    ancestor chain, since the launcher's command line names this script."""
    mine, cur = set(), os.getpid()
    while cur and cur not in mine:
        mine.add(cur)
        try:
            stat = Path(f"/proc/{cur}/stat").read_text()
            cur = int(stat[stat.rindex(")") + 2:].split()[1])
        except Exception:
            break
    hits = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in mine:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().decode("utf-8", "replace") \
                .replace(chr(0), " ").strip()
        except Exception:
            continue
        if any(n in cmd for n in OTHER_RUNNERS):
            mine_lane = f"--lane {lane}" in cmd if lane is not None else "--lane" not in cmd
            if not mine_lane:
                continue
            hits.append(f"{d.name}: {cmd[:120]}")
    return hits


_LANE = None


def preflight(alloc: list, stage1_rows: list) -> dict:
    """The four assertions the text row's stage 2 ran; a red and nothing runs."""
    pairs = [(a, arm) for a in alloc for arm in ARMS]
    per_arm = collections.Counter(arm for _, arm in pairs)
    per_cell_arm = collections.Counter((a["cell"], arm) for a, arm in pairs)
    merged = collections.Counter()
    for r in stage1_rows:
        if r["k"] == 1:
            merged[(r["cell"], r["stale_station"], r["arm"])] += 1
    for a, arm in pairs:
        merged[(a["cell"], a["stale_station"], arm)] += 1
    layers, bad = {}, []
    for cell in sorted({a["cell"] for a in alloc}):
        for arm in ARMS:
            got = sorted((v for (c, s, ar), v in merged.items()
                          if c == cell and ar == arm), reverse=True)
            layers[f"{cell}|{arm}"] = got
            if got != [4, 4, 3]:
                bad.append((cell, arm, got))
    rot_bad = [a for a in alloc if a["rotation_index"] != ROTATION[a["repeat"]]]
    s1_ids = {r["config_id"] for r in stage1_rows}
    stray = sorted({a["config_id"] for a in alloc} - s1_ids)
    others = _other_batch_running(_LANE)
    checks = {
        "1_pair_counts": {"new_pairs": len(pairs), "per_arm": dict(per_arm),
                          "per_cell_per_arm": sorted(set(per_cell_arm.values())),
                          "ok": len(pairs) == 80 and set(per_arm.values()) == {40}
                                and set(per_cell_arm.values()) == {5}},
        "2_merged_layers_443": {"layers": layers, "violations": bad,
                                "rotation_violations": rot_bad,
                                "ok": not bad and not rot_bad},
        "3_config_ids_subset_of_stage1": {"stage1_config_ids": len(s1_ids),
                                          "stray": stray,
                                          "ok": not stray and len(s1_ids) == 24},
        "4_no_other_batch": {"found": others, "ok": not others},
    }
    checks["all_green"] = all(v["ok"] for k, v in checks.items() if k != "all_green")
    return checks


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
    ap.add_argument("--alloc-table", default=str(RoboLab / "protocol" / "revision_stage2.json"))
    ap.add_argument("--stage1-manifest", default=str(REPO / "logs" / "robolab" /
                                                    "image_revision_stage1" / "RUN_MANIFEST.json"))
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "image_revision_stage2"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--configs", default=None, help="comma-separated configuration subset")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--resume-from-ordinal", type=int, default=None,
                    help="skip rows below this ordinal; append to the existing "
                         "out dir; the declared manifest is verified, not rewritten")
    ap.add_argument("--rerun-pair", action="append", default=[],
                    help="config_id|arm|repeat of a pair to run AGAIN at the end, "
                         "both halves, with fresh ordinals and rerun=true. The "
                         "pair is the unit: k=2 inherits k=1's memory, so a "
                         "truncated k=2 cannot be re-run alone. The original "
                         "rows stay on disk, flagged, and are superseded at "
                         "analysis time by the rerun.")
    ap.add_argument("--exclude-pair", action="append", default=[],
                    help="config_id|arm|repeat of a pair excluded on resume "
                         "(listed in RESUME.json; its k rows are not re-run)")
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

    alloc_path = Path(a.alloc_table)
    alloc_doc = json.loads(alloc_path.read_text())
    alloc = alloc_doc["allocations"]
    if a.configs:
        want = {c.strip() for c in a.configs.split(",") if c.strip()}
        alloc = [x for x in alloc if x["config_id"] in want]

    s1_path = Path(a.stage1_manifest)
    stage1_rows = json.loads(s1_path.read_text())["rows"]

    gate = preflight(alloc, stage1_rows) if not a.configs else {"all_green": True,
                                                                "skipped": "subset run"}
    print("preflight (four assertions):")
    for name in sorted(k for k in gate if k not in ("all_green", "skipped")):
        print(f"  {name:32} {'PASS' if gate[name]['ok'] else 'RED'}")
    if not gate["all_green"]:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        red = Path(a.out) / "PREFLIGHT_RED.json"
        red.write_text(json.dumps(gate, indent=2, sort_keys=True, default=str) + "\n")
        raise SystemExit(f"PREFLIGHT_RED: nothing ran; report at {red}")

    rows, o = [], ORDINAL_BASE
    for al in sorted(alloc, key=lambda x: (x["cell"], x["repeat"])):
        for arm in ARMS:
            for k in (1, 2):
                o += 1
                rows.append({"batch": "image-revision", "stage": STAGE,
                             "config_id": al["config_id"],
                             "cell": al["cell"], "arm": arm, "repeat": al["repeat"],
                             "k": k, "ordinal": o,
                             "stale_station": al["stale_station"],
                             "expected_endpoint": al["expected_endpoint"],
                             "tag": f"{al['config_id']}_{arm}_r{al['repeat']}_k{k}"})

    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = {
                   'lane': lane.describe(),
                   'schema': 'robolab-img-stale-stage2-manifest/v1',
                   'namespace': 'carryon-robolab-image-revision-stage2',
                   'stage': STAGE,
                   'alloc_table': alloc_doc,
                   'preflight': gate,
                   'ordinal_scheme': f'continues stage 1: {ORDINAL_BASE + 1}-{ORDINAL_BASE + 160}',
                   'pairs_this_stage': len(rows) // 2,
                   'pairs_pooled_per_arm': 88,
                   'pairs_pooled_both_arms': 176,
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
    if a.resume_from_ordinal is not None:
        # A resume is not a new design. The manifest that was declared before
        # the first episode stays byte-identical; anything about the resume is
        # recorded beside it, never inside it.
        declared = json.loads(mpath.read_text())["rows"] if mpath.is_file() else None
        if declared is None or [ (r["tag"]) for r in declared ] != [r["tag"] for r in rows]:
            raise SystemExit("RESUME_REFUSED: rows on disk differ from the plan")
        excluded = set(a.exclude_pair)
        done_tags = set()
        if (out / "episodes.jsonl").is_file():
            done_tags = {json.loads(l)["tag"] for l in
                         (out / "episodes.jsonl").read_text().splitlines() if l.strip()}
        rows = [r for r in rows
                if r["ordinal"] >= a.resume_from_ordinal
                and f"{r['config_id']}|{r['arm']}|{r['repeat']}" not in excluded
                and r["tag"] not in done_tags]
        # Re-runs go LAST, after the planned rows, with ordinals past the plan,
        # so the resumed sequence is the pre-registered one and the re-run is
        # visibly an addition rather than a substitution slipped into place.
        reruns, next_ord = [], ORDINAL_BASE + 160
        by_key = {(x["config_id"], x["repeat"]): x for x in alloc}
        for spec_ in a.rerun_pair:
            cid, arm, rp = spec_.split("|")
            al = by_key[(cid, int(rp))]
            orig = [json.loads(l)["ordinal"] for l in
                    (out / "episodes.jsonl").read_text().splitlines() if l.strip()
                    and json.loads(l)["config_id"] == cid
                    and json.loads(l)["arm"] == arm
                    and json.loads(l)["repeat"] == int(rp)]
            for k in (1, 2):
                next_ord += 1
                reruns.append({"batch": "image-revision", "stage": STAGE,
                               "config_id": cid, "cell": al["cell"], "arm": arm,
                               "repeat": int(rp), "k": k, "ordinal": next_ord,
                               "stale_station": al["stale_station"],
                               "expected_endpoint": al["expected_endpoint"],
                               "rerun": True, "supersedes_ordinals": sorted(orig),
                               "tag": f"{cid}_{arm}_r{rp}_k{k}_rerun"})
        rows = rows + reruns
        (out / "RESUME.json").write_text(json.dumps({
                                                        'resume_from_ordinal': a.resume_from_ordinal,
                                                        'excluded_pairs': sorted(excluded),
                                                        'rerun_pairs': sorted(a.rerun_pair),
                                                        'rerun_rows': [r['tag'] for r in reruns],
                                                        'rows_already_on_disk': len(done_tags),
                                                        'rows_to_run': [r['tag'] for r in rows]
                                                    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"RESUME: {len(done_tags)} rows on disk; {len(rows)} rows to run")
    else:
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(f"namespace       : {manifest['namespace']}")
    print(f"cells/pairs/eps : {len({r['cell'] for r in rows})}/{len(rows)//2}/{len(rows)}")
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
