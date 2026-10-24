#!/usr/bin/env python3
'Analyse the four endpoint-selection conditions from recorded evaluation outputs.'
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu  # noqa: E402

from statistics_helpers import (ALPHA, BOOTSTRAP, SEED, acc, contrast,  # noqa: E402
                        describe, holm, wilson)

ARMS = ("T", "P", "M", "PM")
SOURCES = {"T": ("main", "T"), "M": ("main", "M"), "P": ("img", "P"), "PM": ("img", "PM")}
PER_ARM = 88


def read(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def manifest_of(p: Path) -> dict:
    m = p.parent / "RUN_MANIFEST.json"
    if not m.is_file():
        return {"missing": str(m)}
    d = json.loads(m.read_text(encoding="utf-8"))
    lane = d.get("lane") or {}
    return {
               'path': str(m),
               'namespace': 'carryon-robolab-analyse-main',
               'schema': d.get('schema'),
               'lane': lane.get('lane'),
               'callback_port': lane.get('callback_port'),
               'frame_port': lane.get('port')
           }


def cell_kinds(rows: list) -> dict:
    kinds = {r["cell"]: r["kind"] for r in rows if r.get("kind")}
    missing = {r["cell"] for r in rows} - set(kinds)
    if missing:
        import cells  # cell table, field `kind`
        table = getattr(cells, "CELLS")
        for c in sorted(missing):
            if isinstance(table, dict):
                kinds[c] = table[c]["kind"] if isinstance(table[c], dict) else table[c].kind
            else:
                spec = next(x for x in table if (x.get("key") if isinstance(x, dict) else x.key) == c)
                kinds[c] = spec["kind"] if isinstance(spec, dict) else spec.kind
    return kinds


def arm_stats(rows: list, kinds: dict, cells: list) -> dict:
    k, n = acc(rows)
    disp = [r for r in rows if r.get("dispatched_arm_endpoint") is not None]
    out = {"correct": k, "valid": n,
           "wilson95": [round(x, 4) for x in wilson(k, n)] if n else None,
           "no_dispatch": n - len(disp),
           "mis_dispatch": sum(1 for r in disp if not (r.get("dispatched_arm_endpoint") == r.get("expected_endpoint"))),
           "plan_error": sum(1 for r in rows if r.get("plan_error")),
           "writes_agent_total": sum(r.get("writes_agent") or 0 for r in rows),
           "by_kind": {}, "per_cell": {}}
    for kind in ("base", "distractor"):
        rs = [r for r in rows if kinds.get(r["cell"]) == kind]
        kk, nn = acc(rs)
        out["by_kind"][kind] = {"correct": kk, "valid": nn,
                                "wilson95": [round(x, 4) for x in wilson(kk, nn)] if nn else None}
    for c in cells:
        out["per_cell"][c] = list(acc([r for r in rows if r["cell"] == c]))
    out["efficiency"] = {
        "search_invocations": describe([r["search_invocations"] for r in rows
                                        if r.get("search_invocations") is not None]),
        "checked_endpoints_len": describe([len(r.get("checked_endpoints") or []) for r in rows]),
        "time_to_dispatch_s_dispatched_only": describe(
            [r["time_to_dispatch_s"] for r in disp if r.get("time_to_dispatch_s") is not None]),
        "n_dispatched": len(disp)}
    if rows and "planner_saw_image" in rows[0]:
        out["planner_saw_image_false"] = sum(1 for r in rows if r.get("planner_saw_image") is False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-t", required=True, help="text/episodes.jsonl (T only)")
    ap.add_argument("--episodes-m", required=True, help="text_memory/episodes.jsonl")
    ap.add_argument("--episodes-img", required=True, help="image/episodes.jsonl (P and PM)")
    ap.add_argument("--preset", choices=("smoke", "paper"), default="paper")
    ap.add_argument("--out", default=None, help="JSON output; a .txt report is written beside it")
    a = ap.parse_args()

    files = {"T": Path(a.episodes_t), "M": Path(a.episodes_m)}
    if a.episodes_img:
        files["P"] = files["PM"] = Path(a.episodes_img)
    raw = {p: read(p) for p in set(files.values())}
    prov = {arm: manifest_of(p) for arm, p in files.items()}

    from cells import CELLS, STATIONS
    specifications = {r['key']: r for r in CELLS}
    repeats = 1 if a.preset == 'smoke' else 11
    expected = {(arm, cell, rep) for arm in ARMS for cell in specifications
                for rep in range(1, repeats + 1)}
    seen, per_arm_rows, errs, extra = set(), {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}, {}
    for path, rows in raw.items():
        allowed = {arm for arm, file in files.items() if file == path}
        for row in rows:
            arm, cell, rep = row.get('arm'), row.get('cell'), row.get('repeat')
            key = (arm, cell, rep)
            if type(rep) is not int or key not in expected or arm not in allowed:
                raise ValueError('Unexpected endpoint-selection request: ' + str(key))
            if key in seen:
                raise ValueError('Duplicate endpoint-selection request: ' + str(key))
            seen.add(key)
            if row.get('batch') != SOURCES[arm][0]:
                raise ValueError('Conflicting request stage: ' + str(key))
            if row.get('expected_endpoint') != STATIONS[specifications[cell]['host']]['endpoint']:
                raise ValueError('Conflicting target endpoint: ' + str(key))
            if row.get('instrument_error') or row.get('service_error'):
                errs[arm].append(row)
            else:
                per_arm_rows[arm].append(row)
    missing = expected - seen
    if missing:
        raise ValueError('Missing endpoint-selection requests: ' + str(sorted(missing)))
    if any(errs.values()):
        report = {'status': 'incomplete', 'preset': a.preset, 'service_errors': errs,
                  'expected_requests': len(expected), 'recorded_requests': len(seen)}
        text = json.dumps(report, indent=2) + '\n'
        if a.out:
            Path(a.out).write_text(text, encoding='utf-8')
        print(text)
        raise SystemExit(1)
    present = [arm for arm in ARMS if per_arm_rows.get(arm)]
    all_valid = [r for arm in present for r in per_arm_rows[arm]]
    kinds = cell_kinds(all_valid)
    cells = sorted(kinds)

    s81 = {arm: arm_stats(per_arm_rows[arm], kinds, cells) for arm in present}
    for arm in present:
        s81[arm]["instrument_error_episodes"] = errs[arm]

    def have(*arms):
        return all(x in present for x in arms)

    # ---- 8.2 text family: Fisher + Holm, unchanged -------------------------
    text_family = [(x, y) for x, y in (("M", "T"), ("M", "P"), ("P", "T")) if have(x, y)]
    c82 = [contrast(per_arm_rows[x], per_arm_rows[y], x, y) for x, y in text_family]
    adj = holm([(c["contrast"], c["p_raw"]) for c in c82]) if c82 else {}
    for c in c82:
        c["p_holm"] = adj[c["contrast"]]
        c["significant"] = adj[c["contrast"]] < ALPHA
    # ---- image-row contrasts: Newcombe only, no p --------------------------
    img_family = [(x, y) for x, y in (("PM", "P"), ("PM", "M"), ("PM", "T")) if have(x, y)]
    c_img = []
    for x, y in img_family:
        c = contrast(per_arm_rows[x], per_arm_rows[y], x, y)
        c.pop("p_raw", None)
        c["no_p_value"] = True
        c_img.append(c)

    # ---- 8.3 strata, unchanged --------------------------------------------
    def stratum(arm, kind):
        return [r for r in per_arm_rows[arm] if kinds[r["cell"]] == kind]

    def gap(arm, rowset):
        kb, nb = acc([r for r in rowset if r["arm"] == arm and kinds[r["cell"]] == "base"])
        kd, nd = acc([r for r in rowset if r["arm"] == arm and kinds[r["cell"]] == "distractor"])
        if not nb or not nd:
            return float("nan")
        return kb / nb - kd / nd

    rng = np.random.default_rng(SEED)

    def boot_gap_diff(arm_a, arm_b):
        by_cell = {c: [r for r in all_valid if r["cell"] == c] for c in cells}
        out = []
        for _ in range(BOOTSTRAP):
            sample = []
            for c in cells:
                pool = by_cell[c]
                if pool:
                    idx = rng.integers(0, len(pool), len(pool))
                    sample.extend(pool[i] for i in idx)
            out.append(gap(arm_a, sample) - gap(arm_b, sample))
        arr = np.asarray([x for x in out if not math.isnan(x)])
        if not arr.size:
            return None
        return {"point": round(gap(arm_a, all_valid) - gap(arm_b, all_valid), 4),
                "ci95": [round(float(np.percentile(arr, 2.5)), 4),
                         round(float(np.percentile(arr, 97.5)), 4)],
                "n_bootstrap": int(arr.size),
                "reading": "support" if float(np.percentile(arr, 2.5)) > 0 else "not support"}

    s83 = {"strata_source": "cells.py field `kind`",
           "base_cells": sorted(c for c in cells if kinds[c] == "base"),
           "distractor_cells": sorted(c for c in cells if kinds[c] == "distractor"),
           "gap_definition": "base accuracy - distractor accuracy",
           "gap": {arm: round(gap(arm, all_valid), 4) for arm in present}}
    if have("P", "T") and stratum("P", "distractor") and stratum("T", "distractor"):
        s83["prediction_P_equals_T_in_distractor"] = {
            **contrast(stratum("P", "distractor"), stratum("T", "distractor"), "P", "T"),
            "falsified_if": "P>T and Holm-adjusted p<0.05 within this family"}
    fam = []
    for arm in ("T", "P"):
        if arm in present:
            kb, nb = acc(stratum(arm, "base"))
            kd, nd = acc(stratum(arm, "distractor"))
            if not nb or not nd:
                continue
            _, p = fisher_exact([[kb, nb - kb], [kd, nd - kd]], alternative="two-sided")
            fam.append((f"{arm}_base_vs_distractor", p))
    adj_fam = holm(fam) if fam else {}
    s83["prediction_TP_lower_on_distractor"] = [
        {"test": name, "p_raw": p, "p_holm": adj_fam[name], "significant": adj_fam[name] < ALPHA}
        for name, p in fam]
    if have("T", "M"):
        s83["gapT_minus_gapM"] = boot_gap_diff("T", "M")
    if have("P", "M"):
        s83["gapP_minus_gapM"] = boot_gap_diff("P", "M")
    s83["no_p_values_for_bootstrap"] = True
    s83["seed"] = SEED

    # ---- 8.4 efficiency, unchanged (M vs T), image row descriptive ---------
    def eff(arm, field, dispatched_only):
        rs = per_arm_rows[arm]
        if dispatched_only:
            rs = [r for r in rs if r.get("dispatched_arm_endpoint") is not None]
        return [r.get(field) for r in rs if r.get(field) is not None]

    s84 = {}
    for field, only in (("search_invocations", False), ("time_to_dispatch_s", True)):
        entry = {"per_arm": {arm: describe(eff(arm, field, only)) for arm in present},
                 "na_note": "time_to_dispatch_s is NA when no dispatch occurred; N per arm above" if only else None}
        if have("M", "T"):
            x, y = eff("M", field, only), eff("T", field, only)
            entry["M_vs_T_mannwhitney_p_raw"] = (mannwhitneyu(x, y, alternative="two-sided").pvalue
                                                if x and y else float("nan"))
        s84[field] = entry
    if have("M", "T"):
        adj_eff = holm([(f, s84[f]["M_vs_T_mannwhitney_p_raw"]) for f in s84])
        for f in s84:
            s84[f]["M_vs_T_p_holm"] = adj_eff[f]
            s84[f]["significant"] = adj_eff[f] < ALPHA

    out = {
              'schema': 'robolab-main-table-analysis/v1',
              'status': 'completed',
              'preset': a.preset,
              'per_arm_rows_expected': len(expected)//4,
              'partial': False,
              'arms_present': present,
              'provenance': prov,
              'unexpected_batches_or_arms': extra,
              'alpha': ALPHA,
              '8.1': s81,
              '8.2_text_family': c82,
              '8.2_image_row': c_img,
              '8.3': s83,
              '8.4': s84,
              'software': {'python': platform.python_version(), 'scipy': scipy.__version__, 'numpy': np.__version__}
          }
    text = json.dumps(out, indent=2, sort_keys=True, default=str) + "\n"

    # ---- readable report --------------------------------------------
    L = []
    L.append(f"RoboLab endpoint selection  {a.preset}")
    L.append("arm   correct        acc%   wilson95           no_disp  mis_disp  plan_err  lane/cb")
    for arm in present:
        s = s81[arm]
        pct = 100 * s["correct"] / s["valid"] if s["valid"] else float("nan")
        w = s["wilson95"] or [float("nan")] * 2
        pv = prov[arm]
        L.append(f"{arm:4}  {s['correct']:3}/{s['valid']:<3}   {pct:6.1f}   [{100*w[0]:5.1f}, {100*w[1]:5.1f}]"
                 f"     {s['no_dispatch']:3}      {s['mis_dispatch']:3}      {s['plan_error']:3}     {pv.get('lane')}/{pv.get('callback_port')}")
    L.append("")
    L.append("by cell kind:")
    for arm in present:
        for kind in ("base", "distractor"):
            b = s81[arm]["by_kind"][kind]
            if b["valid"]:
                L.append(f"  {arm:3} {kind:10} {b['correct']:3}/{b['valid']:<3} = {100*b['correct']/b['valid']:5.1f}%")
    L.append("")
    L.append("efficiency (time on dispatched episodes only):")
    for arm in present:
        e = s81[arm]["efficiency"]
        si, ti = e["search_invocations"], e["time_to_dispatch_s_dispatched_only"]
        L.append(f"  {arm:3} checks median={si['median']} IQR={si['iqr']}   time median={ti['median']}s IQR={ti['iqr']}  n={ti['n']}")
    L.append("")
    L.append("contrasts (pp, Newcombe 95%):")
    for c in c82:
        L.append(f"  {c['contrast']:6} {c['diff_pp']:+6.2f}  {c['diff_newcombe95_pp']}   Fisher p={c['p_raw']:.4g}  Holm p={c['p_holm']:.4g}  {'sig' if c['significant'] else 'ns'}")
    for c in c_img:
        L.append(f"  {c['contrast']:6} {c['diff_pp']:+6.2f}  {c['diff_newcombe95_pp']}   (no p)")
    L.append("")
    L.append("per-cell accuracy:")
    L.append("  cell            " + "".join(f"{arm:>8}" for arm in present))
    for c in cells:
        L.append(f"  {c:15} " + "".join(f"{s81[arm]['per_cell'][c][0]:3}/{s81[arm]['per_cell'][c][1]:<4}" for arm in present))
    L.append("")
    L.append("service errors: " + json.dumps({arm: len(errs[arm]) for arm in present}))
    report = chr(10).join(L) + chr(10)

    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        rp = Path(a.out).with_suffix(".txt")
        rp.write_text(report, encoding="utf-8")
        print(f"wrote {a.out} and {rp}")
    print(report)


if __name__ == "__main__":
    main()
