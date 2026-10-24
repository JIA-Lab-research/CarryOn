#!/usr/bin/env python3
'Analyse scene-memory revision across both allocation stages.'
from __future__ import annotations

import argparse
import collections
import json
import platform
import re
import unicodedata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_metrics import describe, newcombe, pair_items, wilson  # noqa: E402

ITEMS = ("looked", "memory_corrected", "k1_dispatch_correct",
         "k2_direct_correct", "k2_dispatch_correct")
ARMS = ("N", "H")
TARGET_NOUN = {"hammer": "red-handled hammer", "bagel": "bagel",
               "measuring_cup": "measuring cup", "keyboard": "keyboard",
               "banana": "banana", "rubiks": "rubiks cube",
               "birdhouse": "birdhouse", "mustard": "mustard"}


def _inventory_text(value: object) -> str:
    """Normalize spelling punctuation without substituting object identities."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("'", "").replace("\u2019", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def target_presence(inventory: object, noun: str) -> bool | None:
    """Read a target from an objects inventory; uncertainty earns no credit.

    Matching uses complete normalized words (including Rubik's/rubiks), not
    arbitrary nonempty text. Explicit absence is accepted as a retraction.
    Conflicting or uncertain target statements remain unresolved.
    """
    target = _inventory_text(noun)
    if not target:
        raise ValueError("Target noun must be nonempty")
    pattern = re.compile(r"(?<!\w)" + re.escape(target) + r"s?(?!\w)")
    mentions = []
    for item in re.split(r"[,;\n.!?]|\bbut\b", str(inventory or ""), flags=re.I):
        text = _inventory_text(item)
        for match in pattern.finditer(text):
            before, after = text[:match.start()], text[match.end():]
            uncertain = re.search(r"\b(maybe|possibly|perhaps|might|may|unknown|uncertain)\b", text)
            negative = (re.search(r"\b(no|not|without|absent|missing|removed|absence of)\b", before)
                        or re.match(r"\s+(?:(?:is|was|are|were)\s+)?(?:not|absent|missing|removed|gone)\b", after))
            mentions.append(None if uncertain else not bool(negative))
    if not mentions:
        return False
    if None in mentions or len(set(mentions)) > 1:
        return None
    return mentions[0]


def score_pair(k1: dict, k2: dict) -> dict:
    """Score correction of both the current and stale scene-memory entries."""
    d = pair_items(k1, k2)
    scene = k1.get("scene_after") or {}
    exp, stale = k1["expected_endpoint"], k1["stale_station"]
    noun = TARGET_NOUN[k1["cell"]].lower()
    at_exp = target_presence(scene.get(exp), noun) is True
    retracted = target_presence(scene.get(stale), noun) is False
    d["memory_corrected"] = bool(at_exp and retracted)
    d["k2_dispatch_correct"] = (k2 is not None and
                                k2.get("dispatched_arm_endpoint") == k2.get("expected_endpoint"))
    return d


def read(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def expected_rows(preset, stage):
    protocol = Path(__file__).parent / "protocol"
    if stage == 2:
        if preset != "paper":
            raise ValueError("Smoke has no second allocation stage")
        allocations = json.loads((protocol / "revision_stage2.json").read_text())["allocations"]
    else:
        configs = json.loads((protocol / "revision_stage1.json").read_text())["configs"]
        if preset == "smoke":
            selected, seen = [], set()
            for row in configs:
                if row["expected_endpoint"] not in seen:
                    selected.append(row); seen.add(row["expected_endpoint"])
            configs = selected
        allocations = [dict(row, repeat=repeat) for row in configs
                       for repeat in range(1, 2 if preset == "smoke" else 3)]
    return {(r["config_id"], arm, r["repeat"], k): r
            for r in allocations for arm in ARMS for k in (1, 2)}


def pairs_of(rows: list, expected=None) -> tuple:
    if not rows:
        raise ValueError("Empty revision results")
    acc, identities = {}, set()
    for row in rows:
        key = (row["config_id"], row["arm"], row["repeat"], row["k"])
        if type(row["k"]) is not int or row["k"] not in (1, 2) or row["arm"] not in ARMS:
            raise ValueError("Invalid revision arm or request index: " + str(key))
        if type(row["repeat"]) is not int or row["repeat"] < 1:
            raise ValueError("Invalid revision repeat: " + str(key))
        if key in identities:
            raise ValueError("Duplicate revision request: " + str(key))
        identities.add(key)
        if expected is not None:
            if key not in expected:
                raise ValueError("Unexpected revision request: " + str(key))
            for field in ("cell", "expected_endpoint", "stale_station"):
                if row[field] != expected[key][field]:
                    raise ValueError("Revision case differs: " + str(key) + ", " + field)
        acc.setdefault(key[:3], {})[row["k"]] = row
    if expected is not None and identities != set(expected):
        raise ValueError("Missing revision requests: " + str(sorted(set(expected) - identities)))
    for key, pair in acc.items():
        if set(pair) != {1, 2}:
            raise ValueError("Incomplete revision pair: " + str(key))
        for field in ("cell", "expected_endpoint", "stale_station"):
            if pair[1][field] != pair[2][field]:
                raise ValueError("Conflicting pair identity: " + str(key))
    errors = [r for r in rows if r.get("instrument_error")]
    return acc, errors, rows


def summarise(comp: dict, arm: str) -> dict:
    vs = [v for (c, ar, rp), v in sorted(comp.items()) if ar == arm]
    items = [score_pair(v[1], v[2]) for v in vs]
    n = len(items)
    out = {"n_pairs": n}
    for it in ITEMS:
        k = sum(1 for i in items if i[it])
        out[it] = {"count": k, "of": n, "wilson95": wilson(k, n)}
    k1 = [v[1] for v in vs]
    out["k1_split_correct_wrong_none"] = [
        sum(1 for r in k1 if r["dispatched_arm_endpoint"] == r["expected_endpoint"]),
        sum(1 for r in k1 if r["dispatched_arm_endpoint"] not in (None, r["expected_endpoint"])),
        sum(1 for r in k1 if r["dispatched_arm_endpoint"] is None)]
    out["writes_agent_k1"] = describe([r.get("writes_agent") or 0 for r in k1])
    out["k1_with_any_planner_write"] = sum(1 for r in k1 if (r.get("writes_agent") or 0) > 0)
    out["plan_error_k1"] = sum(1 for r in k1 if r.get("plan_error"))
    out["plan_error_k2"] = sum(1 for v in vs if v[2].get("plan_error"))
    out["guard_fired_k1"] = sum(1 for r in k1 if (r.get("lookguard_fired") or 0) > 0)
    out["guard_fired_k2"] = sum(1 for v in vs if (v[2].get("lookguard_fired") or 0) > 0)
    out["checks_k1"] = describe([len(r.get("checked_endpoints") or []) for r in k1])
    out["time_k1"] = describe([r["time_to_dispatch_s"] for r in k1
                               if r["dispatched_arm_endpoint"] is not None])
    out["time_k2"] = describe([v[2]["time_to_dispatch_s"] for v in vs
                               if v[2]["dispatched_arm_endpoint"] is not None])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    s1, e1, _ = pairs_of(read(Path(a.stage1)), expected_rows(a.preset, 1))
    stages = {"stage1": s1}
    errs = {"stage1": [r["tag"] for r in e1]}
    if a.stage2:
        s2, e2, _ = pairs_of(read(Path(a.stage2)), expected_rows(a.preset, 2))
        for k in s2:
            if k in s1:
                raise ValueError(f"Stage 2 collides with stage 1 at {k}")
        stages["stage2"] = s2
        stages["pooled"] = {**s1, **s2}
        errs["stage2"] = [r["tag"] for r in e2]
    if any(errs.values()):
        report = json.dumps({"status": "incomplete", "instrument_errors": errs,
                             "reason": "Service or instrument errors; no performance aggregate produced"}, indent=2)
        if a.out:
            Path(a.out).write_text(report + "\n", encoding="utf-8")
        print(report)
        raise SystemExit(1)
    by_stage = {name: {arm: summarise(c, arm) for arm in ARMS} for name, c in stages.items()}
    primary = "pooled" if "pooled" in stages else "stage1"
    diffs = {}
    for it in ITEMS:
        h, n = by_stage[primary]["H"][it], by_stage[primary]["N"][it]
        diffs[it] = {"H_minus_N_pp": round(100 * (h["count"] / h["of"] - n["count"] / n["of"]), 2)
                     if h["of"] and n["of"] else None,
                     "newcombe95_pp": newcombe(h["count"], h["of"], n["count"], n["of"]),
                     "no_p_value": True}
    out = {"schema": "carryon-robolab-revision/v1", "status": "completed", "preset": a.preset,
           "item2_test": "confirmed station holds the target AND the stale station "
                         "has no entry or no longer lists the target (scoring may use "
                         "the seed noun; the mechanism does not)",
           "primary": primary, "no_p_values": True,
           "pairs_complete": {k: len(v) for k, v in stages.items()},
           "instrument_errors": errs,
           "by_stage": by_stage, "arm_difference_H_minus_N": diffs,
           "software": {"python": platform.python_version()}}
    text = json.dumps(out, indent=2, sort_keys=True, default=str) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
