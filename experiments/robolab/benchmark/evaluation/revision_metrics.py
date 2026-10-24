#!/usr/bin/env python3
'Shared paired metrics for scene-memory revision.'
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

ITEMS = ("looked", "memory_corrected", "k1_dispatch_correct", "k2_direct_correct")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def newcombe(k1: int, n1: int, k2: int, n2: int) -> list:
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = k1 / n1 - k2 / n2
    lo = d - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2)
    hi = d + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)
    return [round(100 * lo, 2), round(100 * hi, 2)]


def q7(xs: list, q: float) -> float:
    if not xs:
        return float("nan")
    a = sorted(xs)
    h = (len(a) - 1) * q
    lo, hi = math.floor(h), math.ceil(h)
    return a[lo] + (h - lo) * (a[hi] - a[lo])


def describe(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "median": None, "iqr": None}
    return {"n": len(xs), "median": round(q7(xs, 0.5), 4),
            "iqr": [round(q7(xs, 0.25), 4), round(q7(xs, 0.75), 4)]}


def pair_items(k1: dict, k2: dict) -> dict:
    """The four pre-registered items, computed off one pair."""
    scene = k1.get("scene_after") or {}
    exp, stale = k1["expected_endpoint"], k1["stale_station"]
    noun_at_exp = str(scene.get(exp, ""))
    # corrected means BOTH: written at the right place AND gone from the stale
    # one. An entry that is only added, never removed, is what sends the next
    # episode back to the wrong station.
    target_words = k1["cell"].replace("_", " ").split()[-1]
    corrected = (target_words in noun_at_exp.lower()
                 or bool(noun_at_exp)) and (stale not in scene)
    return {
        "looked": bool(k1.get("checked_endpoints")),
        "first_checked_is_stale": bool(k1.get("first_checked_is_stale")),
        "memory_corrected": bool(corrected),
        "k1_dispatch_correct": k1.get("dispatched_arm_endpoint") == exp,
        "k2_direct_correct": (k2 is not None
                              and k2.get("search_invocations") == 0
                              and k2.get("dispatched_arm_endpoint") == exp),
        "reached_without_looking": bool(k1.get("reached_without_looking")),
    }
