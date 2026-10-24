#!/usr/bin/env python3
'Statistical helpers for endpoint selection results.'
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import fisher_exact, mannwhitneyu

SEED = 20260902
BOOTSTRAP = 10000
ALPHA = 0.05
ARMS = ("T", "P", "M")


# --------------------------------------------------------------------------
# interval helpers
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval. Degenerate n=0 returns (nan, nan) rather than 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple:
    """Newcombe hybrid score interval for a difference of proportions."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = k1 / n1 - k2 / n2
    lo = d - math.sqrt((k1 / n1 - l1) ** 2 + (u2 - k2 / n2) ** 2)
    hi = d + math.sqrt((u1 - k1 / n1) ** 2 + (k2 / n2 - l2) ** 2)
    return (lo, hi)


def holm(pairs: list) -> dict:
    """Holm step-down. `pairs` is [(name, p)]; returns name -> adjusted p."""
    ordered = sorted(pairs, key=lambda x: x[1])
    m, out, running = len(ordered), {}, 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[name] = running
    return out


def quantile_type7(xs: list, q: float) -> float:
    """R's type-7 quantile, so the medians match the reported statistic."""
    if not xs:
        return float("nan")
    a = sorted(xs)
    h = (len(a) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    return a[lo] + (h - lo) * (a[hi] - a[lo])


def describe(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "median": None, "iqr": None, "range": None}
    return {"n": len(xs),
            "median": round(quantile_type7(xs, 0.5), 4),
            "iqr": [round(quantile_type7(xs, 0.25), 4),
                    round(quantile_type7(xs, 0.75), 4)],
            "range": [round(min(xs), 4), round(max(xs), 4)]}


# --------------------------------------------------------------------------
def load(path: Path) -> tuple:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    main = [r for r in rows if r["batch"] == "main"]
    stale = [r for r in rows if r["batch"] == "M-stale"]
    errs = [r for r in main if r.get("instrument_error")]
    valid = [r for r in main if not r.get("instrument_error")]
    return rows, main, stale, valid, errs


def correct(r: dict) -> bool:
    """no_dispatch counts as a failure, for the endpoint-selection metric."""
    return r.get("dispatched_arm_endpoint") == r.get("expected_endpoint")


def acc(rows: list) -> tuple:
    k = sum(1 for r in rows if correct(r))
    return k, len(rows)


def contrast(a_rows: list, b_rows: list, a: str, b: str) -> dict:
    ka, na = acc(a_rows)
    kb, nb = acc(b_rows)
    table = [[ka, na - ka], [kb, nb - kb]]
    _, p = fisher_exact(table, alternative="two-sided")
    lo, hi = newcombe(ka, na, kb, nb)
    return {"contrast": f"{a}-{b}", "table": table,
            f"{a}_acc": [ka, na], f"{b}_acc": [kb, nb],
            f"{a}_wilson95": [round(x, 4) for x in wilson(ka, na)],
            f"{b}_wilson95": [round(x, 4) for x in wilson(kb, nb)],
            "diff_pp": round(100 * (ka / na - kb / nb), 2) if na and nb else None,
            "diff_newcombe95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "p_raw": p,
            "direction": f"{a}>{b}" if na and nb and ka / na > kb / nb else
                         (f"{b}>{a}" if na and nb and ka / na < kb / nb else "equal")}
