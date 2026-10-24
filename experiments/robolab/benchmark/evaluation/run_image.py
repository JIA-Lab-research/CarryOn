#!/usr/bin/env python3
'Evaluate image requests with and without scene memory.'
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RoboLab = REPO / "evaluation"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RoboLab))

from episode import m_manifest_text, serve_cell, window  # noqa: E402
from lifecycle import append_event, reset_memory  # noqa: E402
import lane as lane_module  # noqa: E402

CANON = RoboLab / "reference_canonical"


ARMS = ("P", "PM")
# One sentence for every cell: the text names nothing.
QUERY_P = "go pick up this"
QUERY_PM = "go pick up mine"
TURN1_PM = "this is mine"                   # semantics; issued as the command below
ENTITY = "mine"
NEUTRAL = "reference.png"                   # the basename the prompt will echo
REMEMBERED_RE = re.compile(r"\[remember_image\] stored:")
DONE_RE = re.compile(r"\[plan-[0-9a-f]+ done\]")


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


def canonical_for(spec: dict) -> Path:
    return CANON / Path(spec["reference_canonical_path"]).name


def serve_neutral(spec: dict, ref_dir: Path) -> tuple[Path, str]:
    """Copy this cell's reference image to a fixed, meaningless filename.

    The CLI replaces an @image token with "[attached image: <basename>]", and
    that marker is text the planner reads. Serving bagel.png would hand the
    model the word the picture exists to replace.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    dst = ref_dir / NEUTRAL
    src = canonical_for(spec)
    shutil.copyfile(src, dst)
    return dst


def _spawn(cfg: Path):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
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
    return proc, q


def _drain(q, lines, deadline, quiet_s, stop_re=None):
    """Read until stop_re matches, the stream ends, or it goes quiet."""
    hit = False
    while time.time() < deadline:
        try:
            ln = q.get(timeout=quiet_s)
        except queue.Empty:
            break
        if ln is None:
            break
        lines.append(ln.rstrip("\n"))
        if stop_re is not None and stop_re.search(ln):
            hit = True
            break
        if stop_re is None and DONE_RE.search(ln):
            hit = True
    return hit


def run_episode_img(cfg: Path, arm: str, turns: list, tag: str, out: Path,
                    timeout_s: float, quiet_s: float) -> dict:
    """One episode. For PM the clock starts at TURN 2.

    PM's turn 1 only files an image; timing it alongside M's single turn would
    compare a two-turn wall clock against a one-turn one and call the
    difference a property of memory.
    """
    proc, q = _spawn(cfg)
    lines: list[str] = []
    t0 = time.time()
    deadline = t0 + timeout_s
    turn1_ok = None
    try:
        if arm == "PM":
            proc.stdin.write(turns[0] + "\n")
            proc.stdin.flush()
            # A definite signal, not a silence: the command echoes on success.
            turn1_ok = _drain(q, lines, min(deadline, time.time() + 60.0),
                              10.0, stop_re=REMEMBERED_RE)
        t_measure = time.time()
        proc.stdin.write(turns[-1] + "\n")
        proc.stdin.flush()
        done = _drain(q, lines, deadline, quiet_s)
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
        "\n".join(lines) + "\n", encoding="utf-8")
    return {"plan_done": done, "wall_s": round(t1 - t0, 3),
            "t_start": t_measure, "t_end": t1,
            "t_process_start": t0, "turn1_ok": turn1_ok,
            "agent_died_early": (t1 - t_measure) < 10.0 and not done}


def profile_has_reference(mem: Path) -> tuple[bool, str | None]:
    """Did turn 1 actually put a readable reference image in the profile?"""
    for md in sorted((mem / "user").glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        if f"## entity:{ENTITY}" not in text:
            continue
        for line in text.splitlines():
            if line.strip().startswith("- reference_image:"):
                p = line.split(":", 1)[1].strip()
                # the store writes a path relative to the agent's cwd; resolve
                # it the same way rather than against whoever runs this check
                full = p if os.path.isabs(p) else os.path.join(str(REPO), p)
                return os.path.isfile(full), p
    return False, None


def planner_saw_image(plog: Path, offset: int) -> tuple[bool, int]:
    """Return whether requests recorded since offset contain an image part."""
    if not plog.is_file():
        return False, offset
    size = plog.stat().st_size
    if size <= offset:
        return False, size
    with plog.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(offset)
        chunk = fh.read()
    return ('"image_url"' in chunk), size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(RoboLab / "targets.json"))
    ap.add_argument("--config", default=str(RoboLab / "planner.yaml"))
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--out", default=str(REPO / "logs" / "robolab" / "image"))
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--quiet-s", type=float, default=45.0)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--cells", default=None,
                    help="comma-separated configuration subset")
    ap.add_argument("--lane", type=int, default=None,
                    help="run in an isolated lane: own port, memory, logs, config")
    a = ap.parse_args()
    lane = lane_module.setup(a.lane, Path(a.config))
    a.config = str(lane.config)

    specs = json.loads(Path(a.spec).read_text())["specs"]
    cells = sorted(specs)
    if a.cells:
        want = [c.strip() for c in a.cells.split(",") if c.strip()]
        unknown = [c for c in want if c not in cells]
        if unknown:
            raise SystemExit(f"unknown cells: {unknown}")
        cells = [c for c in cells if c in want]

    # preflight: every reference image must resolve ON THIS HOST
    missing = [c for c in cells if not canonical_for(specs[c]).is_file()]
    if missing:
        raise SystemExit(f"REFERENCE_IMAGES_MISSING on this host: {missing}")

    rows, o = [], 0
    for cell in cells:
        for rep in range(1, a.repeats + 1):
            for arm in ARMS:
                o += 1
                rows.append({"batch": "img", "cell": cell, "arm": arm,
                             "repeat": rep, "ordinal": o,
                             "expected_endpoint": specs[cell]["host_endpoint"],
                             "kind": specs[cell]["kind"],
                             "tag": f"{cell}_{arm}_r{rep}"})

    out = Path(a.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    manifest = {
                   'lane': lane.describe(),
                   'schema': 'robolab-img-manifest/v1',
                   'namespace': 'carryon-robolab-image',
                   'design': '2x2 completion: rows text/image target, cols no/with location memory; T,M carried from the main table',
                   'arms': {'P': {'memory': 'none', 'query': QUERY_P, 'image': 'in-context'}, 'PM': {'memory': f'scene manifest (same as M) + entity:{ENTITY}/reference_image', 'turn1': TURN1_PM, 'query': QUERY_PM, 'image': 'from user profile'}},
                   'text_never_names_the_target': True,
                   'neutral_attachment_basename': NEUTRAL,
                   'n_cells': len(cells),
                   'repeats': a.repeats,
                   'n_episodes': len(rows),
                   'rows': rows
               }
    mpath = out / "RUN_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    print(f"namespace       : {manifest['namespace']}")
    print(f"cells/eps       : {len(cells)}/{len(rows)}")
    print(f"P  query        : {QUERY_P!r} + [image]")
    print(f"PM query        : {TURN1_PM!r} + [image]  ->  {QUERY_PM!r}")
    if a.manifest_only:
        return

    ledger = out / "reset_ledger.jsonl"
    results = out / "episodes.jsonl"
    mem = lane.mem
    tlog = lane.tlog
    plog = lane.plog
    ref_dir = out / "ref"

    served, errs, ran = None, 0, 0
    for r in rows:
        cell, arm = r["cell"], r["arm"]
        spec = specs[cell]
        if served != cell:
            serve_cell(cell, out / 'frame_server.log', port=lane.port)
            served = cell

        ev = reset_memory(mem, arm=arm, batch="img", repeat=r["repeat"],
                          cell=cell, reason=f"ordinal {r['ordinal']}")
        append_event(ledger, ev)
        ref_path = serve_neutral(spec, ref_dir)

        if arm == "PM":
            # PM knows WHERE things are, exactly as M does -- the row differs
            # from M only in how the target itself is specified.
            (mem / "scene.md").write_text(m_manifest_text(spec), encoding="utf-8")
            turns = [f'/remember_image "{ref_path}" {ENTITY}', QUERY_PM]
        else:
            turns = [f'{QUERY_P} @image:"{ref_path}"']

        if tlog.exists():
            tlog.unlink()
        p_off = plog.stat().st_size if plog.is_file() else 0
        ep = run_episode_img(Path(a.config), arm, turns, r["tag"], out,
                             a.timeout_s, a.quiet_s)
        ep.update(window(tlog, ep["t_start"], ep["t_end"]))
        saw_image, _ = planner_saw_image(plog, p_off)
        ref_ok, ref_stored = (profile_has_reference(mem) if arm == "PM"
                              else (None, None))

        transcript = (out / "transcripts" / f"{r['tag']}.txt").read_text(
            encoding="utf-8", errors="replace")
        row = dict(r)
        row.update(ep)
        checked = row.get("checked_endpoints") or []
        row.update({
                       'instruction': turns[-1],
                       'turns': turns,
                       'planner_saw_image': saw_image,
                       'profile_reference_ok': ref_ok,
                       'profile_reference_path': ref_stored,
                       'seeded_scene_memory': arm == 'PM',
                       'looked': bool(checked),
                       'reached_without_looking': not checked and row.get('dispatched_arm_endpoint') is not None
                   })
        contract = ("cell", "arm", "repeat", "instruction",
                    "dispatched_arm_endpoint", "time_to_dispatch_s",
                    "search_invocations", "checked_endpoints", "writes_agent",
                     "plan_done",
                    "planner_saw_image")
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
        # An arm defined by an image is void without one. Same for PM's turn 1.
        row["image_gate_failed"] = (not saw_image) or (arm == "PM" and not ref_ok)
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
                if row["vlm_error"] else "no image reached the planner"
                                 if not saw_image else
                                 "turn 1 did not file a readable reference"
                                 if arm == "PM" and not ref_ok else
                                 "truncated turn" if truncated
                                 else "agent died early"))
            print(f"[STOP] instrument error at {r['tag']}: {why}", flush=True)
            break
        if ran % 16 == 0:
            print(f"[progress] ran={ran}/{len(rows)} instrument_errors={errs}",
                  flush=True)

    if mem.exists():
        shutil.rmtree(mem)
    mem.mkdir(parents=True, exist_ok=True)
    print(f"[done] ran={ran} instrument_errors={errs}")


if __name__ == "__main__":
    main()
