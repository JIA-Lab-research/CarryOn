"""Run the RoboLab protocol in an isolated working directory with mock endpoints."""
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def provider_failures(directory):
    """Identify infrastructure failures without reclassifying model decisions."""
    failures = []
    for transcript in sorted((Path(directory) / "transcripts").glob("*.txt")):
        text = transcript.read_text(encoding="utf-8", errors="replace")
        statuses = re.findall(r"planner VLM returned ([45][0-9]{2})\b", text)
        checker_failed = "check_endpoint: VLM call failed:" in text
        transport_failed = "planner VLM http error:" in text
        malformed_envelope = bool(re.search(r"planner VLM returned 2[0-9]{2}: (?:invalid JSON:|unexpected response shape:)", text))
        if statuses or checker_failed or transport_failed or malformed_envelope:
            failures.append(dict(transcript=transcript.name, http_statuses=sorted(set(map(int, statuses))), check_endpoint_call_failed=checker_failed, planner_transport_failed=transport_failed, malformed_provider_envelope=malformed_envelope))
    return failures

def local_api_url(value):
    """Accept a credential-free loopback URL for a separately served model."""
    url = urlsplit(value)
    try:
        port = url.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid local API port") from exc
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username is not None or url.password is not None
            or url.query or url.fragment or not port or url.path.rstrip("/") != "/v1"):
        raise argparse.ArgumentTypeError("local API must be http://127.0.0.1:<port>/v1 (or localhost/[::1]), without credentials")
    return value.rstrip("/")


def configure_local_api(workspace, base_url):
    """Change only the API URL in the copied workspace."""
    relative = "evaluation/planner.yaml"
    config = workspace / relative
    old = 'base_url: "https://openrouter.ai/api/v1"'
    text = config.read_text(encoding="utf-8")
    if text.count(old) != 1:raise ValueError("Provider URL not found exactly once")
    config.write_text(text.replace(old, "base_url: " + json.dumps(base_url)), encoding="utf-8")
    return dict(provider_override=dict(base_url=base_url, changed_file=relative,
                changed_field="planner_vlm.base_url", model_and_temperature_unchanged=True))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("paper", "smoke"), default="paper")
    p.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    p.add_argument("--suite", choices=("main", "text-stale", "image-stale"), default="main")
    p.add_argument("--cells", help="Comma-separated main-table cells for a small run")
    p.add_argument("--repeats", type=int, help="Main default 11; stale stage 1 is fixed at 2")
    p.add_argument("--manifest-only", action="store_true", help="Construct protocol manifests without model calls")
    p.add_argument("--local-api-base-url", type=local_api_url, help="Optional loopback OpenAI-compatible service; preserves the frozen model alias and temperature")
    p.add_argument("--lane", type=int, default=1000, help="Uses frame port 18121+lane and callback port 9000+lane")
    args = p.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        p.error("output already exists; choose a new directory")
    if args.lane < 1 or 18121 + args.lane > 65535:
        p.error("invalid lane")
    if args.preset == "smoke" and (args.cells or args.repeats not in (None, 1)):
        p.error("smoke uses the fixed eight cells and one repeat")
    if args.suite != "main" and args.preset == "paper" and (args.cells or args.repeats not in (None, 2)):
        p.error("stale protocol requires the complete fixed stage allocation")
    if args.suite != "main" and not args.manifest_only and not Path("/proc").is_dir():
        p.error("stale stage-2 process exclusivity check requires Linux /proc")
    repeats = 1 if args.preset == "smoke" else args.repeats if args.repeats is not None else (11 if args.suite == "main" else 2)
    if repeats < 1:
        p.error("repeats must be positive")
    if not args.manifest_only and not args.local_api_base_url and not os.environ.get("OPENROUTER_API_KEY"):
        p.error("OPENROUTER_API_KEY is required for model evaluation")
    for port in (18121 + args.lane, 9000 + args.lane):
        with socket.socket() as sock:
            if os.name != "nt":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                p.error(f"port {port} is occupied; choose another lane")
    source = HERE / "benchmark"
    sys.path.insert(0, str(source / "evaluation"))
    output.mkdir(parents=True)
    workspace = output / "workspace"
    shutil.copytree(source, workspace, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    override = configure_local_api(workspace, args.local_api_base_url) if args.local_api_base_url else {}
    evaluation = workspace / "evaluation"
    spec = json.loads((evaluation / "targets.json").read_text())
    requested = set(args.cells.split(",")) if args.cells else set(spec["specs"])
    if not requested or requested - set(spec["specs"]):
        p.error("unknown or empty cell selection")
    spec["specs"] = {k: v for k, v in spec["specs"].items() if k in requested}
    selected = output / "selected_spec.json"
    selected.write_text(json.dumps(spec, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if args.local_api_base_url:
        # Never forward an existing provider credential to a local service.
        env["OPENROUTER_API_KEY"] = "local-loopback-no-secret"
    runs = [("run_text.py", "text", ["--arms", "T"]), ("run_text_memory.py", "text_memory", []), ("run_image.py", "image", [])] if args.suite == "main" else [("text_revision_stage1.py", "text_revision_stage1", []), ("text_revision_stage2.py", "text_revision_stage2", [])] if args.suite == "text-stale" else [("image_revision_stage1.py", "image_revision_stage1", []), ("image_revision_stage2.py", "image_revision_stage2", [])]
    selected_stale = []
    if args.suite != "main" and args.preset == "smoke":
        table = json.loads((evaluation / "protocol/revision_stage1.json").read_text())
        seen = set()
        for row in table["configs"]:
            if row["expected_endpoint"] not in seen:
                selected_stale.append(row["config_id"])
                seen.add(row["expected_endpoint"])
        if len(selected_stale) != 4:
            raise ValueError("The frozen seed table must cover four target stations")
        runs = [(script, name, [*extra, "--configs", ",".join(selected_stale)])
                for script, name, extra in runs if not script.endswith("_stage2.py")]
    info = dict(suite=args.suite, cells=sorted(requested), repeats=repeats, manifest_only=args.manifest_only, pilot=args.suite == "main" and (len(requested) != 8 or repeats != 11), model="qwen/qwen3-vl-32b-instruct", temperature=0.2, robot_transport="mock", runs=[])
    info.update(override)
    info.update(preset=args.preset, pilot=info["pilot"] or args.preset == "smoke",
                stale_configs=selected_stale,
                stale_selection_rule="first seed-table config per target station" if selected_stale else "full frozen allocation")
    (output / "EVALUATION.json").write_text(json.dumps(info, indent=2) + "\n")
    for script, name, extra in runs:
        destination = workspace / "logs/robolab" / name
        cmd = [sys.executable, str(evaluation / script), "--spec", str(selected), "--out", str(destination), "--lane", str(args.lane), *extra]
        if not script.endswith("_stage2.py"):
            cmd += ["--repeats", str(repeats)]
        if args.manifest_only:
            cmd.append("--manifest-only")
        with (output / (name + ".console.txt")).open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
        external_errors = provider_failures(destination) if not args.manifest_only else []
        run_info = dict(name=name, returncode=result.returncode, provider_errors=external_errors)
        failed = bool(result.returncode) or bool(external_errors)
        if not failed:
            manifest = json.loads((destination / "RUN_MANIFEST.json").read_text())
            run_info["expected_rows"] = len(manifest["rows"])
            if not args.manifest_only:
                ledger = destination / "episodes.jsonl"
                rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
                run_info["recorded_rows"] = len(rows)
                run_info["instrument_errors"] = [r.get("tag") for r in rows if r.get("instrument_error")]
                run_info["complete"] = len(rows) == run_info["expected_rows"] and not run_info["instrument_errors"]
                failed = not run_info["complete"]
        info["runs"].append(run_info)
        (output / "EVALUATION.json").write_text(json.dumps(info, indent=2) + "\n")
        print(name, "returncode", result.returncode, "verified", not failed, flush=True)
        if failed:
            if external_errors:
                print("External model service failed; this run is invalid for performance evaluation.", flush=True)
            raise SystemExit(result.returncode or 1)
    print("Completed; model runs retain their full recorded outcomes in workspace/logs/robolab")

if __name__ == "__main__":
    main()
