"""Check simulator readiness and score episodes using runner-supplied paths."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,time,urllib.request
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path: Path, value: dict) -> None:
    if path.exists():
        raise RuntimeError(f"refuse overwrite: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=15.0) as response:
        return json.load(response)

def parse_shim_bridge_announcement(shim_log: Path) -> dict:
    matches = []
    pattern = re.compile(r"shim up:.*?zmq=([^:\s]+):(\d+)/(\d+)/(\d+)")
    for line in shim_log.read_text(encoding="utf-8", errors="replace").splitlines():
        found = pattern.search(line)
        if found:
            matches.append({
                "host": found.group(1), "rep_port": int(found.group(2)),
                "pull_port": int(found.group(3)), "pub_port": int(found.group(4)),
                "source_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
            })
    if len(matches) != 1:
        raise RuntimeError(f"expected one shim bridge announcement, found {len(matches)}")
    return matches[0]

def validate_shim_bridge_announcement(expected: dict, published: dict) -> dict:
    keys = ("host", "rep_port", "pull_port", "pub_port")
    mismatch = {key: {"expected": expected.get(key), "published": published.get(key)}
                for key in keys if expected.get(key) != published.get(key)}
    if mismatch:
        raise RuntimeError(f"shim/CHILD bridge announcement mismatch: {mismatch}")
    return {
        "verdict": "PASS", "published": published,
        "expected": {key: expected[key] for key in keys},
        "ports_source": expected["ports_source"],
        "host_source": expected["host_source"],
        "shim_equals_child_config": True,
    }

def probe_sim_readiness(
    base_url: str,
    *,
    phase: str,
    max_attempts: int = 8,
    retry_delay_s: float = 0.25,
) -> dict:
    attempts = []
    started = time.time()
    for attempt_index in range(1, max_attempts + 1):
        try:
            reply = http_json("GET", base_url.rstrip("/") + "/healthz")
            ok = isinstance(reply, dict) and reply.get("ok") is True
            attempts.append({
                "attempt": attempt_index,
                "status": "ready" if ok else "not_ready_reply",
            })
            if ok:
                return {
                    "phase": phase,
                    "verdict": "PASS",
                    "attempt_count": attempt_index,
                    "max_attempts": max_attempts,
                    "retry_delay_s": retry_delay_s,
                    "elapsed_s": round(time.time() - started, 6),
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({
                "attempt": attempt_index,
                "status": "not_ready_exception",
                "exception_type": type(exc).__name__,
            })
        if attempt_index < max_attempts:
            time.sleep(retry_delay_s)
    return {
        "phase": phase,
        "verdict": "FAIL",
        "attempt_count": max_attempts,
        "max_attempts": max_attempts,
        "retry_delay_s": retry_delay_s,
        "elapsed_s": round(time.time() - started, 6),
        "attempts": attempts,
    }

def probe_camera_readiness(
    cameras: list[dict],
    *,
    phase: str,
    max_attempts: int = 8,
    retry_delay_s: float = 0.25,
) -> dict:
    attempts = []
    started = time.time()
    for attempt_index in range(1, max_attempts + 1):
        targets = []
        all_ready = True
        for camera in cameras:
            record = {"name": camera["name"], "snapshot_url": camera["snapshot_url"]}
            try:
                request = urllib.request.Request(camera["snapshot_url"], method="GET")
                with urllib.request.urlopen(request, timeout=15.0) as response:
                    payload = response.read()
                    status = int(response.status)
                    content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
                jpeg_magic = payload.startswith(b"\xff\xd8\xff")
                ready = status == 200 and jpeg_magic
                record.update({
                    "status": "ready" if ready else "not_ready_reply",
                    "http_status": status,
                    "content_type": content_type,
                    "jpeg_magic": jpeg_magic,
                    "response_bytes": len(payload),
                    "response_sha256": hashlib.sha256(payload).hexdigest(),
                })
                all_ready = all_ready and ready
            except Exception as exc:
                all_ready = False
                record.update({"status": "not_ready_exception", "exception_type": type(exc).__name__})
            targets.append(record)
        attempts.append({"attempt": attempt_index, "targets": targets})
        if all_ready:
            return {
                "phase": phase,
                "verdict": "PASS",
                "attempt_count": attempt_index,
                "max_attempts": max_attempts,
                "retry_delay_s": retry_delay_s,
                "elapsed_s": round(time.time() - started, 6),
                "attempts": attempts,
            }
        if attempt_index < max_attempts:
            time.sleep(retry_delay_s)
    return {
        "phase": phase,
        "verdict": "FAIL",
        "attempt_count": max_attempts,
        "max_attempts": max_attempts,
        "retry_delay_s": retry_delay_s,
        "elapsed_s": round(time.time() - started, 6),
        "attempts": attempts,
    }

def wait_log(path: Path, process: subprocess.Popen, anchors: list[str], timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    text = ""
    while time.monotonic() < deadline:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if all(anchor in text for anchor in anchors):
                return text
        if process.poll() is not None:
            raise RuntimeError(f"process exited before anchors rc={process.returncode}: {anchors}")
        time.sleep(0.5)
    raise RuntimeError(f"timeout waiting for anchors {anchors}: {path}")

def assert_files_stable(
    paths: list[Path], stable_interval_s: float = 1.0
) -> None:
    def sample() -> dict[str, tuple[int, int]]:
        state = {}
        for path in paths:
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing/empty artifact before seal: {path}")
            stat = path.stat()
            state[str(path.resolve())] = (stat.st_size, stat.st_mtime_ns)
        return state

    first = sample()
    time.sleep(stable_interval_s)
    second = sample()
    if first != second:
        raise RuntimeError("artifact changed during pre-seal stability window")

def disturbance_application(raw: dict, row: dict) -> dict:
    if row.get("disturbance") == "none":
        if raw.get("disturbance") != "none" or raw.get("events", []):
            raise ValueError("natural-failure episode has unexpected disturbance evidence")
        return {"configured": "none", "applied": False, "event_count": 0, "natural_failure_no_injection_verified": True}
    events = [
        event for event in raw.get("events", [])
        if isinstance(event, dict) and event.get("kind") == "hold_lost"
    ]
    requested = [float(row["dx_m"]), float(row.get("dy_m", 0.0))]
    event = events[0] if len(events) == 1 else None
    measured = event.get("applied_delta_xy") if event is not None else None
    spatial_ok = (
        isinstance(measured, list)
        and len(measured) == 2
        and all(type(value) in (int, float) for value in measured)
        and all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(measured, requested))
    )
    return {
        "event_count": len(events),
        "applied": bool(spatial_ok),
        "requested_delta_xy_m": requested,
        "measured_delta_xy_m": measured,
        "drop_from_xyz": None if event is None else event.get("drop_from"),
        "drop_observed_to_xyz": (
            None if event is None else event.get("drop_observed_to")
        ),
        "spatial_evidence_ok": bool(spatial_ok),
    }

def build_structured_result(
    raw: dict,
    row: dict,
    trace_files: list[Path],
    runtime_flags_path: Path,
) -> dict:
    steps = []
    for path in trace_files:
        trace = json.loads(path.read_text(encoding="utf-8"))
        recovery = trace.get("recovery", {})
        if isinstance(recovery, dict) and isinstance(recovery.get("steps"), list):
            steps.extend(recovery["steps"])
    handback_step_ok = any(
        isinstance(step, dict)
        and step.get("op") == "handback_ack"
        and step.get("ok") is True
        for step in steps
    )
    if runtime_flags_path.is_file():
        flags = json.loads(runtime_flags_path.read_text(encoding="utf-8"))
    else:
        flags = {
            "destination_transport_started": False,
            "vla_started": False,
            "judge_started": False,
            "transitions": [],
        }
    official = {
        key: raw[key] for key in ("success", "timeout", "terminal_reason")
    }
    result = dict(raw)
    result.update({
        "schema": "destination-suffix-formal-structured-result/v1",
        "official_env_tuple": official,
        "disturbance_application": disturbance_application(raw, row),
        "recovery_evidence": {
            "handback_step_ok": bool(handback_step_ok),
            "destination_transport_started": bool(
                flags.get("destination_transport_started")
            ),
            "vla_started": bool(flags.get("vla_started")),
            "judge_started": bool(flags.get("judge_started")),
            "runtime_flag_transitions": flags.get("transitions", []),
        },
    })
    return result
