"""Assemble the standalone simulation runtime before importing its modules."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def configure(runtime, row_root, *, pose_send=None, pose_route=None):
    runtime, row_root = Path(runtime).resolve(), Path(row_root).resolve()
    if any(name == "carryon" or name.startswith("carryon.") for name in sys.modules):
        raise RuntimeError("Recovery assembly must run in a fresh isolated process")
    manifest = json.loads((runtime / "RUNTIME.json").read_text())
    if manifest["schema"] != "carryon-recovery-runtime/v1":
        raise ValueError("Invalid runtime manifest")
    for name in manifest["required_files"]:
        path = (runtime / name).resolve()
        if not path.is_relative_to(runtime) or not path.is_file():
            raise ValueError("Missing or invalid runtime file: " + name)
    admitted = json.loads((runtime / "ADMITTED_MEMORY.json").read_text())
    suite = admitted["suite"]
    if suite not in ("libero_recovery", "libero_perturbation"):
        raise ValueError("Unknown recovery suite")
    resources = json.loads((runtime / "RESOURCES.json").read_text())
    rows = json.loads(Path(resources["row_manifest"]).read_text())["rows"]
    selected = [row for row in rows if row["row_id"] == row_root.name]
    if len(selected) != 1:
        raise ValueError("Exactly one bound row is required")
    row = selected[0]
    judge_on = suite == "libero_perturbation"
    if row["judge_control_enabled"] is not judge_on:
        raise ValueError("Judge setting differs from the suite protocol")
    if not judge_on and (row["judge_model_calls"] != 0 or row["protocol_recovery_request"] is not True):
        raise ValueError("Recorded Recovery requires Judge-off protocol recovery")
    active = json.loads((runtime / "ACTIVE.json").read_text())
    env = {
        "CARRYON_RECOVERY_RESOURCES": runtime / "RESOURCES.json",
        "CARRYON_RECOVERY_PIPELINE": "semantic", "CARRYON_GRASP_EXECUTION_MODE": "topdown",
        "CARRYON_RECOVERY_ROW_ROOT": row_root,
        "CARRYON_VISUAL_OBJECT_MEMORY_MANIFEST": runtime / "bindings/semantic_memory_config.json",
        "CARRYON_RECOVERY_PARAMETER_MEMORY_PATH": runtime / "memory/PARAMETER.jsonl",
        "CARRYON_RECOVERY_SEMANTIC_MEMORY_PATH": runtime / "memory/SEMANTIC.jsonl",
        "CARRYON_MAX_ATTEMPTS": "1", "CARRYON_MEMORY_SNAPSHOT_POINTER": runtime / "ACTIVE.json",
        "CARRYON_MEMORY_CONTRACT_PATH": runtime / "contracts/COMMON_CONTRACT.json",
        "CARRYON_MEMORY_CONTRACT_SHA256": active["contract_sha256"],
    }
    os.environ.update({k: str(v) for k, v in env.items()})
    search = [runtime / "runtime", runtime / "bindings", runtime / "worker",
              runtime / "bindings/TEMPORAL", runtime]
    sys.path[:0] = [str(p) for p in search]
    os.environ["PYTHONPATH"] = os.pathsep.join(str(p) for p in search)
    if suite == "libero_recovery":
        if not callable(pose_send):
            raise ValueError("Recorded Recovery requires an explicit pose transport")
        from pose_worker_runtime import register
        contract = json.loads((runtime / "contracts/COMMON_CONTRACT.json").read_text())
        scope = contract["experiment_scope"]
        register(row_root=row_root, snapshot_path=scope["pose_memory"]["path"],
                 snapshot_sha=scope["pose_memory"]["sha256"], send_once=pose_send,
                 max_pose_calls=scope["runtime_HTTP_budget"]["pose_classifier"],
                 provider=(pose_route or {}).get("provider"),
                 response_provider_aliases=(pose_route or {}).get("response_provider_aliases"))
    inner = importlib.import_module(
        "candidate_presentation")
    from protocol_binding import configure_inner
    configure_inner(inner, "GRID", row_root)
    import memory_reader
    memory_reader.install(inner)
    return inner
