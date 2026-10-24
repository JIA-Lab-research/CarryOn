"""Immutable, relocatable recovery memory with explicit parent and image bindings."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from ..files import (bound_file, inside, json_bytes, new_directory, read_json,
                     sha256, write_json)
from .parsers import parse_p, parse_s

SUITES = ("libero_perturbation", "libero_recovery")


def _jsonl(values):
    return b"".join(json.dumps(v, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False).encode() + b"\n"
                    for v in values)


def _key(value):
    return (value["object_key"], value.get("pose"))


def _validate_rows(rows, *, semantic=False, source_parents=None):
    seen = set()
    for row in rows:
        key = row["object_key"]
        if not isinstance(key, str) or not key or key != key.strip().lower():
            raise ValueError("Memory object keys must be canonical lowercase strings")
        if _key(row) in seen:
            raise ValueError("Duplicate memory object/pose")
        seen.add(_key(row))
        if semantic:
            parse_s(json.dumps({"decision": "WRITE", "template": row["template"],
                                "rationale": row["rationale"]}))
        else:
            # Stored values are absolute metres, validated using the original bounds.
            value = row["corrected_z_m"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Stored z must be numeric metres")
            anchor=value
            if row.get("schema")=="carryon-two-round-vlm-z-object-memory/v2":
                source=row.get("source_v2",{})
                if source.get("schema")!="bound-mechanical-source/v2":
                    raise ValueError("Invalid parameter source schema")
                if row.get("prior_z_m")!=0. or row.get("value_source")!="vlm_runtime":
                    raise ValueError("Parameter memory requires absolute runtime values")
                allowed={"DESCENT_JAM_ACTUALLY_EXECUTED","TERMINAL_MEASURED_EMPTY",
                         "MEASURED_LOSS_AFTER_NONEMPTY","MEASURED_WIDTH_UNSTABLE"}
                if not isinstance(source.get("classes"),list) or not set(source["classes"])<=allowed:
                    raise ValueError("Invalid mechanical evidence classes")
                generation=source.get("reflection_round")
                if generation not in (1,2) or isinstance(generation,bool):
                    raise ValueError("Invalid parameter reflection generation")
                if source_parents is not None and source.get("parent_snapshot_sha256")!=source_parents.get(str(generation)):
                    raise ValueError("Parameter source parent does not match memory lineage")
                digest=source.get("batch_sha256")
                if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid parameter batch binding")
                anchor=source["anchor_z_m"]
            from decimal import Decimal
            parse_p(json.dumps({"action": "UPDATE_Z", "corrected_z_mm": float(Decimal(str(value)) * 1000),
                                "rationale": row["rationale"]}), anchor)
            if row.get("pose") not in (None, "UPRIGHT", "LYING"):
                raise ValueError("Unknown poses cannot enter parameter memory")


def load(root):
    root = Path(root).resolve()
    manifest = read_json(root / "MANIFEST.json")
    if manifest.get("schema") != "carryon-recovery-memory/v1":
        raise ValueError("Not a prepared CarryOn memory directory")
    if manifest["suite"] not in SUITES or manifest["generation"] not in (0, 1, 2):
        raise ValueError("Invalid suite or generation")
    for relative, digest in manifest["files"].items():
        bound_file({"path": relative, "sha256": digest}, root)
    state = {"root": root, "manifest": manifest,
             "snapshot_sha256": sha256(root / "MANIFEST.json")}
    for role, name in (("P", "PARAMETER.jsonl"), ("S", "SEMANTIC.jsonl")):
        raw = (root / name).read_bytes()
        state[role + "_bytes"] = raw
        state[role] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    state["pose_bytes"] = (root / "POSE_MEMORY.json").read_bytes()
    state["pose"] = json.loads(state["pose_bytes"])
    _validate_rows(state["P"],source_parents=manifest["provenance"].get("source_parent_by_round",{}))
    _validate_rows(state["S"], semantic=True)
    _validate_rows(state["pose"]["records"])
    if manifest["suite"] == "libero_recovery" and state["P"]:
        raise ValueError("Recorded Recovery requires pose-indexed parameters")
    if manifest["suite"] == "libero_perturbation" and state["pose"]["records"]:
        raise ValueError("Perturbation does not consume pose-indexed parameters")
    if manifest["generation"] == 0 and (state["P_bytes"] or state["S_bytes"] or state["pose"]["records"]):
        raise ValueError("R0 requires empty recovery memory")
    for row in state["S"]:
        bound_file({"path": row["reference_path"], "sha256": row["reference_sha256"]}, root)
    return state


def _finish(root, *, suite, generation, parent_sha256, provenance, fixture=False):
    files = {p.relative_to(root).as_posix(): None
             for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "MANIFEST.json", {
        "schema": "carryon-recovery-memory/v1", "suite": suite,
        "generation": generation, "parent_snapshot_sha256": parent_sha256,
        "files": files, "fixture_only": fixture, "provenance": provenance,
    })


def empty(output, suite):
    if suite not in SUITES:
        raise ValueError("Unknown recovery suite")
    with new_directory(output) as root:
        (root / "PARAMETER.jsonl").write_bytes(b"")
        (root / "SEMANTIC.jsonl").write_bytes(b"")
        write_json(root / "POSE_MEMORY.json", {"schema": "isolated-pose-snapshot/v1",
                   "coordinate": "world_contact_tcp_z_m", "records": []})
        write_json(root / "DECISIONS.json", [])
        _finish(root, suite=suite, generation=0, parent_sha256=None,
                provenance={"mode": "empty", "source_parent_by_round": {}})
    return load(output)


def publish(parent, batches, responses, output, *, fixture=False):
    """Apply each bound reflection response once.

    Invalid responses retain the parent value. Model calls occur separately, and
    service errors remain recorded."""
    previous = load(parent)
    generation = previous["manifest"]["generation"] + 1
    if generation not in (1, 2):
        raise ValueError("No reflection after R2")
    if previous["manifest"]["fixture_only"] and not fixture:
        raise ValueError("Fixture memory cannot become a live experiment parent")
    p = {_key(v): copy.deepcopy(v) for v in previous["P"]}
    s = {v["object_key"]: copy.deepcopy(v) for v in previous["S"]}
    poses = {_key(v): copy.deepcopy(v) for v in previous["pose"]["records"]}
    events, seen, new_references = [], set(), {}
    for response in responses:
        item = response["item"]
        if item in seen:
            raise ValueError("Duplicate response; implicit retry is forbidden")
        seen.add(item)
        batch = batches[item]
        if batch["parent_snapshot_sha256"] != previous["snapshot_sha256"]:
            raise ValueError("Response batch was built with another parent")
        if batch["reflection_round"] != generation:
            raise ValueError("Response generation mismatch")
        if response["request_sha256"] != batch["request"]["sha256"]:
            raise ValueError("Response/request binding mismatch")
        request = read_json(bound_file(batch["request"]))
        if response.get("fixture", False) and not fixture:
            raise ValueError("Fixture response refused for live memory")
        event = {"item": item, "request_sha256": response["request_sha256"],
                 "parent_snapshot_sha256": previous["snapshot_sha256"]}
        if response.get("status") in ("service_error", "resource_truncated"):
            event.update(effect=response["status"].upper() + "_RETAINED", error=response.get("error"))
            events.append(event)
            continue
        raw_path = bound_file(response["response"])
        event["response_sha256"] = sha256(raw_path)
        value = read_json(raw_path)
        if isinstance(value, dict) and "choices" in value:
            if value.get("model") and value["model"] != request["model"]:
                # Any service-specific alias must be checked by the configured transport.
                if value["model"] not in batch.get("response_model_aliases", []):
                    raise ValueError("Response model differs from configured model")
            raw = value["choices"][0]["message"]["content"]
            event["usage"] = value.get("usage")
        else:
            raw = json.dumps(value, ensure_ascii=False)
        key, role = batch["object_key"], batch["role"]
        event.update(object_key=key, role=role)
        try:
            if role == "S":
                parsed = parse_s(raw)
                if key in s:
                    event["effect"] = "RETAIN_EXISTING_ENABLED"
                elif parsed["decision"] == "SKIP":
                    event["effect"] = "SKIP_RETAIN_DISABLED"
                else:
                    ref = batch["reference"]
                    image = bound_file(ref)
                    relative = "references/" + ref["sha256"] + image.suffix
                    new_references[relative] = image
                    s[key] = {"schema": "carryon-fixv2-semantic-memory/v1",
                              "created_utc": datetime.now(timezone.utc).isoformat(),
                              "object_key": key, "template": parsed["template"],
                              "rationale": parsed["rationale"], "reference_path": relative,
                              "reference_sha256": ref["sha256"],
                              "registry_sha256": batch["registry_sha256"],
                              "trigger": "postrow_recovery_failure", "value_source": "vlm_runtime"}
                    event["effect"] = "WRITE_ISOLATED"
            elif role == "P":
                pose = batch.get("pose")
                recorded = previous["manifest"]["suite"] == "libero_recovery"
                if recorded and pose not in ("UPRIGHT", "LYING"):
                    raise ValueError("Recorded Recovery P requires a bound known pose")
                if not recorded and pose is not None:
                    raise ValueError("Perturbation P is indexed by object only")
                parsed = parse_p(raw, batch["anchor_z_m"])
                event["effect"] = "NO_UPDATE_RETAINED"
                if parsed["applied"]:
                    if recorded:
                        # The pose admission module verifies image/request/answer bindings.
                        from .pose import admit_parameter
                        record = admit_parameter(batch, raw, previous["pose"], response)
                    else:
                        record = {"schema": "carryon-two-round-vlm-z-object-memory/v2",
                                  "object_key": key, "created_utc": datetime.now(timezone.utc).isoformat(),
                                  "corrected_z_m": parsed["corrected_z_m"], "prior_z_m": 0.,
                                  "rationale": parsed["rationale"], "value_source": "vlm_runtime",
                                  "source_v2": {"schema": "bound-mechanical-source/v2",
                                      "classes": batch["mechanical_source_classes"],
                                      "reflection_round": generation,
                                      "batch_sha256": hashlib.sha256(json_bytes(batch)).hexdigest(),
                                      "parent_snapshot_sha256": previous["snapshot_sha256"],
                                      "anchor_z_m": batch["anchor_z_m"], "input_unit": "mm",
                                      "stored_unit": "m", "conversion_count": 1}}
                    (poses if recorded else p)[(key, pose)] = record
                    event["effect"] = "UPDATE_Z_ISOLATED"
            else:
                raise ValueError("Unknown reflector role")
            event["parsed"] = parsed
        except (ValueError, KeyError, TypeError) as exc:
            event.update(effect="PARSE_REJECTED_RETAINED", error=str(exc))
        events.append(event)
    if set(batches) != seen:
        raise ValueError("Every prepared request needs a response or explicit service-error record")
    with new_directory(output, inputs=[parent]) as root:
        if (previous["root"] / "references").is_dir():
            shutil.copytree(previous["root"] / "references", root / "references")
        for relative, source in new_references.items():
            target = inside(root, relative)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
        new_p = [p[k] for k in sorted(p)]
        new_s = [s[k] for k in sorted(s)]
        p_unchanged = p == {_key(v): v for v in previous["P"]}
        s_unchanged = s == {v["object_key"]: v for v in previous["S"]}
        (root / "PARAMETER.jsonl").write_bytes(previous["P_bytes"] if p_unchanged else _jsonl(new_p))
        (root / "SEMANTIC.jsonl").write_bytes(previous["S_bytes"] if s_unchanged else _jsonl(new_s))
        pose_document = dict(previous["pose"], records=[poses[k] for k in sorted(poses)])
        if poses == {_key(v): v for v in previous["pose"]["records"]}:
            (root / "POSE_MEMORY.json").write_bytes(previous["pose_bytes"])
        else:
            write_json(root / "POSE_MEMORY.json", pose_document)
        write_json(root / "DECISIONS.json", events)
        _finish(root, suite=previous["manifest"]["suite"], generation=generation,
                parent_sha256=previous["snapshot_sha256"], fixture=fixture,
                provenance={"mode": "fresh", "prepared_requests": len(batches),
                            "source_parent_by_round": {**previous["manifest"]["provenance"].get("source_parent_by_round",{}),
                                                       str(generation):previous["snapshot_sha256"]},
                            "responses": len(responses), "empty_batch_inherited": not batches,
                            "complete": not any(e["effect"] in ("SERVICE_ERROR_RETAINED", "RESOURCE_TRUNCATED_RETAINED")
                                                for e in events)})
    return load(output)
