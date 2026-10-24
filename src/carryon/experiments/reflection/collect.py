"""Deterministic failure selection and bound evidence collection."""
from __future__ import annotations
from collections import defaultdict
import copy
import hashlib
from pathlib import Path

from ..files import bound_file, inside, new_directory, read_json, sha256, write_json
from .extract import extract_object, extract_recorded
from .extract_m2 import extract_object_m2
from .memory import load

SEED = 20260915
SAMPLES_PER_OBJECT = 3


def select_failures(rows, suite, generation):
    groups = defaultdict(list)
    seen = set()
    seen_ordinals = set()
    ordinal_order = suite == "libero_perturbation" and generation == 2
    for row in rows:
        if ordinal_order:
            ordinal = row["ordinal"]
            if type(ordinal) is not int or ordinal < 0 or ordinal in seen_ordinals:
                raise ValueError("Second-round reflection requires unique nonnegative R1 ordinals")
            seen_ordinals.add(ordinal)
        if row["logical_key"] in seen:
            raise ValueError("Duplicate logical case in round")
        seen.add(row["logical_key"])
        if row["status"] == "service_error":
            continue
        if row["status"] != "completed" or type(row["success"]) is not bool:
            raise ValueError("Every case requires a terminal status and a Boolean success")
        if row["success"]:
            continue
        if ordinal_order and not row["recovery_triggered"]:
            continue
        groups[row["object_key"]].append(row)
    selected = []
    for key in sorted(groups):
        candidates = groups[key]
        if ordinal_order:
            order = lambda row: row["ordinal"]
        else:
            order = lambda row: hashlib.sha256(
                f"{SEED}|{key}|{row['logical_key']}".encode("utf-8")).hexdigest()
        selected.extend(sorted(candidates, key=order)[:SAMPLES_PER_OBJECT])
    if ordinal_order:
        selected.sort(key=lambda row: row["ordinal"])
    return selected


def collect(experiment, parent, output, *, suite, generation):
    experiment, parent = Path(experiment).resolve(), Path(parent).resolve()
    previous = load(parent)
    if generation not in (1, 2) or previous["manifest"]["generation"] != generation - 1:
        raise ValueError("Parent generation mismatch")
    if previous["manifest"]["suite"] != suite:
        raise ValueError("Parent suite mismatch")
    round_file = experiment / "ROUND.json"
    document = read_json(round_file)
    if document["status"] != "closed" or document["round"] != "R" + str(generation - 1):
        raise ValueError("Reflection requires the closed preceding round")
    if document["memory_snapshot_sha256"] != previous["snapshot_sha256"]:
        raise ValueError("Round did not consume the specified parent memory")
    rows = copy.deepcopy(document["cases"])
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate row id")
    if {row["logical_key"] for row in rows} != set(document["expected_cases"]):
        raise ValueError("Closed-round case inventory mismatch")
    roots = {}
    for row in rows:
        roots[row["row_id"]] = inside(experiment, row["evidence_dir"])
        result_path = bound_file(row["result"], experiment)
        result = read_json(result_path)
        if result["status"] != row["status"] or result.get("success") != row.get("success"):
            raise ValueError("Round/result outcome binding mismatch")
    selected = select_failures(rows, suite, generation)
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["object_key"]].append({"row_id": row["row_id"]})
    sources = {"objects": [{"object_name": key, "selected_sources": value}
                            for key, value in sorted(grouped.items())]}
    references = {row["reference_sha256"]: inside(parent, row["reference_path"])
                  for row in previous["S"]}
    object_references = {row["object_key"]: {"path": str(inside(parent, row["reference_path"])),
                         "sha256": row["reference_sha256"]} for row in previous["S"]}
    # R0 also carries the originally registered reference images, independently of S.
    if document.get("references"):
        registry = read_json(bound_file(document["references"], experiment))
        for record in registry["files"]:
            image = bound_file(record, experiment)
            references[record["sha256"]] = image
            key = record["object_key"]
            if key in object_references and object_references[key]["sha256"] != record["sha256"]:
                raise ValueError("Round reference differs from the parent memory reference")
            object_references[key] = {"path": str(image), "sha256": record["sha256"]}
    with new_directory(output, inputs=[experiment, parent]) as root:
        extractor = extract_object if suite == "libero_perturbation" else extract_recorded
        p_index = {row["object_key"]: row for row in previous["P"]}
        s_index = {row["object_key"]: row for row in previous["S"]}
        specialised = set(grouped) & p_index.keys() & s_index.keys() if suite == "libero_perturbation" and generation == 2 else set()
        general = {"objects": [ob for ob in sources["objects"] if ob["object_name"] not in specialised]}
        batches, audit = extractor(general, roots, references, root)
        if specialised:
            sample = [dict(row, global_ordinal=row["ordinal"]) for row in selected if row["object_key"] in specialised]
            extra, records = extract_object_m2(sample, roots, references, root, p_index, s_index)
            batches.update(extra)
            audit.extend(records)
        families = {entry["attempt_id"]: entry for entry in audit}
        for key, batch in batches.items():
            batch["reflection_round"] = generation
            batch["source_round"] = "R" + str(generation - 1)
            batch["parent_snapshot_sha256"] = previous["snapshot_sha256"]
            current = batch["memory_context"]
            current["parent_snapshot_sha256"] = previous["snapshot_sha256"]
            current["S_entry_present"] = key in s_index
            current["active_template"] = s_index.get(key, {}).get("template")
            if key in p_index:
                current["P_current_z_mm"] = p_index[key]["corrected_z_m"] * 1000
                current["anchor_z_mm"] = current["P_current_z_mm"]
                current["anchor_source"] = "existing P"
            for attempt in batch["attempts"]:
                attempt["source_round"] = batch["source_round"]
                attempt["generation"] = generation - 1
                if key not in specialised:
                    attempt["consumed_semantic_state"] = {
                        "S_entry_present": key in s_index,
                        "active_template": current["active_template"]}
                if suite == "libero_recovery" and generation == 2:
                    entry = families.get(attempt["attempt_id"])
                    if entry is None:
                        attempt["consumed_parameter_state"] = {"status": "UNKNOWN_MISSING_FAMILY"}
                    else:
                        family = read_json(bound_file({"path": entry["family_path"],
                                                      "sha256": entry["family_sha256"]}))["common_family"]
                        from .evidence import mm
                        attempt["consumed_parameter_state"] = {
                            "P_applied": family.get("P_applied"),
                            "P_current_z_mm": mm(family["P_record"]["corrected_z_m"])
                                              if family.get("P_applied") else None,
                            "pose": family.get("pose_binding", {}).get("pose")}
            for image in batch["roles"]:
                path = Path(image["local_path"])
                if path.is_relative_to(root):
                    image["local_path"] = str(Path(output).resolve() / path.relative_to(root))
        write_json(root / "BATCHES.json", batches)
        write_json(root / "IMAGE_BINDINGS.json", audit)
        write_json(root / "SOURCE_ROOTS.json", {key: str(value) for key, value in roots.items()})
        write_json(root / "REFERENCES.json", object_references)
        write_json(root / "SELECTION.json", {"suite": suite, "generation": generation,
            "seed": SEED, "per_object_maximum": SAMPLES_PER_OBJECT,
            "rule": "ordinal" if suite == "libero_perturbation" and generation == 2 else "sha256(seed|object|logical_key)",
            "round_sha256": sha256(round_file), "selected": selected,
            "service_errors": [r for r in rows if r["status"] == "service_error"],
            "evidence_based_replacement": False})
    return read_json(Path(output) / "BATCHES.json")
