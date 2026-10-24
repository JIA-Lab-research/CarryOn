"""Reflection prompt profiles and request-response bindings."""
from __future__ import annotations
import base64
import copy
import hashlib
import json
from pathlib import Path

from ..files import bound_file, json_bytes, new_directory, read_json, sha256, write_json
from .memory import load

PROMPTS = Path(__file__).with_name("prompts")
FACT_SLOT = "{collector-built source fields, actual frozen S state, attempts and role bindings}"
V2_FEEDBACK = "\nThe whole-task terminal feedback is official_task_failure=true. It does not identify the failed stage, correct object, or correct parameter. Use recorded observations; NO_UPDATE/SKIP requires its reason."
V3_FEEDBACK = "\nOnly official_task_failure=true is terminal task feedback; it is not identity, grasp failure, or z cause certification."
M2_FEEDBACK = "\nOnly official_task_failure=true is terminal task feedback; it is not a certified stage, identity, grasp, or z diagnosis."


def prompt(role, profile):
    s_system = (PROMPTS / "semantic_system.txt").read_text(encoding="utf-8")
    if profile not in ("object_m1_v2", "object_m1_v3", "object_m2", "recovery_m1", "recovery_m2"):
        raise ValueError("Unknown frozen reflection prompt profile")
    family = "task_failure" if profile in ("object_m1_v2", "object_m2") else "evidence"
    name = "parameter" if role == "P" else "semantic"
    text = (PROMPTS / (name + "_" + family + ".txt")).read_text(encoding="utf-8")
    feedback = V2_FEEDBACK if profile == "object_m1_v2" else V3_FEEDBACK
    if profile == "object_m2":
        feedback = M2_FEEDBACK
        if role == "S":
            text += "\nEvery listed episode is an official R1 task failure with a real recovery trigger. Whole-task failure does not diagnose identity. Current M1 S state is already enabled; WRITE or SKIP leaves the same stored entry unchanged. Missing selected evidence is unknown, not a mismatch.\n"
        else:
            text += "\nEvery listed episode is an official R1 task failure with a real recovery trigger. Whole-task failure does not diagnose z, grasp, or identity. This batch-policy call is required even when no bound parameter attempt exists; decide UPDATE_Z or NO_UPDATE from the supplied evidence. Current M1 corrected z is the active numeric state.\n"
    return (s_system, text, feedback) if role == "S" else (text, "OBJECT_BATCH=" + FACT_SLOT, feedback)


def request_body(role, profile, facts, roles, model_config):
    """No sender and no inference; construct the exact provider request locally."""
    if role not in ("P", "S"):
        raise ValueError("Unknown reflector role")
    model = model_config["model"]
    if not isinstance(model, str) or not model:
        raise ValueError("An explicit model identifier is required")
    envelope = copy.deepcopy(read_json(PROMPTS / "model_requests.json")[role])
    envelope["model"] = model
    provider = model_config.get("provider")
    if provider is not None:
        if not provider.get("only") or provider.get("allow_fallbacks") is not False:
            raise ValueError("Pin a provider and disable fallback")
        envelope["provider"] = copy.deepcopy(provider)
    else:
        # Local compatible servers do not accept provider routing fields.
        if model_config.get("transport") != "local":
            raise ValueError("Remote inference requires explicit provider routing")
        envelope.pop("provider")
    system, template, feedback = prompt(role, profile)
    text = json.dumps(facts, ensure_ascii=False, allow_nan=False)
    if any(term in text.lower() for term in ("offline_gt", "correct_candidate", "pi_label", "gt_position", "oracle_pose")):
        raise ValueError("Privileged evaluator fields cannot enter model facts")
    content = [{"type": "text", "text": template.replace(FACT_SLOT, text) + feedback}]
    unique = {}
    for image in roles:
        bound_file({"path": image["local_path"], "sha256": image["sha256"]})
        unique.setdefault(image["sha256"], []).append(image)
    for digest, images in unique.items():
        image = images[0]
        content.append({"type": "text", "text": json.dumps({"sha256": digest,
            "applies_to": [{k: entry[k] for k in ("row_id", "role", "phase")} for entry in images]},
            ensure_ascii=profile.startswith("recovery"))})
        content.append({"type": "image_url", "image_url": {"url": "data:" + image["mime"] + ";base64," +
                       base64.b64encode(Path(image["local_path"]).read_bytes()).decode()}})
    envelope["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    return envelope


def save_requests(specs, parent, model_config, output):
    """Save validated evidence projections and their immutable request bindings."""
    previous = load(parent)
    generation = previous["manifest"]["generation"] + 1
    if generation not in (1, 2):
        raise ValueError("No reflection after R2")
    pairs = set()
    bindings = {}
    with new_directory(output, inputs=[parent]) as root:
        for spec in specs:
            key, role, pose = spec["object_key"], spec["role"], spec.get("pose")
            pair = key, role, pose
            if pair in pairs:
                raise ValueError("At most one request per object/role/pose/round")
            pairs.add(pair)
            if spec["reflection_round"] != generation or spec["parent_snapshot_sha256"] != previous["snapshot_sha256"]:
                raise ValueError("Request evidence parent/generation mismatch")
            # A digest directory avoids interpreting object names as file paths.
            item = hashlib.sha256(json_bytes([key, role, pose])).hexdigest()[:24]
            directory = root / item
            directory.mkdir()
            request = request_body(role, spec["prompt_profile"], spec["facts"], spec["roles"], model_config)
            request_path = directory / "request.body.bin"
            request_path.write_bytes(json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
            write_json(directory / "FACTS.json", spec["facts"])
            write_json(directory / "IMAGE_ROLES.json", spec["roles"])
            binding = {k: copy.deepcopy(v) for k, v in spec.items() if k not in ("facts", "roles")}
            binding["request"] = {"path": str(Path(output).resolve() / item / request_path.name), "sha256": sha256(request_path)}
            binding["facts_sha256"] = sha256(directory / "FACTS.json")
            binding["image_roles_sha256"] = sha256(directory / "IMAGE_ROLES.json")
            binding["response_model_aliases"] = model_config.get("response_model_aliases", [])
            bindings[item] = binding
        write_json(root / "REQUESTS.json", bindings)
        write_json(root / "CALL_PLAN.json", {"requests": len(bindings), "max_attempts": len(bindings),
            "automatic_retries": 0, "automatic_repair_calls": 0, "model": model_config["model"],
            "provider": model_config.get("provider"), "sent": False})
    return read_json(Path(output) / "REQUESTS.json")
