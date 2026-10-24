"""Read-only semantic-memory routing and deterministic SoM identities.

This module deliberately contains no robot action and no online write path.  It
validates an immutable semantic snapshot, chooses at most one finite template,
and supplies byte-stable identities used by the runtime ledger.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


BASELINE_TEMPLATE = "BASELINE_BPRIME_V1"
SIM_TEMPLATE = "SIM_FC_SOM_REVERIFY_V1"
REAL_TEMPLATE = "REAL_FC_SOM_ABSENT_V1"
ALLOWED_TEMPLATES = {BASELINE_TEMPLATE, SIM_TEMPLATE, REAL_TEMPLATE}
ALLOWED_STATES = {
    "PROPOSED", "CANDIDATE_FROZEN", "EXPERIMENT_ONLY",
    "ADMITTED", "REJECTED", "REVOKED",
}
ROUTABLE_STATES = {"CANDIDATE_FROZEN", "EXPERIMENT_ONLY", "ADMITTED"}
FORBIDDEN_DECISION_FIELDS = {
    "task_id", "init_id", "init_state_index", "success", "terminal_state",
    "simulator_state", "hidden_simulator_state", "hints", "free_text_plan",
    "numeric_offset",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def array_sha256(value: Any) -> str:
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": str(arr.dtype), "shape": list(arr.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return sha256_bytes(header + b"\0" + arr.tobytes(order="C"))


def rgb_sha256(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return sha256_bytes(bytes(value))
    return array_sha256(value)


def mask_sha256(mask: Any) -> str:
    import numpy as np

    return array_sha256(np.asarray(mask, dtype=np.bool_))


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:  # pragma: no cover - numpy is present in runtime/tests
        pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"intrinsics contain unsupported value {type(value).__name__}")


def scene_bundle_identity(
    rgb: Any, depth_m: Any, intrinsics: Any, *, captured_unix_ns: int,
) -> dict:
    """Identity for one perceive call; contains no task/outcome/state privilege."""
    intrinsics_json = _jsonable(intrinsics)
    result = {
        "schema": "carryon-scene-bundle-identity/v1",
        "captured_unix_ns": int(captured_unix_ns),
        "rgb_sha256": rgb_sha256(rgb),
        "depth_sha256": array_sha256(depth_m),
        "intrinsics_sha256": canonical_json_sha256(intrinsics_json),
    }
    result["bundle_sha256"] = canonical_json_sha256(result)
    return result


def deterministic_permutation(
    count: int, *, formal_seed: int, episode_key: str,
    semantic_call_index: int, prompt_schema_sha256: str, round_index: int,
) -> tuple[list[int], int]:
    material = {
        "formal_seed": int(formal_seed),
        "episode_key": str(episode_key),
        "semantic_call_index": int(semantic_call_index),
        "prompt_schema_sha256": str(prompt_schema_sha256),
        "round_index": int(round_index),
    }
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    derived = int.from_bytes(digest[:8], byteorder="big", signed=False)
    order = list(range(int(count)))
    random.Random(derived).shuffle(order)
    return order, derived


@dataclass(frozen=True)
class SemanticRoute:
    template_id: str
    entry_id: Optional[str]
    match_count: int
    reason: str


class SemanticMemorySnapshot:
    """Strict immutable snapshot.  Matching never sees task/init/outcome fields."""

    def __init__(self, path: str):
        self.path = Path(path).resolve()
        raw_bytes = self.path.read_bytes()
        self.sha256 = sha256_bytes(raw_bytes)
        raw = json.loads(raw_bytes.decode("utf-8"))
        if set(raw) != {"schema", "template_registry_sha256", "entries"}:
            raise ValueError("semantic snapshot has unknown or missing top-level fields")
        if raw["schema"] != "carryon-semantic-memory-snapshot/v1":
            raise ValueError("unsupported semantic snapshot schema")
        registry_sha = str(raw["template_registry_sha256"])
        if len(registry_sha) != 64:
            raise ValueError("invalid template_registry_sha256")
        int(registry_sha, 16)
        self.template_registry_sha256 = registry_sha
        entries = raw["entries"]
        if not isinstance(entries, list):
            raise ValueError("semantic snapshot entries must be a list")
        self.entries = tuple(self._validate_entry(item) for item in entries)
        ids = [item["entry_id"] for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate semantic entry_id")

    @staticmethod
    def _validate_entry(item: Any) -> dict:
        if not isinstance(item, dict):
            raise ValueError("semantic entry must be an object")
        allowed = {
            "schema", "entry_id", "lifecycle_state", "environment",
            "applicability", "template_id", "evidence", "consolidation_gate",
            "created_utc", "expires_utc",
        }
        required = allowed - {"expires_utc"}
        if set(item) - allowed or not required.issubset(item):
            raise ValueError("semantic entry has unknown or missing fields")
        if FORBIDDEN_DECISION_FIELDS.intersection(item):
            raise ValueError("forbidden runtime decision field in semantic entry")
        if item["schema"] != "carryon-semantic-memory-entry/v1":
            raise ValueError("unsupported semantic entry schema")
        if item["lifecycle_state"] not in ALLOWED_STATES:
            raise ValueError("invalid semantic lifecycle state")
        if item["environment"] not in {"sim", "real"}:
            raise ValueError("invalid semantic environment")
        if item["template_id"] not in ALLOWED_TEMPLATES:
            raise ValueError("unknown semantic template")
        app = item["applicability"]
        if not isinstance(app, dict) or set(app) != {
            "target_object_key", "reference_sha256", "required_feature_flags",
            "use_hints",
        }:
            raise ValueError("invalid applicability schema")
        if app["use_hints"] is not False:
            raise ValueError("semantic use_hints must be false")
        if app["required_feature_flags"] != []:
            raise ValueError("v1 admits no scene-feature detector flags")
        ref_sha = str(app["reference_sha256"])
        if len(ref_sha) != 64:
            raise ValueError("invalid semantic reference SHA")
        int(ref_sha, 16)
        if item["environment"] == "sim" and item["template_id"] == REAL_TEMPLATE:
            raise ValueError("real template cannot be bound to sim entry")
        if item["environment"] == "real" and item["template_id"] == SIM_TEMPLATE:
            raise ValueError("sim template cannot be bound to real entry")
        return json.loads(json.dumps(item, sort_keys=True))

    def route(
        self, *, environment: str, target_object_key: str,
        reference_sha256: str,
    ) -> SemanticRoute:
        matches = []
        for item in self.entries:
            app = item["applicability"]
            if item["lifecycle_state"] not in ROUTABLE_STATES:
                continue
            if item["environment"] != environment:
                continue
            if app["target_object_key"] != target_object_key:
                continue
            if app["reference_sha256"] != reference_sha256:
                continue
            matches.append(item)
        if not matches:
            return SemanticRoute(BASELINE_TEMPLATE, None, 0, "NO_MEMORY_MATCH")
        if len(matches) != 1:
            return SemanticRoute(
                BASELINE_TEMPLATE, None, len(matches), "AMBIGUOUS_MEMORY_MATCH",
            )
        entry = matches[0]
        template = entry["template_id"]
        if template == REAL_TEMPLATE:
            return SemanticRoute(
                REAL_TEMPLATE, entry["entry_id"], 1, "REAL_TEMPLATE_DISABLED",
            )
        return SemanticRoute(template, entry["entry_id"], 1, "EXACT_MEMORY_MATCH")
