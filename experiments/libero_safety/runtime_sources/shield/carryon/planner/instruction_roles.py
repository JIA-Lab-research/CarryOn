#!/usr/bin/env python3
"""Parse task instructions into the manipulation roles consumed by grounding.

The parser receives instruction text. A failed or malformed parse returns
(None, False). SOURCE may be absent when the instruction does not specify the
target support."""
import copy
import hashlib
import json
import os
import re
import sys
import time

# Model and sampling parameters used for online role inference.
ROLE_PARSE_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
ROLE_PARSE_PROVIDER = {
    "only": ["venice/fp8"],
    "allow_fallbacks": False,
    "require_parameters": True,
}
LIVE_ROLES_MODE = "live"
ROLE_PARSE_MAX_ATTEMPTS = 3
ROLE_PARSE_RETRY_BACKOFF_S = (1.0, 3.0)
_ROLE_CACHE = {}
_LAST_ROLES_PROVENANCE = None
_LAST_API_INVOKED = False


def _roles_mode():
    mode = os.environ.get("SAFE_ROLES_MODE", "") or LIVE_ROLES_MODE
    if mode != LIVE_ROLES_MODE:
        raise ValueError("Only online instruction-role inference is supported")
    return mode


def _reject_duplicate_json_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate_json_key")
        out[key] = value
    return out


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _live_provenance(instruction=None, *, lookup="not_parsed", invoked=False,
                     status="not_parsed"):
    return {
        "roles_mode": LIVE_ROLES_MODE,
        "roles_instruction_sha256": (
            _sha256_bytes(str(instruction).encode("utf-8"))
            if instruction is not None else None
        ),
        "roles_task": None,
        "roles_lookup": lookup,
        "roles_live_api_invoked": invoked,
        "roles_status": status,
        "roles_model": ROLE_PARSE_MODEL,
        "roles_prompt_sha256": _sha256_bytes(ROLE_PARSE_SYS.encode("utf-8")),
        "roles_requested_provider": copy.deepcopy(ROLE_PARSE_PROVIDER),
        "roles_temperature": 0.0,
        "roles_max_tokens": 200,
    }


def get_roles_provenance(instruction=None):
    mode = _roles_mode()
    pending = _live_provenance(instruction)
    last = _LAST_ROLES_PROVENANCE
    if last and last["roles_mode"] == mode and (
        instruction is None
        or pending["roles_instruction_sha256"] == last["roles_instruction_sha256"]
    ):
        return copy.deepcopy(last)
    return pending

ROLE_PARSE_SYS = (
    "You extract manipulation roles from a robot instruction. Reply with EXACTLY "
    "one JSON object and nothing else:\n"
    '  {"targets": ["<noun phrase naming one object the robot must grasp or move>", ...], '
    '"source": "<the surface the target currently rests on, or null if the '
    'instruction does not say>", '
    '"destination": "<the surface or container the target must end up on or in>", '
    '"plural": true|false}\n'
    "targets is a LIST. It has one entry per DISTINCT object the instruction "
    "names as something to grasp or move. An instruction that names two different "
    "objects has two entries. An instruction that names one kind of object of "
    "which there are several has one entry.\n"
    "Each entry in targets names an object itself. Remove quantity words such as "
    "'both' or 'two', but preserve the target noun's original singular or plural "
    "form. The plural flag does not replace plural wording; never rewrite a "
    "plural noun as its singular lemma.\n"
    "source is where the target starts and destination is where it must end up. "
    "They are their own fields; neither is part of targets.\n"
    "plural is true when the instruction commands that more than one object be moved.\n"
    "Use the instruction's own words for each noun phrase. If a role is genuinely "
    "absent, use null. Do not invent objects the instruction does not mention."
)


def _emit_role_parse_infra(payload):
    """Emit only sanitized retry metadata; never key, prompt, or response text."""
    print(
        "[ROLE_PARSE_INFRA] " + json.dumps(payload, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _retryable_request_failure(status_code):
    if status_code is None:
        return True
    return status_code in {408, 409, 425, 429} or 500 <= status_code < 600


def _llm_roles(instruction):
    """Instruction-TEXT-only LLM parse -> (roles, plural) or (None, False).
    Plurality is returned SEPARATELY and never placed in the roles dict."""
    global _LAST_API_INVOKED
    _LAST_API_INVOKED = False
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        _emit_role_parse_infra({
            "attempt": 0,
            "failure_class": "MissingConfiguration",
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "api_key_presence",
            "retryable": False,
            "status_code": None,
        })
        return None, False
    try:
        import requests
    except Exception as exc:
        _emit_role_parse_infra({
            "attempt": 0,
            "failure_class": type(exc).__name__,
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "requests_import",
            "retryable": False,
            "status_code": None,
        })
        return None, False

    txt = None
    for attempt in range(1, ROLE_PARSE_MAX_ATTEMPTS + 1):
        response = None
        try:
            _LAST_API_INVOKED = True
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json={"model": ROLE_PARSE_MODEL,
                      "provider": copy.deepcopy(ROLE_PARSE_PROVIDER),
                      "messages": [{"role": "system", "content": ROLE_PARSE_SYS},
                                   {"role": "user", "content": "INSTRUCTION: " + str(instruction)}],
                      "max_tokens": 200, "temperature": 0.0},
                timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            status_code = getattr(
                getattr(exc, "response", None),
                "status_code",
                getattr(response, "status_code", None),
            )
            retryable = _retryable_request_failure(status_code)
            will_retry = retryable and attempt < ROLE_PARSE_MAX_ATTEMPTS
            _emit_role_parse_infra({
                "attempt": attempt,
                "failure_class": type(exc).__name__,
                "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
                "outcome": "retry" if will_retry else "final_failure",
                "phase": "request",
                "retryable": retryable,
                "status_code": status_code,
            })
            if not will_retry:
                return None, False
            time.sleep(ROLE_PARSE_RETRY_BACKOFF_S[attempt - 1])
            continue
        except Exception as exc:
            _emit_role_parse_infra({
                "attempt": attempt,
                "failure_class": type(exc).__name__,
                "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
                "outcome": "final_failure",
                "phase": "request",
                "retryable": False,
                "status_code": getattr(response, "status_code", None),
            })
            return None, False

        try:
            txt = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            _emit_role_parse_infra({
                "attempt": attempt,
                "failure_class": type(exc).__name__,
                "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
                "outcome": "final_failure",
                "phase": "response_decode",
                "retryable": False,
                "status_code": getattr(response, "status_code", None),
            })
            return None, False
        if attempt > 1:
            _emit_role_parse_infra({
                "attempt": attempt,
                "failure_class": None,
                "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
                "outcome": "recovered",
                "phase": "request",
                "retryable": None,
                "status_code": getattr(response, "status_code", None),
            })
        break

    m = re.search(r"\{.*\}", txt, re.DOTALL) if isinstance(txt, str) else None
    if not m:
        _emit_role_parse_infra({
            "attempt": attempt,
            "failure_class": "NoJSONObject",
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "response_json_extract",
            "retryable": False,
            "status_code": getattr(response, "status_code", None),
        })
        return None, False
    try:
        o = json.loads(m.group(0), object_pairs_hook=_reject_duplicate_json_keys)
    except Exception as exc:
        _emit_role_parse_infra({
            "attempt": attempt,
            "failure_class": type(exc).__name__,
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "response_json_parse",
            "retryable": False,
            "status_code": getattr(response, "status_code", None),
        })
        return None, False
    if (not isinstance(o, dict)
            or set(o) != {"targets", "source", "destination", "plural"}
            or type(o["plural"]) is not bool
            or (o["source"] is not None and (
                not isinstance(o["source"], str) or not o["source"].strip()))):
        _emit_role_parse_infra({
            "attempt": attempt,
            "failure_class": "InvalidRoleSchema",
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "schema_validation",
            "retryable": False,
            "status_code": getattr(response, "status_code", None),
        })
        return None, False
    tgts, dst = o.get("targets"), o.get("destination")
    if not isinstance(tgts, list) or not tgts:
        _emit_role_parse_infra({
            "attempt": attempt,
            "failure_class": "InvalidTargetsContainer",
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "schema_validation",
            "retryable": False,
            "status_code": getattr(response, "status_code", None),
        })
        return None, False
    targets, seen = [], set()
    for target in tgts:
        if not isinstance(target, str) or not target.strip():
            _emit_role_parse_infra({
                "attempt": attempt,
                "failure_class": "InvalidTargetItem",
                "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
                "outcome": "final_failure",
                "phase": "schema_validation",
                "retryable": False,
                "status_code": getattr(response, "status_code", None),
            })
            return None, False
        target = target.strip().lower()
        if target in seen:
            _emit_role_parse_infra({
                "attempt": attempt,
                "failure_class": "DuplicateTarget",
                "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
                "outcome": "final_failure",
                "phase": "schema_validation",
                "retryable": False,
                "status_code": getattr(response, "status_code", None),
            })
            return None, False
        seen.add(target)
        targets.append(target)
    if not isinstance(dst, str) or not dst.strip():
        _emit_role_parse_infra({
            "attempt": attempt,
            "failure_class": "InvalidDestination",
            "max_attempts": ROLE_PARSE_MAX_ATTEMPTS,
            "outcome": "final_failure",
            "phase": "schema_validation",
            "retryable": False,
            "status_code": getattr(response, "status_code", None),
        })
        return None, False
    src = o.get("source")
    src = src.strip() if isinstance(src, str) and src.strip() else None
    roles = {"TARGET": targets, "SOURCE": src.lower() if src else None,
             "DESTINATION": dst.strip().lower()}
    return roles, bool(o.get("plural"))



def parse_roles(instruction):
    """Parse manipulation roles online using the configured model.

    Cache successful parses by exact instruction. Return the model's role phrases
    without noun or article rewriting."""
    global _LAST_ROLES_PROVENANCE
    _roles_mode()
    instruction = str(instruction)
    if instruction in _ROLE_CACHE:
        _LAST_ROLES_PROVENANCE = _live_provenance(
            instruction, lookup="process_cache", status="ok")
        return copy.deepcopy(_ROLE_CACHE[instruction])
    roles, plural = _llm_roles(instruction)
    _LAST_ROLES_PROVENANCE = _live_provenance(
        instruction, lookup="live_api", invoked=_LAST_API_INVOKED,
        status="ok" if roles is not None else "unavailable")
    if roles is None:
        # A temporary service failure must not poison all later episodes.
        return None, False
    assert set(roles) == {"TARGET", "SOURCE", "DESTINATION"}, "role dict leaked extra keys"
    out = (roles, plural)
    _ROLE_CACHE[instruction] = copy.deepcopy(out)
    return out
