"""One-attempt model transport with a conservative, durable cost reservation."""
from __future__ import annotations
from decimal import Decimal
import json
import os
from pathlib import Path
import urllib.parse

from ..files import bound_file, new_directory, read_json, sha256, write_json


def send(request_directory, config, output):
    import httpx
    request_directory = Path(request_directory).resolve()
    bindings = read_json(request_directory / "REQUESTS.json")
    if not bindings:
        with new_directory(output, inputs=[request_directory]) as root:
            write_json(root / "RESPONSES.json", [])
            write_json(root / "COST.json", {"attempts": 0, "cost_upper_bound_usd": "0",
                "ledger": [], "complete": True, "reason": "NO_QUALIFYING_FAILURE_EVIDENCE"})
        return []
    endpoint = config["endpoint"]
    url = urllib.parse.urlsplit(endpoint)
    local = config.get("transport") == "local"
    if local:
        if url.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Local inference must use an explicit loopback endpoint")
    elif url.scheme != "https":
        raise ValueError("Remote inference requires HTTPS")
    maximum = config["max_requests"]
    if type(maximum) is not int or maximum < 0:
        raise ValueError("max_requests must be a nonnegative integer")
    budget = Decimal(str(config.get("budget_usd", 0)))
    ceiling = Decimal(str(config.get("per_request_ceiling_usd", 0)))
    if not budget.is_finite() or not ceiling.is_finite() or budget < 0 or ceiling < 0:
        raise ValueError("Invalid cost limit")
    if not local and ceiling <= 0:
        raise ValueError("Remote calls require a total budget and a conservative per-request cost ceiling")
    if maximum==0 or (not local and budget<ceiling):
        responses=[{"item":item,"request_sha256":batch["request"]["sha256"],"fixture":False,
                    "status":"resource_truncated","error":"Request or cost cap reached; no call made"}
                   for item,batch in bindings.items()]
        with new_directory(output,inputs=[request_directory]) as root:
            write_json(root/"RESPONSES.json",responses)
            write_json(root/"COST.json",{"attempts":0,"max_requests":maximum,
                "cost_upper_bound_usd":"0","budget_usd":str(budget),"ledger":[],
                "complete":False,"reason":"REQUEST_OR_COST_CAP"})
        return responses
    headers = {"Content-Type": "application/json"}
    key_env = config.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise ValueError("Required credential environment variable is unset: " + key_env)
        headers["Authorization"] = "Bearer " + key
    elif not local:
        raise ValueError("Remote inference requires an explicit credential environment name")
    responses, ledger = [], []
    spent, attempted = Decimal(0), 0
    halted = False
    with new_directory(output, inputs=[request_directory]) as root:
        # Disable HTTP retries and implicit proxy/environment routing.
        with httpx.Client(timeout=config.get("timeout_s", 180), follow_redirects=False,
                          transport=httpx.HTTPTransport(retries=0), trust_env=False) as client:
            for item, batch in bindings.items():
                base = {"item": item, "request_sha256": batch["request"]["sha256"], "fixture": False}
                if halted or attempted >= maximum or (not local and spent + ceiling > budget):
                    responses.append({**base, "status": "resource_truncated", "error": "Request or cost cap reached; no call made"})
                    continue
                path = bound_file(batch["request"])
                request = read_json(path)
                if request["model"] != config["model"] or request.get("provider") != config.get("provider"):
                    raise ValueError("Prepared request model/provider differs from the sending configuration")
                attempted += 1
                reserved = Decimal(0) if local else ceiling
                spent += reserved
                directory = root / item
                directory.mkdir()
                # Record before opening the connection. Unknown outcomes consume the reservation.
                write_json(directory / "ATTEMPT.json", {**base, "attempt": attempted,
                    "reserved_usd": str(reserved), "cumulative_upper_bound_usd": str(spent),
                    "endpoint_host": url.hostname, "transport": "local" if local else "remote", "automatic_retries": 0})
                status, error, usage = "completed", None, None
                response_path = directory / "response.body.bin"
                try:
                    response = client.post(endpoint, content=path.read_bytes(), headers=headers)
                    response_path.write_bytes(response.content)
                    response.raise_for_status()
                    value = response.json()
                    if not isinstance(value.get("choices"), list) or not value["choices"]:
                        raise ValueError("Model service did not return a choice")
                    content = value["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Empty model response")
                    if value.get("model") not in {config["model"], *config.get("response_model_aliases", [])}:
                        raise ValueError("Returned model identity differs from the configured model")
                    if not local:
                        providers = config.get("response_provider_aliases", [])
                        if not providers or value.get("provider") not in providers:
                            raise ValueError("Returned provider identity was not admitted")
                    usage = value.get("usage")
                    cost = usage.get("cost") if isinstance(usage, dict) else None
                    if cost is not None and not local:
                        actual = Decimal(str(cost))
                        if not actual.is_finite() or actual < 0:
                            raise ValueError("Invalid service cost")
                        spent += actual - reserved
                        if actual > ceiling:
                            halted = True
                            raise ValueError("Service exceeded configured per-request cost ceiling; further calls stopped")
                except Exception as exc:
                    # Retain the response bytes, but never persist request authorization headers.
                    status = "service_error"
                    error = type(exc).__name__
                    if isinstance(exc, ValueError):
                        error += ": " + str(exc)
                record = {**base, "status": status, "error": error, "usage": usage}
                if response_path.exists():
                    record["response"] = {"path": str(Path(output).resolve() / item / response_path.name),
                                          "sha256": sha256(response_path)}
                write_json(directory / "RESULT.json", record)
                responses.append(record)
                ledger.append({"item": item, "status": status, "usage": usage,
                               "cumulative_upper_bound_usd": str(spent)})
        write_json(root / "RESPONSES.json", responses)
        write_json(root / "COST.json", {"attempts": attempted, "max_requests": maximum,
            "cost_upper_bound_usd": str(spent), "budget_usd": str(budget),
            "per_request_ceiling_usd": str(ceiling), "ledger": ledger,
            "complete": all(r["status"] == "completed" for r in responses)})
    return responses
