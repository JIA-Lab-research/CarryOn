"""Collect jam evidence and request bounded grasp-height corrections."""
from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from .semantic_memory import FORBIDDEN_DECISION_FIELDS


SIGNAL_FIELDS = frozenset({
    "object_key",
    "planned_mm",
    "moved_mm",
    "dpos_mm",
    "stuck_z_m",
    "current_z_m",
})
RESPONSE_FIELDS = frozenset({"corrected_z_m", "rationale"})
OPENROUTER_ENVELOPE = {
    "model": "qwen/qwen3-vl-235b-a22b-instruct",
    "temperature": 0.0,
    "top_p": 0.8,
    "max_tokens": 1400,
    "provider": {
        "data_collection": "deny",
    },
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MIN_Z_M = 0.0
MAX_Z_M = 0.25
MAX_ABS_Z_DELTA_M = 0.12


class JamZClient(Protocol):
    def complete(self, prompt: str, envelope: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class JamResidualSignal:
    object_key: str
    planned_mm: float
    moved_mm: float
    dpos_mm: float
    stuck_z_m: float
    current_z_m: float

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        if set(value) != SIGNAL_FIELDS:
            raise ValueError("jam signal schema drift")
        key = str(value["object_key"]).strip().lower()
        if not key:
            raise ValueError("object_key is required")
        value["object_key"] = key
        for name in SIGNAL_FIELDS - {"object_key"}:
            number = value[name]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"{name} must be numeric")
            number = float(number)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            value[name] = number
        if value["planned_mm"] < 0 or value["moved_mm"] < 0 or value["dpos_mm"] < 0:
            raise ValueError("millimetre residuals must be nonnegative")
        if not MIN_Z_M <= value["current_z_m"] <= MAX_Z_M:
            raise ValueError("current_z_m outside physical bounds")
        if not MIN_Z_M <= value["stuck_z_m"] <= MAX_Z_M + 0.20:
            raise ValueError("stuck_z_m outside physical bounds")
        if FORBIDDEN_DECISION_FIELDS.intersection(value):
            raise ValueError("forbidden decision field in jam signal")
        return value


@dataclass(frozen=True)
class JamZDecision:
    applied: bool
    corrected_z_m: float
    source: str
    rationale: str = ""
    parse_error: Optional[str] = None
    latch_reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenRouterJamZClient:
    """Synchronous adapter that requires the parameter transport retry handler."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        timeout_s: float = 60.0,
        client: Any = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter api_key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._client = client
        self.calls = 0

    async def _complete_async(self, prompt: str, envelope: dict[str, Any]) -> str:
        from ..providers.openrouter import OpenRouterChatClient

        frozen = json.loads(json.dumps(envelope))
        if frozen != OPENROUTER_ENVELOPE:
            raise ValueError("grasp-height reflector OpenRouter envelope mismatch")
        messages = [
            {
                "role": "system",
                "content": (
                    "You adjust only the grasp target z behaviour parameter after "
                    "a classified physical descent jam. Return strict JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        client = self._client or OpenRouterChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_s=self.timeout_s,
        )
        owns_client = self._client is None
        try:
            chat_envelope = getattr(client, "chat_envelope", None)
            if not callable(chat_envelope) or not (
                getattr(chat_envelope, "__carryon_parameter_transport_retry__", False)
            ):
                raise RuntimeError("grasp-height parameter transport retry handler is not installed")
            result = await chat_envelope(envelope=frozen, messages=messages)
            text = getattr(result, "text", None)
            if not isinstance(text, str):
                raise RuntimeError("grasp-height transport returned a non-text result")
            return text
        finally:
            if owns_client:
                await client.aclose()

    def complete(self, prompt: str, envelope: dict[str, Any]) -> str:
        self.calls += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("synchronous grasp-height reflector called from a running event loop")
        return asyncio.run(self._complete_async(prompt, envelope))


class JamZReflector:
    """One-run VLM reflector with per-object corrected-z latch."""

    def __init__(
        self,
        client: JamZClient,
        *,
        ledger_path: Optional[Path] = None,
        min_z_m: float = MIN_Z_M,
        max_z_m: float = MAX_Z_M,
        max_abs_delta_m: float = MAX_ABS_Z_DELTA_M,
    ) -> None:
        self.client = client
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.min_z_m = float(min_z_m)
        self.max_z_m = float(max_z_m)
        self.max_abs_delta_m = float(max_abs_delta_m)
        self._latch: dict[str, JamZDecision] = {}
        if self.ledger_path is not None:
            self._initialize_ledger()

    @property
    def latch(self) -> dict[str, JamZDecision]:
        return dict(self._latch)

    def reflect(
        self,
        signal: JamResidualSignal,
        *,
        target_before_xyz_m: Optional[list[float]] = None,
    ) -> JamZDecision:
        try:
            payload = signal.payload()
        except Exception as exc:
            return self._fallback(signal.current_z_m, f"invalid_signal: {exc}")

        key = payload["object_key"]
        latched = self._latch.get(key)
        if latched is not None:
            decision = JamZDecision(
                applied=True,
                corrected_z_m=latched.corrected_z_m,
                source="vlm_run_latch",
                rationale=latched.rationale,
                latch_reused=True,
            )
            self._write_ledger(payload, decision, target_before_xyz_m, raw_response=None)
            return decision

        prompt = self._prompt(payload)
        try:
            raw = self.client.complete(prompt, json.loads(json.dumps(OPENROUTER_ENVELOPE)))
            decision = self._parse(raw, current_z_m=payload["current_z_m"])
        except Exception as exc:
            decision = self._fallback(
                payload["current_z_m"],
                f"vlm_or_parse_failure: {type(exc).__name__}: {exc}",
            )
            raw = None
        if decision.applied:
            self._latch[key] = decision
        self._write_ledger(payload, decision, target_before_xyz_m, raw_response=raw)
        return decision

    def _prompt(self, payload: dict[str, Any]) -> str:
        if set(payload) != SIGNAL_FIELDS:
            raise ValueError("prompt payload schema drift")
        if FORBIDDEN_DECISION_FIELDS.intersection(payload):
            raise ValueError("forbidden decision field in prompt payload")
        return (
            "A z-axis descent jam has already been classified by fixed mechanics. "
            "Read the residual telemetry and choose one corrected grasp target z in "
            "metres. You may change only z; do not propose thresholds or verification. "
            "Return exactly one JSON object with keys corrected_z_m (number) and "
            "rationale (short string), with no markdown. SIGNAL="
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )

    def _parse(self, raw: str, *, current_z_m: float) -> JamZDecision:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("response must be a JSON object")
        if set(value) != RESPONSE_FIELDS:
            raise ValueError("response has unknown or missing fields")
        corrected = value["corrected_z_m"]
        if isinstance(corrected, bool) or not isinstance(corrected, (int, float)):
            raise ValueError("corrected_z_m must be numeric")
        corrected = float(corrected)
        if not math.isfinite(corrected):
            raise ValueError("corrected_z_m must be finite")
        if not self.min_z_m <= corrected <= self.max_z_m:
            raise ValueError("corrected_z_m outside physical bounds")
        if abs(corrected - float(current_z_m)) > self.max_abs_delta_m:
            raise ValueError("corrected_z_m delta outside physical bounds")
        rationale = value["rationale"]
        if not isinstance(rationale, str) or len(rationale) > 240:
            raise ValueError("rationale must be a short string")
        return JamZDecision(
            applied=True,
            corrected_z_m=corrected,
            source="vlm_runtime",
            rationale=rationale,
        )

    @staticmethod
    def _fallback(current_z_m: float, error: str) -> JamZDecision:
        return JamZDecision(
            applied=False,
            corrected_z_m=float(current_z_m),
            source="mechanical_fallback",
            parse_error=error,
        )

    def _initialize_ledger(self) -> None:
        self._append_ledger_row(
            {
                "schema": "carryon-row6-vlm-z-parameter-memory-ledger-init/v2",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "event": "REFLECTOR_INITIALIZED",
                "reflector_active": True,
                "episode_local_latch_empty": len(self._latch) == 0,
                "vlm_calls_at_initialization": int(getattr(self.client, "calls", 0)),
            },
            exclusive=True,
        )

    def _append_ledger_row(self, row: dict[str, Any], *, exclusive: bool) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "x" if exclusive else "a"
        with self.ledger_path.open(mode, encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_ledger(
        self,
        payload: dict[str, Any],
        decision: JamZDecision,
        target_before_xyz_m: Optional[list[float]],
        *,
        raw_response: Optional[str],
    ) -> None:
        if self.ledger_path is None:
            return
        before = list(target_before_xyz_m) if target_before_xyz_m is not None else [
            None,
            None,
            payload["current_z_m"],
        ]
        after = list(before)
        after[2] = decision.corrected_z_m
        row = {
            "schema": "carryon-row6-vlm-z-parameter-memory-application/v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "applied": decision.applied,
            "object_key": payload["object_key"],
            "prior_z_m": decision.corrected_z_m - payload["current_z_m"],
            "target_before_xyz_m": before,
            "target_after_prior_xyz_m": after,
            "value_source": decision.source,
            "latch_reused": decision.latch_reused,
            "jam_signal": payload,
            "vlm_response_raw": raw_response,
            "parse_error": decision.parse_error,
        }
        self._append_ledger_row(row, exclusive=False)


__all__ = [
    "JamResidualSignal",
    "JamZDecision",
    "JamZReflector",

    "OpenRouterJamZClient",
    "OPENROUTER_ENVELOPE",
    "SIGNAL_FIELDS",
]

