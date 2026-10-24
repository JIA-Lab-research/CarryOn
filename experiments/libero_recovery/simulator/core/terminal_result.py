"""Single-owner, atomic terminal records for LIBERO recovery evaluation.

The writer owns the complete terminal triple. Callers select a reason; they
cannot independently supply ``success`` or ``timeout`` and therefore cannot
land a partial or internally inconsistent terminal record.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Optional


@dataclass(frozen=True)
class TerminalOutcome:
    success: bool
    timeout: bool
    terminal_reason: str


TERMINAL_SEMANTICS: dict[str, TerminalOutcome] = {
    "normal_success": TerminalOutcome(True, False, "normal_success"),
    "normal_failure": TerminalOutcome(False, False, "normal_failure"),
    "env_timeout": TerminalOutcome(False, True, "env_timeout"),
    "wall_cap": TerminalOutcome(False, True, "wall_cap"),
    "cancel": TerminalOutcome(False, False, "cancel"),
    "cap_refusal": TerminalOutcome(False, False, "cap_refusal"),
    "handback_failure": TerminalOutcome(False, False, "handback_failure"),
    "exception": TerminalOutcome(False, False, "exception"),
}

_TERMINAL_KEYS = frozenset({"success", "timeout", "terminal_reason"})


class TerminalResultWriter:
    """Own and atomically persist exactly one episode terminal outcome."""

    def __init__(
        self,
        path: Optional[str],
        payload_factory: Callable[[], Mapping[str, object]],
    ) -> None:
        self._path = pathlib.Path(path) if path else None
        self._payload_factory = payload_factory
        self._outcome: Optional[TerminalOutcome] = None
        self._written = False
        self._lock = threading.RLock()

    @property
    def outcome(self) -> Optional[TerminalOutcome]:
        with self._lock:
            return self._outcome

    @property
    def written(self) -> bool:
        with self._lock:
            return self._written

    def finalize(self, terminal_reason: str, *, write: bool = True) -> TerminalOutcome:
        """Bind one reason and optionally commit the complete record.

        Repeating the same terminal is idempotent. A competing terminal is a
        programming error: first terminal wins, and no second path may rewrite it.
        """
        try:
            candidate = TERMINAL_SEMANTICS[str(terminal_reason)]
        except KeyError as exc:
            raise ValueError(f"unknown terminal_reason: {terminal_reason!r}") from exc
        with self._lock:
            if self._outcome is None:
                self._outcome = candidate
            elif self._outcome != candidate:
                raise RuntimeError(
                    "terminal already finalized as "
                    f"{self._outcome.terminal_reason}; refusing {terminal_reason}"
                )
            if write:
                self._commit_locked()
            return self._outcome

    def commit(self) -> None:
        with self._lock:
            if self._outcome is None:
                raise RuntimeError("cannot write a result before terminal finalization")
            self._commit_locked()

    def _commit_locked(self) -> None:
        if self._written or self._path is None:
            self._written = True
            return
        assert self._outcome is not None
        payload = dict(self._payload_factory())
        forbidden = _TERMINAL_KEYS.intersection(payload)
        if forbidden:
            raise ValueError(
                "payload_factory must not own terminal keys: "
                + ", ".join(sorted(forbidden))
            )
        payload.update({
            "success": self._outcome.success,
            "timeout": self._outcome.timeout,
            "terminal_reason": self._outcome.terminal_reason,
        })
        if not isinstance(payload["success"], bool):
            raise TypeError("terminal success must be bool")
        if not isinstance(payload["timeout"], bool):
            raise TypeError("terminal timeout must be bool")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, self._path)
            self._written = True
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
