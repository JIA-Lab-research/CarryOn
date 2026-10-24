from __future__ import annotations

import asyncio
import sys
from typing import TextIO

from ..types import JobStatus


class Notifier:
    """Prints async notification lines without fighting the user's prompt.

    If stdout is a TTY and `readline` is available, we clear the current line,
    print the message, then let the REPL redraw the prompt on next input cycle.
    Otherwise we just print (good enough for pipes / tests).

    When ``verbose`` is false, only user-facing lines (``say``), plan-finished
    summaries, and ``info(..., important=True)`` errors are printed; dispatch /
    reset / reactor chatter is suppressed.

    While ``busy`` is true (a user turn is still in flight), messages are printed
    *without* redrawing ``carryon> `` so the prompt only returns after the turn
    finishes.
    """

    def __init__(
        self,
        out: TextIO | None = None,
        prompt: str = "carryon> ",
        *,
        verbose: bool = False,
    ) -> None:
        self.out = out if out is not None else sys.stdout
        self.prompt = prompt
        self.verbose = verbose
        self._busy = False
        self._lock = asyncio.Lock()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)

    async def say(self, line: str) -> None:
        """Always-visible user-facing line (e.g. planner ``user_reply``)."""
        await self._emit(line)

    async def info(self, line: str, *, important: bool = False) -> None:
        if not self.verbose and not important:
            return
        await self._emit(line)

    async def job_update(
        self,
        plan_id: str,
        subtask_id: str,
        endpoint_name: str,
        status: JobStatus,
        brief: str = "",
    ) -> None:
        if not self.verbose:
            return
        tag = f"[{plan_id} {subtask_id} @ {endpoint_name}]"
        msg = f"{tag} {status}" + (f": {brief}" if brief else "")
        await self._emit(msg)

    async def plan_finished(self, plan_id: str, summary: str) -> None:
        await self._emit(f"[{plan_id} done] {summary}")

    async def _emit(self, line: str) -> None:
        async with self._lock:
            try:
                redraw = "" if self._busy else self.prompt
                if self.out.isatty():
                    # Clear the current line (which currently shows the REPL
                    # prompt + any half-typed input), print the async message,
                    # then redraw the prompt only when the REPL is idle.
                    self.out.write("\r\x1b[2K" + line + "\n" + redraw)
                else:
                    self.out.write(line + "\n")
                self.out.flush()
            except Exception:
                # Never raise from the notifier.
                pass
