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
    """

    def __init__(self, out: TextIO | None=None, prompt: str='carryon> ') -> None:
        self.out = out if out is not None else sys.stdout
        self.prompt = prompt
        self._lock = asyncio.Lock()

    async def info(self, line: str) -> None:
        await self._emit(line)

    async def job_update(self, plan_id: str, subtask_id: str, endpoint_name: str, status: JobStatus, brief: str='') -> None:
        tag = f'[{plan_id} {subtask_id} @ {endpoint_name}]'
        msg = f'{tag} {status}' + (f': {brief}' if brief else '')
        await self._emit(msg)

    async def plan_finished(self, plan_id: str, summary: str) -> None:
        await self._emit(f'[{plan_id} done] {summary}')

    async def _emit(self, line: str) -> None:
        async with self._lock:
            try:
                if self.out.isatty():
                    self.out.write('\r\x1b[2K' + line + '\n' + self.prompt)
                else:
                    self.out.write(line + '\n')
                self.out.flush()
            except Exception:
                pass
