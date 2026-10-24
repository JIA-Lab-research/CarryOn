from __future__ import annotations
from typing import Protocol
from ...types import OnResult

class ResultChannel(Protocol):

    async def start(self, on_result: OnResult) -> None:
        ...

    async def stop(self) -> None:
        ...

    def external_url(self, job_key: str) -> str:
        ...
