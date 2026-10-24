from __future__ import annotations

from typing import Protocol

from ..endpoints import RobotEndpoint
from ..types import SubmitResult


class Transport(Protocol):
    name: str

    async def send(
        self,
        endpoint: RobotEndpoint,
        instruction: str,
        inputs: dict,
        meta: dict,
    ) -> SubmitResult: ...

    async def cancel(
        self,
        endpoint: RobotEndpoint,
        request_id: str,
        *,
        return_home: bool = False,
    ) -> bool:
        """Best-effort cancel. ``return_home`` is forwarded to the executor
        (e.g. via HTTP body ``{"return_home": <bool>}``) so it can decide
        whether to drive the arm back to its init pose after braking.
        Transports that don't support a body (e.g. ROS topic publish) MAY
        ignore the flag, but should still log it.
        """
        ...

    async def aclose(self) -> None: ...
