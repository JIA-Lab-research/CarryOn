"""Built-in ``wait`` skill: pause for a real-world duration between subtasks.

Used by the Planner when a recipe demands a delay (boil 3 minutes, let X
cool down, etc.). Honors ``asyncio.CancelledError`` so /cancel and plan
cancellation work as expected.
"""

from __future__ import annotations

import asyncio
import time

from ..types import ToolResult


class WaitSkill:
    name = "wait"

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        raw = inputs or {}
        duration = _coerce_float(raw.get("duration_s"))
        if "until_ts" in raw:
            until = _coerce_float(raw.get("until_ts"))
            if until is not None:
                duration = max(0.0, until - time.time())
        if duration is None or duration < 0:
            return ToolResult(
                ok=False,
                error=(
                    "wait skill requires inputs.duration_s (float seconds) or "
                    "inputs.until_ts (unix seconds)"
                ),
            )
        started = time.time()
        if duration > 0:
            await asyncio.sleep(duration)
        return ToolResult(ok=True, data={
            "waited_s": time.time() - started,
            "requested_s": duration,
        })


def _coerce_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
