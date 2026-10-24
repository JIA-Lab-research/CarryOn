"""Fail-closed placement endpoint confirmation.

The visual judge's first success is provisional.  A terminal success is exposed
only after a second, fresh, released/contained/stable observation acquired after
the configured settle interval.  This module deliberately has no benchmark or
environment-predicate dependency.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementSuccessObservation:
    frame_identity: str
    captured_at_s: float
    producer_frame_seq: int | None
    producer_frame_ts: float | None
    simulation_step_id: int | None
    simulation_time_s: float | None
    camera_role: str
    camera_source: str
    judge_success: bool
    target_released: bool
    target_in_destination: bool
    stable: bool
    job_terminal: bool = False


@dataclass(frozen=True)
class PlacementSuccessDecision:
    provisional: bool
    finalize: bool
    clear_provisional: bool
    reason: str


def is_placement_instruction(instruction: str) -> bool:
    text = str(instruction or "").lower()
    return bool(
        re.search(r"\b(?:place|put|drop)\b", text)
        and re.search(r"\b(?:in|into|inside|on|onto)\b", text)
    )


class PlacementSuccessGate:
    def __init__(self, *, min_settle_interval_s: float = 1.0) -> None:
        interval = float(min_settle_interval_s)
        if interval < 0:
            raise ValueError("min_settle_interval_s must be non-negative")
        self.min_settle_interval_s = interval
        self._first: PlacementSuccessObservation | None = None
        self._finalized = False

    @staticmethod
    def _finite_number(value) -> bool:
        return bool(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    @staticmethod
    def _qualifies(obs: PlacementSuccessObservation) -> bool:
        return bool(
            obs.judge_success
            and obs.frame_identity
            and isinstance(obs.producer_frame_seq, int)
            and not isinstance(obs.producer_frame_seq, bool)
            and obs.producer_frame_seq >= 0
            and PlacementSuccessGate._finite_number(obs.producer_frame_ts)
            and isinstance(obs.simulation_step_id, int)
            and not isinstance(obs.simulation_step_id, bool)
            and obs.simulation_step_id >= 0
            and PlacementSuccessGate._finite_number(obs.simulation_time_s)
            and bool(obs.camera_role)
            and bool(obs.camera_source)
            and obs.target_released
            and obs.target_in_destination
            and obs.stable
        )

    def observe(self, obs: PlacementSuccessObservation) -> PlacementSuccessDecision:
        if obs.job_terminal or self._finalized:
            return PlacementSuccessDecision(
                provisional=False,
                finalize=False,
                clear_provisional=False,
                reason="job already terminal",
            )
        if not self._qualifies(obs):
            cleared = self._first is not None
            self._first = None
            return PlacementSuccessDecision(
                provisional=False,
                finalize=False,
                clear_provisional=cleared,
                reason="success evidence incomplete; provisional cleared",
            )
        if self._first is None:
            self._first = obs
            return PlacementSuccessDecision(
                provisional=True,
                finalize=False,
                clear_provisional=False,
                reason="first qualifying success is provisional",
            )
        first = self._first
        if (
            obs.camera_role != first.camera_role
            or obs.camera_source != first.camera_source
        ):
            self._first = None
            return PlacementSuccessDecision(
                provisional=False,
                finalize=False,
                clear_provisional=True,
                reason="camera role/source changed; provisional cleared",
            )
        if obs.producer_frame_seq <= first.producer_frame_seq:
            return PlacementSuccessDecision(
                provisional=True,
                finalize=False,
                clear_provisional=False,
                reason="producer frame sequence did not advance",
            )
        if float(obs.producer_frame_ts) <= float(first.producer_frame_ts):
            return PlacementSuccessDecision(
                provisional=True,
                finalize=False,
                clear_provisional=False,
                reason="producer frame timestamp did not advance",
            )
        if obs.simulation_step_id <= first.simulation_step_id:
            return PlacementSuccessDecision(
                provisional=True,
                finalize=False,
                clear_provisional=False,
                reason="simulation step did not advance",
            )
        if float(obs.simulation_time_s) <= float(first.simulation_time_s):
            return PlacementSuccessDecision(
                provisional=True,
                finalize=False,
                clear_provisional=False,
                reason="simulation time did not advance",
            )
        elapsed = float(obs.simulation_time_s) - float(first.simulation_time_s)
        if elapsed < self.min_settle_interval_s:
            return PlacementSuccessDecision(
                provisional=True,
                finalize=False,
                clear_provisional=False,
                reason=(
                    "source-fresh frame arrived before simulation settle "
                    "interval"
                ),
            )
        self._finalized = True
        self._first = None
        return PlacementSuccessDecision(
            provisional=False,
            finalize=True,
            clear_provisional=False,
            reason=(
                "two source-fresh simulation-progressed "
                "released/contained/stable frames confirmed"
            ),
        )
