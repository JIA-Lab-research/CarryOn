#!/usr/bin/env python3
"""Named-constraint timeline and dual-protocol projection helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable, Sequence

ROBOT_CONTACT_NAMES = frozenset({"checkrobotcontact"})
SCHEMA_VERSION = 2
STOCK_PROJECTION = "ignore CheckRobotContact events; preserve every other named constraint instance"


def is_robot_contact_name(name: object) -> bool:
    return str(name).replace("_", "").lower() in ROBOT_CONTACT_NAMES


def constraint_event(step: int, ordinal: int, constraint: Sequence[object]) -> dict[str, object]:
    if not constraint:
        raise ValueError("empty constraint")
    name = str(constraint[0])
    args = [str(value) for value in constraint[1:]]
    return {
        "step": int(step),
        "constraint_index": int(ordinal),
        "constraint_name": name,
        "constraint_args": args,
        "constraint_key": ":".join([str(ordinal), name.lower(), *args]),
        "robot_contact_class": is_robot_contact_name(name),
    }


def evaluate_step(
    step: int,
    constraints: Sequence[Sequence[object]],
    evaluator: Callable[[Sequence[object]], object],
) -> list[dict[str, object]]:
    return [
        constraint_event(step, ordinal, constraint)
        for ordinal, constraint in enumerate(constraints)
        if bool(evaluator(constraint))
    ]


def digest_events(events: Iterable[dict[str, object]]) -> str:
    payload = json.dumps(list(events), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def first_violation_steps(events: Iterable[dict[str, object]]) -> tuple[int | None, int | None]:
    rows = list(events)
    fixed = min((int(row["step"]) for row in rows), default=None)
    stock = min(
        (
            int(row["step"])
            for row in rows
            if not bool(row.get("robot_contact_class"))
        ),
        default=None,
    )
    return fixed, stock


def protocol_success(success_step: int | None, violation_step: int | None) -> int:
    return int(
        success_step is not None
        and (violation_step is None or int(success_step) < int(violation_step))
    )


def finalize(
    events: list[dict[str, object]],
    success_step: int | None,
    horizon_steps: int,
) -> dict[str, object]:
    fixed_step, stock_step = first_violation_steps(events)
    robot_events = sum(bool(row.get("robot_contact_class")) for row in events)
    robot_episodes = int(robot_events > 0)
    return {
        "success": int(success_step is not None),
        "success_step": success_step,
        "fixed_violation": int(fixed_step is not None),
        "fixed_violation_step": fixed_step,
        "stock_violation": int(stock_step is not None),
        "stock_violation_step": stock_step,
        "paper_success_fixed": protocol_success(success_step, fixed_step),
        "paper_success_stock": protocol_success(success_step, stock_step),
        "constraint_events": events,
        "constraint_event_count": len(events),
        "robot_contact_event_count": robot_events,
        "robot_contact_episode": robot_episodes,
        "constraint_digest_sha256": digest_events(events),
        "horizon_steps": int(horizon_steps),
        "full_horizon": True,
        "constraint_timeline_schema": SCHEMA_VERSION,
        "stock_projection": STOCK_PROJECTION,
    }


def validate(record: dict[str, object], expected_horizon_steps: int | None = None) -> None:
    events = record.get("constraint_events")
    if not isinstance(events, list):
        raise AssertionError("constraint_events missing")
    if record.get("constraint_digest_sha256") != digest_events(events):
        raise AssertionError("constraint digest mismatch")
    fixed_step, stock_step = first_violation_steps(events)
    if record.get("fixed_violation_step") != fixed_step:
        raise AssertionError("fixed first-step mismatch")
    if record.get("stock_violation_step") != stock_step:
        raise AssertionError("stock first-step mismatch")
    success_step = record.get("success_step")
    if record.get("paper_success_fixed") != protocol_success(success_step, fixed_step):
        raise AssertionError("fixed paper-SR mismatch")
    if record.get("paper_success_stock") != protocol_success(success_step, stock_step):
        raise AssertionError("stock paper-SR mismatch")
    if record.get("full_horizon") is not True:
        raise AssertionError("record is not full horizon")
    if expected_horizon_steps is not None and record.get("horizon_steps") != expected_horizon_steps:
        raise AssertionError("horizon length mismatch")
