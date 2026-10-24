#!/usr/bin/env python3
"""Phase 1 blind env-step supervisor: native stop remains outside the runner."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROTOCOL = "aegis.phase1.blind_step_supervisor.v1"
FORBIDDEN_KEYS = frozenset(
    {"done", "reward", "info", "success", "outcome", "result"}
)
OBS_FIELDS = frozenset(
    {"kind", "protocol", "sequence", "phase", "observation"}
)
PERMIT_FIELDS = frozenset({"kind", "protocol", "sequence"})
ACK_FIELDS = frozenset({"kind", "protocol", "sequence"})
ACTION_FIELDS = frozenset({"kind", "protocol", "sequence", "action"})


class ProtocolViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class SupervisorConfig:
    settle_steps: int
    max_control_steps: int
    coordinator_namespace: str
    coordinator_record_name: str
    public_pointer_path: str

    def validate(self) -> None:
        if self.settle_steps < 0 or self.max_control_steps <= 0:
            raise ValueError("invalid settle/control step limit")
        namespace = Path(self.coordinator_namespace).resolve()
        if "evaluation_private" not in namespace.parts:
            raise ValueError("namespace must contain evaluation_private")
        if Path(self.coordinator_record_name).name != self.coordinator_record_name:
            raise ValueError("coordinator_record_name must be a basename")


def canonical_bytes(value: Any) -> bytes:
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return (text + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def iter_key_paths(value: Any, prefix: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            path = prefix + (str(raw_key),)
            yield path
            yield from iter_key_paths(child, path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_key_paths(
                child, prefix + (f"[{index}]",)
            )


def assert_observation_schema(observation: Any) -> str:
    """Positively assert every observation dict and return a key-schema SHA."""
    if not isinstance(observation, Mapping):
        raise ProtocolViolation("observation must be a mapping")
    rows = []
    for path in iter_key_paths(observation):
        if path[-1].casefold() in FORBIDDEN_KEYS:
            raise ProtocolViolation(
                "forbidden observation key at " + ".".join(path)
            )
        rows.append(".".join(path))
    return hashlib.sha256(
        "\n".join(sorted(rows)).encode("utf-8")
    ).hexdigest()


def require_packet(
    packet: Any, fields: frozenset[str], kind: str
) -> None:
    if (
        not isinstance(packet, Mapping)
        or frozenset(packet.keys()) != fields
    ):
        raise ProtocolViolation(
            f"{kind} packet differs from frozen schema"
        )
    if packet["kind"] != kind or packet["protocol"] != PROTOCOL:
        raise ProtocolViolation(f"invalid {kind} packet header")


def write_coordinator_record(
    config: SupervisorConfig,
    private_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write private bytes; expose only their path, byte count, and SHA."""
    namespace = Path(config.coordinator_namespace).resolve()
    private_path = (
        namespace / config.coordinator_record_name
    ).resolve()
    if (
        os.path.commonpath((str(namespace), str(private_path)))
        != str(namespace)
    ):
        raise ValueError("coordinator record escaped namespace")
    if not isinstance(private_payload, Mapping):
        raise ProtocolViolation(
            "driver coordinator payload must be a mapping"
        )
    if (
        private_payload.get("schema")
        != "carryon.safety.episode_endpoints.v1"
    ):
        raise ProtocolViolation(
            "driver coordinator payload schema drift"
        )
    payload = canonical_bytes(dict(private_payload))
    digest = hashlib.sha256(payload).hexdigest()
    write_exclusive(private_path, payload, 0o600)
    pointer = {
        "schema": "aegis.phase1.coordinator_artifact_pointer.v1",
        "coordinator_only_artifact_path": str(private_path),
        "artifact_sha256": digest,
        "artifact_bytes": len(payload),
    }
    pointer_path = Path(config.public_pointer_path)
    write_exclusive(
        pointer_path, canonical_bytes(pointer), 0o644
    )
    return pointer


def run_supervisor(
    driver_factory: Callable[[], Any],
    observation_send: Any,
    runner_receive: Any,
    config: SupervisorConfig,
) -> None:
    """Own env.step and translate native termination only into runner EOF."""
    config.validate()
    driver = driver_factory()
    sequence = 0
    settle_count = 0
    control_count = 0
    pending_close = None
    phase = "settle" if config.settle_steps else "control"
    try:
        observation_send.send(
            {
                "kind": "observation",
                "protocol": PROTOCOL,
                "sequence": sequence,
                "phase": phase,
                "observation": driver.reset(),
            }
        )
        while True:
            acknowledgement = runner_receive.recv()
            require_packet(
                acknowledgement, ACK_FIELDS, "observation_ack"
            )
            if acknowledgement["sequence"] != sequence:
                raise ProtocolViolation(
                    "observation acknowledgement sequence mismatch"
                )
            if pending_close is not None:
                private_payload = driver.coordinator_payload(
                    sequence, pending_close
                )
                runner_receive.close()
                observation_send.close()
                write_coordinator_record(config, private_payload)
                return

            observation_send.send(
                {
                    "kind": "action_permit",
                    "protocol": PROTOCOL,
                    "sequence": sequence,
                }
            )
            action_packet = runner_receive.recv()
            require_packet(action_packet, ACTION_FIELDS, "action")
            if action_packet["sequence"] != sequence:
                raise ProtocolViolation(
                    "action sequence mismatch"
                )
            step_record = driver.step(action_packet["action"])
            if (
                not isinstance(step_record, (tuple, list))
                or len(step_record) < 3
            ):
                raise ProtocolViolation(
                    "driver.step must return at least three items"
                )
            next_observation = step_record[0]
            native_method_stop = bool(step_record[2])

            if phase == "settle":
                settle_count += 1
                if settle_count >= config.settle_steps:
                    phase = "control"
            else:
                control_count += 1

            if native_method_stop:
                pending_close = "native_method_stop"
            elif control_count >= config.max_control_steps:
                pending_close = "method_horizon"

            sequence += 1
            observation_send.send(
                {
                    "kind": "observation",
                    "protocol": PROTOCOL,
                    "sequence": sequence,
                    "phase": phase,
                    "observation": next_observation,
                }
            )
    finally:
        for endpoint in (runner_receive, observation_send):
            try:
                endpoint.close()
            except OSError:
                pass
        close_driver = getattr(driver, "close", None)
        if callable(close_driver):
            close_driver()


def run_guarded_runner(
    observation_receive: Any,
    supervisor_send: Any,
    action_builder: Callable[
        [Mapping[str, Any], str, int], Any
    ],
    public_receipt_path: str,
    check_channel_closed: bool = False,
    observation_consumer: Callable[
        [Mapping[str, Any], str, int], None
    ]
    | None = None,
) -> None:
    """Runner boundary: observations and permits only; no native stop state."""
    observation_packets = 0
    schema_assertions = 0
    action_packets = 0
    schema_hashes = []
    observed_field_sets = []
    observation_eof = False
    action_close = False

    while True:
        try:
            packet = observation_receive.recv()
        except EOFError:
            observation_eof = True
            break
        require_packet(packet, OBS_FIELDS, "observation")
        observed_field_sets.append(sorted(packet.keys()))
        sequence = int(packet["sequence"])
        phase = str(packet["phase"])
        observation = packet["observation"]
        schema_hashes.append(
            assert_observation_schema(observation)
        )
        if observation_consumer is not None:
            observation_consumer(
                observation, phase, sequence
            )
        observation_packets += 1
        schema_assertions += 1
        supervisor_send.send(
            {
                "kind": "observation_ack",
                "protocol": PROTOCOL,
                "sequence": sequence,
            }
        )
        try:
            permit = observation_receive.recv()
        except EOFError:
            observation_eof = True
            break
        require_packet(
            permit, PERMIT_FIELDS, "action_permit"
        )
        observed_field_sets.append(sorted(permit.keys()))
        if permit["sequence"] != sequence:
            raise ProtocolViolation(
                "action permit sequence mismatch"
            )
        action = action_builder(
            observation, phase, sequence
        )
        supervisor_send.send(
            {
                "kind": "action",
                "protocol": PROTOCOL,
                "sequence": sequence,
                "action": action,
            }
        )
        action_packets += 1


    if check_channel_closed:
        try:
            supervisor_send.send(
                {
                    "kind": "closure_probe",
                    "protocol": PROTOCOL,
                    "sequence": -1,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            action_close = True

    reserved_locals = sorted(
        set(locals()).intersection(FORBIDDEN_KEYS)
    )
    if reserved_locals:
        raise ProtocolViolation(
            "runner local namespace contains reserved flag name"
        )
    if schema_assertions != observation_packets:
        raise ProtocolViolation(
            "not every observation received a schema assertion"
        )
    receipt = {
        "schema": "aegis.phase1.runner_attestation.v1",
        "protocol": PROTOCOL,
        "runner_pid": os.getpid(),
        "observation_packets": observation_packets,
        "schema_assertions_passed": schema_assertions,
        "action_packets": action_packets,
        "observation_schema_sha256": schema_hashes,
        "observed_packet_field_sets": observed_field_sets,
        "observation_channel_eof_observed": observation_eof,
        "action_channel_close_observed": action_close,
        "termination_signal_source": (
            "supervisor_to_runner_channel_eof"
        ),
        "exact_reserved_local_names_present": reserved_locals,
        "forbidden_observation_keys_checked": sorted(
            FORBIDDEN_KEYS
        ),
    }
    write_exclusive(
        Path(public_receipt_path),
        canonical_bytes(receipt),
        0o644,
    )
    supervisor_send.close()
    observation_receive.close()
