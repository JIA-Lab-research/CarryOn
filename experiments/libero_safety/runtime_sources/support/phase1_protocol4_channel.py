#!/usr/bin/env python3
"""Python 3.8/3.10 compatible protocol-4 channel for the blind step boundary."""

from __future__ import annotations

import pickle
import time
from collections.abc import Mapping

PICKLE_PROTOCOL = 4
FORBIDDEN_KEYS = frozenset(
    {"done", "reward", "info", "success", "outcome", "result"}
)
STEP_MESSAGE_SCHEMAS = {
    "observation": frozenset(
        {"kind", "protocol", "sequence", "phase", "observation"}
    ),
    "observation_ack": frozenset({"kind", "protocol", "sequence"}),
    "action_permit": frozenset({"kind", "protocol", "sequence"}),
    "action": frozenset({"kind", "protocol", "sequence", "action"}),
}


class BoundaryViolation(RuntimeError):
    pass


def iter_key_paths(value, prefix=()):
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            path = prefix + (str(raw_key),)
            yield path
            yield from iter_key_paths(child, path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_key_paths(child, prefix + ("[%d]" % index,))


def assert_no_conclusion_keys(value):
    for path in iter_key_paths(value):
        if path[-1].casefold() in FORBIDDEN_KEYS:
            raise BoundaryViolation(
                "forbidden cross-interpreter key at " + ".".join(path)
            )


def assert_step_message_schema(value):
    if not isinstance(value, Mapping):
        raise BoundaryViolation("cross-interpreter message must be a mapping")
    kind = value.get("kind")
    if kind in STEP_MESSAGE_SCHEMAS:
        if frozenset(value.keys()) != STEP_MESSAGE_SCHEMAS[kind]:
            raise BoundaryViolation("step message fields drift: " + str(kind))
    assert_no_conclusion_keys(value)


class Protocol4Connection:
    """Connection.send/recv facade pinned to pickle protocol 4 with metrics."""

    def __init__(self, connection, label):
        self._connection = connection
        self.label = str(label)
        self._closed = False
        self._send_records = []
        self._recv_records = []

    def send(self, value):
        assert_step_message_schema(value)
        encode_started = time.perf_counter_ns()
        payload = pickle.dumps(value, protocol=PICKLE_PROTOCOL)
        if not payload.startswith(b"\x80\x04"):
            raise BoundaryViolation("outbound pickle protocol is not 4")
        encode_finished = time.perf_counter_ns()
        self._connection.send_bytes(payload)
        send_finished = time.perf_counter_ns()
        self._send_records.append(
            {
                "kind": str(value.get("kind", "")),
                "payload_bytes": len(payload),
                "pickle_protocol_byte": payload[1],
                "encode_ns": encode_finished - encode_started,
                "send_ns": send_finished - encode_finished,
                "total_ns": send_finished - encode_started,
                "finished_ns": send_finished,
            }
        )

    def recv(self):
        receive_started = time.perf_counter_ns()
        payload = self._connection.recv_bytes()
        if not payload.startswith(b"\x80\x04"):
            raise BoundaryViolation("inbound pickle protocol is not 4")
        receive_finished = time.perf_counter_ns()
        value = pickle.loads(payload)
        decode_finished = time.perf_counter_ns()
        assert_step_message_schema(value)
        self._recv_records.append(
            {
                "kind": str(value.get("kind", "")),
                "payload_bytes": len(payload),
                "pickle_protocol_byte": payload[1],
                "receive_ns": receive_finished - receive_started,
                "decode_ns": decode_finished - receive_finished,
                "total_ns": decode_finished - receive_started,
                "finished_ns": decode_finished,
            }
        )
        return value

    def close(self):
        if not self._closed:
            self._closed = True
            self._connection.close()

    def metrics(self):
        return {
            "schema": "aegis.phase1.protocol4_connection_metrics.v1",
            "label": self.label,
            "pickle_protocol": PICKLE_PROTOCOL,
            "forbidden_keys": sorted(FORBIDDEN_KEYS),
            "step_message_schemas": {
                key: sorted(value)
                for key, value in sorted(STEP_MESSAGE_SCHEMAS.items())
            },
            "send_records": list(self._send_records),
            "receive_records": list(self._recv_records),
            "send_payload_bytes": sum(
                record["payload_bytes"] for record in self._send_records
            ),
            "receive_payload_bytes": sum(
                record["payload_bytes"] for record in self._recv_records
            ),
        }


def protocol4_connect(address, authkey, label):
    from multiprocessing.connection import Client

    connection = Client(address, family="AF_UNIX", authkey=authkey)
    return Protocol4Connection(connection, label)


def protocol4_accept(listener, label):
    return Protocol4Connection(listener.accept(), label)
