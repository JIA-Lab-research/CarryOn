"""Joint-state buffering and command transport over ZMQ."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional, Protocol, Sequence

from .schema import TrajSample


# --------------------------------------------------------------------------- #
# pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def quat_xyzw_to_wxyz(q: Sequence[float]) -> list[float]:
    x, y, z, w = q[0], q[1], q[2], q[3]
    return [w, x, y, z]


def traj_sample_from_state(state: dict, *, quat_order: str = "xyzw") -> TrajSample:
    """Convert a bridge PUB ``state`` dict into a :class:`TrajSample`.

    ``quat_order`` is the convention of ``state['ee_quat']`` as published by the
    bridge; polymetis FK is ``xyzw`` so we default to that and convert to the
    schema's ``wxyz``. ``gripper_width`` prefers the hardware-measured ``w_meas``
    and falls back to the commanded ``w_cmd``. ``ts`` falls back to wall-clock if
    absent.
    """
    ee_quat = list(state.get("ee_quat", [1.0, 0.0, 0.0, 0.0]))
    if quat_order == "xyzw" and len(ee_quat) == 4:
        ee_quat = quat_xyzw_to_wxyz(ee_quat)
    w_cmd = state.get("w_cmd")
    w_meas = state.get("w_meas")
    grasp_contact = state.get("grasp_contact")
    if not isinstance(grasp_contact, bool):
        grasp_contact = None
    release_commanded = state.get("release_commanded")
    if not isinstance(release_commanded, bool):
        release_commanded = None
    # Prefer measured -> commanded -> explicit 'gripper' -> 0.0.
    if w_meas is not None:
        gripper = w_meas
    elif w_cmd is not None:
        gripper = state.get("gripper", w_cmd)
    else:
        gripper = state.get("gripper", 0.0)
    return TrajSample(
        ts=float(state.get("ts", time.time())),
        q=[float(v) for v in state.get("q", [])],
        ee_pos=[float(v) for v in state.get("ee_pos", [0.0, 0.0, 0.0])],
        ee_quat=[float(v) for v in ee_quat],
        gripper_width=float(gripper),
        w_cmd=float(w_cmd) if w_cmd is not None else None,
        w_meas=float(w_meas) if w_meas is not None else None,
        grasp_contact=grasp_contact,
        release_commanded=release_commanded,
    )


class _StateSource(Protocol):
    def get_state(self) -> Optional[dict]: ...


# --------------------------------------------------------------------------- #
# ring buffer                                                                  #
# --------------------------------------------------------------------------- #
class TrajectoryRingBuffer:
    """Fixed-size, time-ordered buffer of visited robot states.

    The set of samples here defines the *physically-visited* configurations a
    rewind is allowed to return to (see ``select_rewind_target``). ``maxlen``
    should cover the longest plausible look-back window (default ~60 s at 15 Hz).
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._buf: deque[TrajSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, sample: TrajSample) -> None:
        with self._lock:
            self._buf.append(sample)

    def ingest_state(self, state: dict, *, quat_order: str = "xyzw") -> TrajSample:
        s = traj_sample_from_state(state, quat_order=quat_order)
        self.append(s)
        return s

    def snapshot(self) -> list[TrajSample]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __bool__(self) -> bool:
        # An empty buffer is still a valid object; avoid `buffer or default`
        # accidentally treating a fresh (len 0) buffer as falsy.
        return True

    def latest(self) -> Optional[TrajSample]:
        with self._lock:
            return self._buf[-1] if self._buf else None


# --------------------------------------------------------------------------- #
# background recorder                                                          #
# --------------------------------------------------------------------------- #
class StateStreamRecorder:
    """Background thread that pumps a state source into a ring buffer.

    Works with any object exposing ``get_state() -> dict | None`` (the real
    :class:`ZmqBridgeClient` or a fake), so it is fully testable without pyzmq.
    """

    def __init__(
        self,
        source: _StateSource,
        buffer: Optional[TrajectoryRingBuffer] = None,
        *,
        poll_hz: float = 50.0,
        quat_order: str = "xyzw",
        on_sample: Optional[Callable[[TrajSample], None]] = None,
    ) -> None:
        self.source = source
        self.buffer = buffer if buffer is not None else TrajectoryRingBuffer()
        self._period = 1.0 / max(1e-3, poll_hz)
        self._quat_order = quat_order
        self._on_sample = on_sample
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def poll_once(self) -> Optional[TrajSample]:
        state = self.source.get_state()
        if not state:
            return None
        s = self.buffer.ingest_state(state, quat_order=self._quat_order)
        if self._on_sample is not None:
            try:
                self._on_sample(s)
            except Exception:
                pass
        return s

    def set_on_sample(self, fn: Optional[Callable[[TrajSample], None]]) -> None:
        """(Re)attach the per-sample hook after construction.

        The D4 gripper-transition checkpoint monitor registers here (via
        ``EpisodeRecorderRegistry.set_gripper_sample_hook``) so it sees every
        50Hz sample the moment it is ingested -- transition detection at
        sample resolution instead of judge-poll resolution."""
        self._on_sample = fn

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def _run() -> None:
            consec = 0
            while not self._stop.is_set():
                try:
                    self.poll_once()
                    consec = 0
                except Exception as e:  # noqa: BLE001
                    consec += 1
                    if consec == 50 or consec % 1000 == 0:




                        print(
                            "[recorder] ALARM: state poll failing (%d consecutive: "
                            "%s: %s); trajectory ring is STALE" % (consec, type(e).__name__, e),
                            flush=True,
                        )
                time.sleep(self._period)

        self._thread = threading.Thread(target=_run, name="recovery-state-recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# --------------------------------------------------------------------------- #
# ZMQ bridge client (lazy pyzmq)                                               #
# --------------------------------------------------------------------------- #
class ZmqBridgeClient:
    """Mirror of gello's ``BridgeClient`` for CarryOn-side use.

    During a recovery takeover (after gello enters standby) CarryOn uses
    ``send_command`` to drive the bridge and ``get_state`` to read robot state.
    Connects to the *same* ports as gello (they share the bridge on the 4090).
    """

    def __init__(self, host: str, rep_port: int, pull_port: int, pub_port: int) -> None:
        import zmq  # type: ignore

        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.req = ctx.socket(zmq.REQ)
        # LINGER=0 so closing/terminating never blocks on undrained messages. The
        # shared Context.instance() is term()'d at interpreter exit; with the
        # default LINGER=-1 a PUSH socket whose PULL peer isn't draining makes
        # term() hang forever -> Ctrl-C can't kill the process. 0 = drop on close.
        self.req.setsockopt(zmq.LINGER, 0)
        self.req.setsockopt(zmq.RCVTIMEO, 10_000)
        self.req.setsockopt(zmq.SNDTIMEO, 10_000)
        self.req.connect(f"tcp://{host}:{rep_port}")
        self.push = ctx.socket(zmq.PUSH)
        self.push.setsockopt(zmq.LINGER, 0)
        self.push.connect(f"tcp://{host}:{pull_port}")
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        # ZMQ_CONFLATE only keeps the newest message, but it MUST be set BEFORE
        # connect() or it silently has no effect — then the SUB queue accumulates
        # frames and recv() returns the OLDEST (stale) one, so current_q() lags far
        # behind the real arm (this is what made strict-settle never converge and
        # the streamed deltas read ~q_start the whole run).
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(f"tcp://{host}:{pub_port}")

    def get_state(self) -> Optional[dict]:
        try:
            return self.sub.recv_json(flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None

    def send_command(self, arm: Sequence[float], gripper_width: float) -> None:
        self.push.send_json(
            {"type": "cmd", "arm": list(arm), "gripper": float(gripper_width)}
        )

    def handshake(self, init_q: Sequence[float]) -> dict[str, Any]:
        """Re-handshake -> robot moves to ``init_q``. NOT used for recovery
        (recovery streams ``send_command`` instead so it never forces init)."""
        self.req.send_json({"type": "init", "virtual_init_arm": list(init_q)})
        return self.req.recv_json()

    def close(self) -> None:
        """Close the three sockets (LINGER=0, so this never blocks). The shared
        Context.instance() is left for zmq's atexit term; closing the sockets
        first means that term has nothing to wait on."""
        for sock in (getattr(self, "sub", None), getattr(self, "push", None), getattr(self, "req", None)):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
