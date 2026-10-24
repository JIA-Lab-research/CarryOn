"""Event-triggered judging from gripper-state transitions.

The monitor tracks drop, release and rehold events from timestamped state
samples. Optional empty_grasp detection counts closes without a stable hold.
Holding requires width stability for stable_s within stable_band_m.

A checkpoint can pause the VLA while the judge evaluates the boundary frame.
Paused intervals are recorded and excluded from grace timers and job deadlines;
the supervisor then resumes execution, reports success or invokes recovery."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


def _log_default(msg: str) -> None:
    print(f"[checkpoint] {msg}", flush=True)


@dataclass
class GripperEvent:
    kind: str            # "drop" | "release" | "rehold"
    ts: float            # sample timestamp of the transition
    width: float         # gripper width at the transition sample
    index: int           # 0-based event ordinal within this armed job
    job_key: str = ""
    pause: str = "none"  # none | pending | ok | failed (pause_fn outcome)
    consumed: bool = False
    checkpoint_event_wall_ts: float = 0.0
    pause_request_wall_ts: Optional[float] = None
    pause_ack_wall_ts: Optional[float] = None
    grasp_contact: Optional[bool] = None
    release_commanded: Optional[bool] = None
    terminal_route: str = "none"

    def to_meta(self) -> dict:
        return {
            "kind": self.kind, "ts": self.ts, "width": round(self.width, 4),
            "index": self.index, "pause": self.pause,
            "consumed": self.consumed,
            "checkpoint_event_wall_ts": self.checkpoint_event_wall_ts,
            "pause_request_wall_ts": self.pause_request_wall_ts,
            "pause_ack_wall_ts": self.pause_ack_wall_ts,
            "grasp_contact": self.grasp_contact,
            "release_commanded": self.release_commanded,
            "terminal_route": self.terminal_route,
        }


class CheckpointPauseControl:
    """Pause/resume one VLA job over the gello control plane.

    ``pause`` mirrors the recovery takeover (``cancel(standby=True)``: gello
    brakes, holds the arm, yields the bridge). ``resume`` mirrors the recovery
    handback (``execute(skip_reset=True)`` under the same request_id, so the
    episode continues from the current pose, in-distribution). ``job_lookup``
    resolves the JobRecord for a job_key so the resume carries the original
    ``callback_url`` (without it the env's completion callback goes nowhere).
    """

    def __init__(
        self,
        gello: Any,
        *,
        job_lookup: Optional[Callable[[str], Any]] = None,
        log: Callable[[str], None] = _log_default,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.gello = gello
        self.job_lookup = job_lookup
        self._log = log
        self._clock = clock
        self._resume_timings: dict[str, dict[str, Any]] = {}

    def _callback_url(self, job_key: str) -> str:
        if self.job_lookup is None:
            return ""
        try:
            rec = self.job_lookup(job_key)
        except Exception:  # noqa: BLE001
            return ""
        return str(getattr(rec, "callback_url", "") or "")

    def _record(self, job_key: str):
        if self.job_lookup is None:
            return None
        try:
            return self.job_lookup(job_key)
        except Exception:  # noqa: BLE001
            return None

    def pause(self, job_key: str) -> bool:
        rec = self._record(job_key)
        deadline_frozen = bool(
            rec is not None and rec.pause_deadline(self._clock())
        )
        try:
            res = self.gello.cancel(job_key, return_home=False, standby=True)
        except Exception as e:  # noqa: BLE001
            if deadline_frozen:
                rec.resume_deadline(self._clock())
            self._log(f"pause({job_key}) raised: {type(e).__name__}: {e}")
            return False
        ok = bool(getattr(res, "standby", False))
        if not ok and deadline_frozen:
            rec.resume_deadline(self._clock())
        deadline_state = "frozen" if ok and rec is not None else "untracked"
        self._log(
            f"pause({job_key}) -> standby={ok} deadline={deadline_state}"
        )
        return ok

    def resume_timing(self, job_key: str) -> dict[str, Any]:
        return dict(self._resume_timings.get(str(job_key), {}))

    def resume(self, job_key: str, instruction: str) -> bool:
        request_wall_ts = self._clock()
        try:
            res = self.gello.execute(
                instruction,
                request_id=job_key,
                skip_reset=True,
                callback_url=self._callback_url(job_key),
            )
        except Exception as e:  # noqa: BLE001
            response_wall_ts = self._clock()
            self._resume_timings[str(job_key)] = {
                "resume_request_wall_ts": request_wall_ts,
                "resume_response_wall_ts": response_wall_ts,
                "resume_generation": None,
                "first_resumed_action_wall_ts": None,
                "first_resumed_cycle_wall_ts": None,
            }
            self._log(f"resume({job_key}) raised: {type(e).__name__}: {e}")
            return False
        response_wall_ts = self._clock()
        ok = bool(getattr(res, "accepted", False))
        raw = getattr(res, "raw", None)
        raw = raw if isinstance(raw, dict) else {}
        self._resume_timings[str(job_key)] = {
            "resume_request_wall_ts": request_wall_ts,
            "resume_response_wall_ts": response_wall_ts,
            "resume_generation": raw.get("resume_generation"),
            "first_resumed_action_wall_ts": raw.get("first_action_returned_ts"),
            "first_resumed_cycle_wall_ts": raw.get("first_cycle_completed_ts"),
        }
        rec = self._record(job_key)
        paused_s = rec.resume_deadline(self._clock()) if ok and rec is not None else None
        deadline_state = (
            f"+{paused_s:.1f}s" if paused_s is not None else "unchanged"
        )
        self._log(
            f"resume({job_key}) -> accepted={ok} deadline={deadline_state}"
        )
        return ok


class GripperTransitionMonitor:
    """Sample-level gripper FSM + event queue + instant-pause trigger.

    Fed from the shared 50Hz state stream (``on_traj_sample`` hooks
    ``StateStreamRecorder``); armed/disarmed by the judge supervisor around
    each supervised job. Single-active-job by design (the LIBERO eval runs
    one arm job at a time); ``arm`` resets the FSM and replaces the job.

    Thread-safety: ``on_sample`` runs on the recorder thread; ``arm`` /
    ``disarm`` / ``pending_event`` / ``consume`` run on the event loop
    thread. All shared state is guarded by one lock; ``pause_fn`` (an HTTP
    call) runs on a spawned daemon thread so the 50Hz feed never blocks.
    """

    def __init__(
        self,
        *,
        hold_min_m: float = 0.008,
        hold_max_m: float = 0.075,
        empty_max_m: float = 0.005,
        stable_s: float = 0.06,
        stable_band_m: float = 0.003,
        pause_fn: Optional[Callable[[str], bool]] = None,
        hold_loss_fn: Optional[Callable[[GripperEvent], None]] = None,
        arm_fn: Optional[Callable[[str], None]] = None,
        max_pauses: int = 6,
        emptygrasp_count: int = 0,
        log: Callable[[str], None] = _log_default,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.hold_min_m = float(hold_min_m)
        self.hold_max_m = float(hold_max_m)
        self.empty_max_m = float(empty_max_m)
        self.stable_s = float(stable_s)
        self.stable_band_m = float(stable_band_m)
        self.pause_fn = pause_fn
        self.hold_loss_fn = hold_loss_fn
        self.arm_fn = arm_fn
        self.max_pauses = max(0, int(max_pauses))
        # Channel 3: fire an event after this many consecutive closes onto
        # air with no stable hold in between. 0 = disabled.
        self.emptygrasp_count = max(0, int(emptygrasp_count))
        self._log = log
        self._clock = clock
        self._lock = threading.Lock()
        self._armed = False
        self._job_key = ""
        self._reset_fsm_locked()

    # -- lifecycle ---------------------------------------------------------
    def _reset_fsm_locked(self) -> None:
        self._state = "unknown"        # unknown | empty | open | holding
        # Rolling stability window over in-band samples: restarts whenever the
        # width leaves the band or drifts more than stable_band_m.
        self._win_start_ts: Optional[float] = None
        self._win_min = 0.0
        self._win_max = 0.0
        self._dropped = False          # a drop happened; next stable hold = rehold
        self._rehold_pending = False   # rehold seen, identity check not yet done
        self._events: list[GripperEvent] = []
        self._release_count = 0
        self._empty_close_run = 0      # consecutive closes-on-air since a hold
        self._pauses_done = 0
        self._paused = False
        self._hold_loss_published = False
        # Post-handback grace: suppress the failure fast-lanes for the
        # first N samples after a re-arm, so the recovery->VLA control
        # transfer's own gripper transitions do not fire an instant second
        # takeover that clobbers a just-recovered episode (t3_ep0). N is a
        # forward-progress proxy (samples = the VLA stepping); a genuine
        # LATER failure (drop mid-carry, after grace) is still caught.
        self._grace_remaining = 0
        # Wall-clock [start, end] spans the world spent frozen under our pause
        # (end None = still frozen). Consumed by paused_overlap().
        self._pause_intervals: list[list[Optional[float]]] = []

    def arm(self, job_key: str, *, grace_samples: int = 0) -> None:
        with self._lock:
            self._armed = True
            self._job_key = str(job_key)
            self._reset_fsm_locked()
            self._grace_remaining = max(0, int(grace_samples))
        if self.arm_fn is not None:
            self.arm_fn(str(job_key))
        self._log(
            f"armed for job={job_key}"
            + (f" (post-handback grace={grace_samples} samples)"
               if grace_samples else "")
        )

    def disarm(self) -> None:
        with self._lock:
            was = self._armed
            self._armed = False
        if was:
            self._log(f"disarmed (job={self._job_key})")

    def is_armed(self) -> bool:
        with self._lock:
            return self._armed

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def note_resumed(self) -> None:
        """The supervisor resumed the VLA after a pause."""
        with self._lock:
            self._paused = False
            self._close_pause_interval_locked()

    def note_resume_failed(self) -> None:
        """A resume attempt failed: the world is (probably) still frozen, but
        letting paused_overlap grow unbounded would freeze the stuck/min-fail
        clocks too -- the watch could never escalate a wedged episode. Close
        the frozen interval (clocks run again -> the stuck lane eventually
        fires and hands the still-standby world to recovery) and keep
        ``_paused`` so the supervisor retries the resume on later polls
        (close-or-alarm)."""
        with self._lock:
            self._close_pause_interval_locked()
        self._log(
            "ALARM: resume failed with the world paused; un-freezing the "
            "judge clocks so escalation can proceed (will retry resume)"
        )

    def _close_pause_interval_locked(self) -> None:
        if self._pause_intervals and self._pause_intervals[-1][1] is None:
            self._pause_intervals[-1][1] = self._clock()

    def paused_overlap(self, t0: float, t1: float) -> float:
        """Seconds of wall-clock span [t0, t1] the world spent frozen under a
        checkpoint pause. The supervisor subtracts this from every time-based
        trigger (stuck timeout, min-fail floor, proprio grace) so a pause is
        budget-free for ALL clocks, not just the sim horizon."""
        if t1 <= t0:
            return 0.0
        now = self._clock()
        total = 0.0
        with self._lock:
            for start, end in self._pause_intervals:
                s = max(t0, float(start))
                e = min(t1, now if end is None else float(end))
                if e > s:
                    total += e - s
        return total

    # -- introspection ------------------------------------------------------
    def pending_event(self) -> Optional[GripperEvent]:
        with self._lock:
            for ev in self._events:
                if not ev.consumed:
                    return ev
        return None

    def consume(self, event: GripperEvent) -> None:
        with self._lock:
            event.consumed = True

    def release_count(self) -> int:
        with self._lock:
            return self._release_count

    def rehold_pending(self) -> bool:
        with self._lock:
            return self._rehold_pending

    def clear_rehold(self) -> None:
        with self._lock:
            self._rehold_pending = False

    def event_summary(self) -> list[dict]:
        with self._lock:
            return [ev.to_meta() for ev in self._events]

    def local_terminal_published(self, job_key: str) -> bool:
        """True once this job's hold-loss was routed to terminal intake."""
        with self._lock:
            return bool(
                self._job_key == str(job_key) and self._hold_loss_published
            )

    # -- sample feed ---------------------------------------------------------
    def on_traj_sample(self, sample: Any) -> None:
        """Hook for StateStreamRecorder.on_sample (TrajSample-shaped input)."""
        try:
            ts = float(getattr(sample, "ts", 0.0) or 0.0)
            width = float(getattr(sample, "gripper_width", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        contact = getattr(sample, "grasp_contact", None)
        if not isinstance(contact, bool):
            contact = None
        release_commanded = getattr(sample, "release_commanded", None)
        if not isinstance(release_commanded, bool):
            w_cmd = getattr(sample, "w_cmd", None)
            try:
                release_commanded = (
                    float(w_cmd) > self.hold_max_m if w_cmd is not None else False
                )
            except (TypeError, ValueError):
                release_commanded = False
        self.on_sample(
            ts, width, grasp_contact=contact,
            release_commanded=release_commanded,
        )

    def _band(self, w: float) -> Optional[str]:
        if w < self.empty_max_m:
            return "empty"
        if self.hold_min_m <= w <= self.hold_max_m:
            return "hold"
        if w > self.hold_max_m:
            return "open"
        return None  # dead zone between empty_max and hold_min

    def on_sample(
        self,
        ts: float,
        width: float,
        *,
        grasp_contact: Optional[bool] = None,
        release_commanded: Optional[bool] = None,
    ) -> None:
        fire: Optional[GripperEvent] = None
        with self._lock:
            if not self._armed:
                return
            if self._grace_remaining > 0:
                self._grace_remaining -= 1
                grace = True
            else:
                grace = False
            band = self._band(width)
            command_evidence_available = isinstance(release_commanded, bool)
            release_is_commanded = release_commanded is True




            if release_is_commanded:
                self._win_start_ts = None
                if self._state == "holding":
                    self._release_count += 1
                    fire = self._record_locked(
                        "commanded_release", ts, width,
                        grasp_contact=grasp_contact,
                        release_commanded=True,
                    )
                    fire.consumed = True
                self._state = band or "release_commanded"
            # Prefer tactile contact when the producer provides it. Width is
            # the fallback for bridges without a contact signal.
            holding_signal = (
                grasp_contact is True
                if grasp_contact is not None
                else band == "hold"
            )
            if not release_is_commanded and holding_signal:
                # Rolling stability window: restart whenever the width drifts
                # more than stable_band_m from the window extremes (a closing/
                # opening sweep restarts every sample; duplicated samples share
                # a ts and add no duration).
                if self._win_start_ts is None:
                    self._win_start_ts, self._win_min, self._win_max = ts, width, width
                else:
                    self._win_min = min(self._win_min, width)
                    self._win_max = max(self._win_max, width)
                    if (self._win_max - self._win_min) > self.stable_band_m:
                        self._win_start_ts, self._win_min, self._win_max = (
                            ts, width, width,
                        )
                if (
                    self._state != "holding"
                    and (ts - self._win_start_ts) >= self.stable_s
                ):
                    self._state = "holding"
                    # Once a real stable hold exists, a later uncommanded loss
                    # must not be hidden by post-handback grace.
                    self._grace_remaining = 0
                    self._empty_close_run = 0   # a real hold resets channel 3
                    if self._dropped:
                        self._dropped = False
                        self._rehold_pending = True
                        # Flag-only: the identity check rides the next judge
                        # poll's context; no pause, no pending-event queueing
                        # (a correct re-pick is the common self-heal path).
                        fire = self._record_locked(
                            "rehold", ts, width,
                            grasp_contact=grasp_contact,
                            release_commanded=False,
                        )
                        fire.consumed = True
                return
            self._win_start_ts = None
            if release_is_commanded:
                # Normal placement release continues to Endpoint-R; there is no
                # pause, no remote-Judge detour, and no recovery trigger.
                return
            if band is None and grasp_contact is None:
                # Width-only legacy lane keeps HOLDING through the dead zone;
                # classify only once it reaches empty or open.
                return
            if band is None:
                band = "not_holding"
            if self._state == "holding":
                self._state = band
                if not grace:
                    # Callers without contact or command intent use width-only events.
                    # Explicit intent enables the local hold-loss path.
                    if (
                        grasp_contact is None
                        and not command_evidence_available
                    ):
                        kind = "drop" if band == "empty" else "release"
                        if kind == "release":
                            self._release_count += 1
                        else:
                            self._dropped = True
                    else:
                        kind = "hold_loss"
                        self._dropped = True
                    fire = self._record_locked(
                        kind, ts, width,
                        grasp_contact=grasp_contact,
                        release_commanded=release_commanded,
                    )
                # during grace: state updated, no event/pause (this is
                # the recovery->VLA handback transition, not a new failure)
            else:
                # Channel 3: entering CLOSED-ON-AIR from anywhere but a hold
                # is an empty close (a full pinch onto nothing). The dead
                # zone between empty_max and hold_min never changes state, so
                # width noise across the empty boundary cannot double-count.
                if (
                    not grace
                    and band == "empty"
                    and self._state != "empty"
                    and self.emptygrasp_count > 0
                ):
                    self._empty_close_run += 1
                    if self._empty_close_run >= self.emptygrasp_count:
                        self._empty_close_run = 0  # N more misses to re-fire
                        fire = self._record_locked(
                            "empty_grasp", ts, width,
                            grasp_contact=grasp_contact,
                            release_commanded=False,
                        )
                self._state = band
        if fire is not None:
            self._log(
                f"event {fire.kind}#{fire.index} ts={fire.ts:.3f} "
                f"width={fire.width:.4f} job={fire.job_key} pause={fire.pause}"
            )
            if fire.pause == "pending":
                self._spawn_pause(fire)

    def _record_locked(
        self,
        kind: str,
        ts: float,
        width: float,
        *,
        grasp_contact: Optional[bool] = None,
        release_commanded: Optional[bool] = None,
    ) -> GripperEvent:
        ev = GripperEvent(
            kind=kind, ts=ts, width=width,
            index=len(self._events), job_key=self._job_key,
            checkpoint_event_wall_ts=self._clock(),
            grasp_contact=grasp_contact,
            release_commanded=release_commanded,
        )
        wants_pause = (
            kind in ("hold_loss", "drop", "release", "empty_grasp")
            and self.pause_fn is not None
            and not self._paused
            and self._pauses_done < self.max_pauses
        )
        if wants_pause:
            ev.pause = "pending"
            self._pauses_done += 1
            self._paused = True  # optimistic: cleared if the pause call fails



        self._events.append(ev)
        return ev

    def _spawn_pause(self, ev: GripperEvent) -> None:
        def _run() -> None:
            ok = False
            terminal_callback = None
            req_ts = self._clock()
            ev.pause_request_wall_ts = req_ts
            try:
                ok = bool(self.pause_fn(ev.job_key))
            except Exception as e:  # noqa: BLE001
                self._log(f"pause thread raised: {type(e).__name__}: {e}")
            with self._lock:
                ev.pause_ack_wall_ts = self._clock()
                ev.pause = "ok" if ok else "failed"
                if ok:
                    self._pause_intervals.append([req_ts, None])
                    if (
                        ev.kind == "hold_loss"
                        and self.hold_loss_fn is not None
                        and not self._hold_loss_published
                    ):
                        self._hold_loss_published = True
                        self._armed = False
                        ev.consumed = True
                        ev.terminal_route = "local_pending"
                        terminal_callback = self.hold_loss_fn
                else:
                    # The world never froze: no interval, timers keep running.
                    self._paused = False
            if terminal_callback is not None:
                try:
                    terminal_callback(ev)
                    ev.terminal_route = "local_published"
                except Exception as e:  # noqa: BLE001
                    ev.terminal_route = "local_publish_failed"
                    self._log(
                        "hold-loss terminal callback raised: "
                        f"{type(e).__name__}: {e}"
                    )

        threading.Thread(target=_run, name="checkpoint-pause", daemon=True).start()
