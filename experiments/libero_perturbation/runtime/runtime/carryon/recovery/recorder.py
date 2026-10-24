"""Build per-episode records from a continuous state stream.

A shared StateStreamRecorder owns the ring buffer. Each EpisodeRecorder keeps
its time window, judge timeline and first failure timestamp; build() slices
the shared buffer and attaches that episode's recovery trace."""
from __future__ import annotations
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
from .bridge import StateStreamRecorder, TrajectoryRingBuffer
from .schema import CameraSpec, EpisodeRecord, FrameRef, JudgeTick, JudgeVerdict, RecoveryInvocation
DEFAULT_FRAME_LEAD_S = 1.0
PIN_PREFAIL_LEAD_S = 3.0
PIN_PREFAIL_TRAIL_S = 0.5
_STATUS_TO_VERDICT: dict[str, JudgeVerdict] = {'success': 'ok', 'failed': 'fail'}

class FrameRingBuffer:
    """Time-ordered buffer of camera captures.

    Continuous RGB-D capture supplies frames near the failure and rewind timestamps.
    maxlen bounds the number of retained FrameRef entries."""

    def __init__(self, maxlen: int=600, *, on_evict: Optional[Callable[[FrameRef], None]]=None) -> None:
        self._buf: deque[FrameRef] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._on_evict = on_evict
        self._protected: set[str] = set()

    def set_protected(self, rgb_paths) -> None:
        """Replace the protected set (paths whose files survive eviction)."""
        with self._lock:
            self._protected = {p for p in rgb_paths if p}

    def append(self, frame: FrameRef) -> None:
        evicted: Optional[FrameRef] = None
        with self._lock:
            if self._buf.maxlen is not None and len(self._buf) == self._buf.maxlen and self._buf:
                candidate = self._buf[0]
                if candidate.rgb_path not in self._protected:
                    evicted = candidate
            self._buf.append(frame)
        if evicted is not None and self._on_evict is not None:
            try:
                self._on_evict(evicted)
            except Exception:
                pass

    def snapshot(self) -> list[FrameRef]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __bool__(self) -> bool:
        return True

class _FrameSource(Protocol):

    def capture(self) -> list[tuple[str, bytes, Optional[bytes]]]:
        """Return ``[(camera, rgb_bytes, depth_bytes_or_None), ...]`` for *now*."""
        ...

class GelloFrameSource:
    """Adapter that snapshots the calibrated recovery cameras over gello HTTP.

    Kept separate from disk I/O so :class:`FrameStreamRecorder` stays testable
    with a fake source. ``use_raw_rgb`` prefers ``latest_raw.jpg`` (the RGB that
    is pixel-aligned with the depth PNG) so a recorded (rgb, depth) pair is
    consistent for perception replay. A per-camera fetch failure is swallowed so
    one dead camera never stalls the whole recorder.
    """

    def __init__(self, gello: Any, cameras: list[str], *, use_raw_rgb: bool=True) -> None:
        self._gello = gello
        self._cameras = list(cameras)
        self._use_raw = use_raw_rgb

    def capture(self) -> list[tuple[str, bytes, Optional[bytes]]]:
        out: list[tuple[str, bytes, Optional[bytes]]] = []
        for cam in self._cameras:
            try:
                rgb = self._gello.fetch_raw_rgb_jpeg(cam) if self._use_raw else self._gello.fetch_rgb_jpeg(cam)
            except Exception:
                continue
            depth: Optional[bytes]
            try:
                depth = self._gello.fetch_depth_png(cam)
            except Exception:
                depth = None
            out.append((cam, rgb, depth))
        return out

class FrameStreamRecorder:
    """Background thread that snapshots cameras to disk into a :class:`FrameRingBuffer`.

    Writes ``<out_dir>/<camera>/<ts_ms>.jpg`` (+ ``.png`` for depth) and appends a
    :class:`FrameRef` carrying the paths + a single capture timestamp per poll.
    The default ``poll_hz`` (~6 Hz) keeps any instant within ~100 ms of a frame
    (the gate tolerance) without flooding disk; evicted frames are unlinked.
    """

    def __init__(self, source: _FrameSource, buffer: FrameRingBuffer, *, out_dir: str='logs/recovery_episodes/frames', poll_hz: float=6.0, save_depth: bool=True, clock=time.time) -> None:
        self.source = source
        self.buffer = buffer
        self._out_dir = Path(out_dir)
        self._period = 1.0 / max(0.001, poll_hz)
        self._save_depth = save_depth
        self._clock = clock
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def poll_once(self) -> list[FrameRef]:
        captures = self.source.capture()
        if not captures:
            return []
        ts = self._clock()
        stamp = int(ts * 1000)
        refs: list[FrameRef] = []
        for camera, rgb, depth in captures:
            cam_dir = self._out_dir / camera
            cam_dir.mkdir(parents=True, exist_ok=True)
            rgb_path = cam_dir / f'{stamp}.jpg'
            rgb_path.write_bytes(rgb)
            depth_path: Optional[str] = None
            if self._save_depth and depth is not None:
                dp = cam_dir / f'{stamp}.png'
                dp.write_bytes(depth)
                depth_path = str(dp)
            ref = FrameRef(ts=ts, camera=camera, rgb_path=str(rgb_path), depth_path=depth_path)
            self.buffer.append(ref)
            refs.append(ref)
        return refs

    @staticmethod
    def _unlink_frame(frame: FrameRef) -> None:
        for p in (frame.rgb_path, frame.depth_path):
            if not p:
                continue
            try:
                os.unlink(p)
            except OSError:
                pass

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def _run() -> None:
            while not self._stop.is_set():
                try:
                    self.poll_once()
                except Exception:
                    pass
                time.sleep(self._period)
        self._thread = threading.Thread(target=_run, name='recovery-frame-recorder', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

def judge_status_to_verdict(status: str) -> JudgeVerdict:
    """Map a vlm_judge ``status`` to a :data:`JudgeVerdict`. Unknown / transient
    statuses (``running``, ``cancelled``) are ``unsure``."""
    return _STATUS_TO_VERDICT.get(status, 'unsure')

def data_tier_quality(record: EpisodeRecord) -> dict:
    """Compute the record's **data-tier** freezability for the on-disk hint.

    Call this when an episode is *persisted* (the record is complete), NOT on a
    mid-episode ``build()`` snapshot -- a half-recorded window has no rewind
    target / frame coverage yet and would stamp a meaningless verdict.

    Only the data-tier checks run here (``geometry=None``): geometry-tier checks
    (IK feasibility, collision-free rewind, perception determinism) need an
    offline backend the live stack does not carry, so they stay ``pending`` and
    do not block. This is therefore a *hint* -- it reliably catches recording
    defects (missing calibration, lost frames, bad rewind target) but does NOT
    guarantee true freezability; the reflection gate recomputes (optionally with
    a geometry backend) and stays the authority. Never raises: a checks error
    degrades to ``freezable=None``.
    """
    try:
        from .assertions import CheckTier, run_checks
        report = run_checks(record, geometry=None, only_tier=CheckTier.DATA)
        return {'freezable': report.freezable, 'tier': 'data', 'blocking_failures': [r.id for r in report.blocking_failures], 'summary': report.summary()}
    except Exception as e:
        return {'freezable': None, 'tier': 'data', 'error': f'{type(e).__name__}: {e}'}

class EpisodeRecorder:
    """One episode = a time window over the shared trajectory buffer + a judge
    timeline. ``first_fail_ts`` tracks the start of the *current* contiguous fail
    run (reset on ``ok``/``unsure``), mirroring the judge's streak reset so the
    rewind anchor is the failure that actually led to the trigger."""

    def __init__(self, episode_id: str, goal: str, buffer: TrajectoryRingBuffer, *, frame_buffer: Optional[FrameRingBuffer]=None, clock=time.time, robot: str='fr3', gello_version: Optional[str]=None, recovery_lib_version: Optional[str]=None) -> None:
        self.episode_id = episode_id
        self.goal = goal
        self._buffer = buffer
        self._frame_buffer = frame_buffer
        self._clock = clock
        self.ts_start = clock()
        self.ts_end: Optional[float] = None
        self.judge_timeline: list[JudgeTick] = []
        self.first_fail_ts: Optional[float] = None
        self._pinned_first_fail_ts: Optional[float] = None
        self._pinned_traj: list = []
        self._pinned_frames: list[FrameRef] = []
        self._reference_frame: Optional[FrameRef] = None
        self.episode_success: Optional[bool] = None
        self.cameras: list[CameraSpec] = []
        self.frames: list[FrameRef] = []
        self.recovery: Optional[RecoveryInvocation] = None
        self.robot = robot
        self.gello_version = gello_version
        self.recovery_lib_version = recovery_lib_version
        self._last_verdict: Optional[JudgeVerdict] = None

    def add_judge_tick(self, status_or_verdict: str, *, score: Optional[float]=None, note: Optional[str]=None, ts: Optional[float]=None) -> JudgeTick:
        verdict: JudgeVerdict = status_or_verdict if status_or_verdict in ('ok', 'fail', 'unsure') else judge_status_to_verdict(status_or_verdict)
        ts = ts if ts is not None else self._clock()
        tick = JudgeTick(ts=ts, verdict=verdict, score=score, note=note)
        self.judge_timeline.append(tick)
        if verdict == 'fail':
            if self._last_verdict != 'fail':
                self.first_fail_ts = ts
                self._pin_prefail_window(ts)
        else:
            self.first_fail_ts = None
        self._last_verdict = verdict
        return tick

    def _pin_prefail_window(self, anchor_ts: float) -> None:
        """Snapshot trajectory + frames around ``anchor_ts`` (the just-detected
        failure) out of the live ring buffers and keep them, so the final
        post-recovery :meth:`build` still has the rewind-target / first-fail
        keyframes even if a slow recovery scrolled them out of the rings.

        Replaces any prior pin: only the most recent fail run matters (an earlier
        run that reset on an ok/unsure tick never triggered recovery)."""
        lo = anchor_ts - PIN_PREFAIL_LEAD_S
        hi = anchor_ts + PIN_PREFAIL_TRAIL_S
        self._pinned_traj = [s for s in self._buffer.snapshot() if lo <= s.ts <= hi]
        if self._frame_buffer is not None:
            snap = self._frame_buffer.snapshot()
            self._pinned_frames = [f for f in snap if lo <= f.ts <= hi]
            if self._reference_frame is None:
                ep_frames = [f for f in snap if f.ts >= self.ts_start]
                if ep_frames:
                    self._reference_frame = min(ep_frames, key=lambda f: f.ts)
            protected = {f.rgb_path for f in self._pinned_frames}
            if self._reference_frame is not None:
                protected.add(self._reference_frame.rgb_path)
            self._frame_buffer.set_protected(protected)
        else:
            self._pinned_frames = []

    def add_camera(self, cam: CameraSpec) -> None:
        self.cameras.append(cam)

    def add_frame(self, frame: FrameRef) -> None:
        self.frames.append(frame)

    def set_recovery(self, inv: RecoveryInvocation) -> None:
        """Attach the recovery trace to subsequent episode records.

        Retain the invocation's trigger timestamp even if later judge observations
        clear the transient failure state."""
        self.recovery = inv
        if inv is not None and inv.triggered_ts is not None:
            self._pinned_first_fail_ts = float(inv.triggered_ts)
        elif self._pinned_first_fail_ts is None:
            self._pinned_first_fail_ts = self.first_fail_ts

    def stop(self, *, success: Optional[bool]=None, ts_end: Optional[float]=None) -> None:
        self.ts_end = ts_end if ts_end is not None else self._clock()
        if success is not None:
            self.episode_success = success

    def build(self, *, frame_lead_s: float=DEFAULT_FRAME_LEAD_S) -> EpisodeRecord:
        """Snapshot the buffer window into an :class:`EpisodeRecord`. Safe to call
        live (mid-episode): uses ``now`` as ``ts_end`` if not stopped.

        The full joint ``trajectory`` is kept (the rewind anchor must snap to a
        state the VLA actually visited *before* the failure). Camera ``frames``,
        however, are scoped to the **recovery window** ``[first_fail_ts -
        frame_lead_s, end]`` so the video the LLM reflects over is the recovery
        segment (triggering failure -> handback), not the whole VLA rollout --
        the LLM only edits recovery, so the rest is noise. With no failure yet,
        frames pass through unsliced."""
        end = self.ts_end if self.ts_end is not None else self._clock()
        live_traj = self._buffer.snapshot()
        _seen_ts = {s.ts for s in live_traj}
        merged_traj = live_traj + [s for s in self._pinned_traj if s.ts not in _seen_ts]
        traj = sorted((s for s in merged_traj if self.ts_start <= s.ts <= end), key=lambda s: s.ts)
        first_fail_ts = self.first_fail_ts
        if self.recovery is not None and self._pinned_first_fail_ts is not None:
            if self.first_fail_ts is None or self.first_fail_ts <= self._pinned_first_fail_ts:
                first_fail_ts = self._pinned_first_fail_ts
        all_frames = list(self.frames)
        if self._frame_buffer is not None:
            all_frames.extend(self._frame_buffer.snapshot())
        _seen_rgb = {f.rgb_path for f in all_frames}
        all_frames.extend((f for f in self._pinned_frames if f.rgb_path not in _seen_rgb))
        all_frames.sort(key=lambda f: f.ts)
        if first_fail_ts is not None:
            f0 = first_fail_ts - frame_lead_s
            if self.recovery is not None and self.recovery.rewind_target_ts is not None:
                f0 = min(f0, float(self.recovery.rewind_target_ts) - 0.15)
            frames = [f for f in all_frames if f0 <= f.ts <= end]
        else:
            frames = [f for f in all_frames if self.ts_start <= f.ts <= end]
        if self._reference_frame is not None and all((f.rgb_path != self._reference_frame.rgb_path for f in frames)):
            frames = [self._reference_frame] + frames
        record = EpisodeRecord(episode_id=self.episode_id, goal=self.goal, ts_start=self.ts_start, ts_end=end, cameras=list(self.cameras), frames=frames, trajectory=traj, judge_timeline=list(self.judge_timeline), first_fail_ts=first_fail_ts, episode_success=self.episode_success, recovery=self.recovery, robot=self.robot, gello_version=self.gello_version, recovery_lib_version=self.recovery_lib_version)
        return record

class EpisodeRecorderRegistry:
    """Owns the always-on state recorder and a map of ``job_key -> EpisodeRecorder``.

    Wire ``state_source`` to a dedicated SUB-side :class:`ZmqBridgeClient`. Without
    one, the buffer must be filled manually (tests).
    """

    def __init__(self, *, state_source: Any=None, buffer: Optional[TrajectoryRingBuffer]=None, default_cameras: Optional[list[CameraSpec]]=None, poll_hz: float=50.0, quat_order: str='xyzw', frame_source: Any=None, frame_buffer: Optional[FrameRingBuffer]=None, frame_dir: str='logs/recovery_episodes/frames', frame_poll_hz: float=6.0, frame_save_depth: bool=True, clock=time.time, robot: str='fr3', gello_version: Optional[str]=None, recovery_lib_version: Optional[str]=None) -> None:
        self.buffer = buffer if buffer is not None else TrajectoryRingBuffer()
        self._clock = clock
        self._provenance = dict(robot=robot, gello_version=gello_version, recovery_lib_version=recovery_lib_version)
        self._default_cameras = list(default_cameras or [])
        self._stream = StateStreamRecorder(state_source, self.buffer, poll_hz=poll_hz, quat_order=quat_order) if state_source is not None else None
        self.frame_buffer = frame_buffer if frame_buffer is not None else FrameRingBuffer(on_evict=FrameStreamRecorder._unlink_frame)
        self._frame_stream = FrameStreamRecorder(frame_source, self.frame_buffer, out_dir=frame_dir, poll_hz=frame_poll_hz, save_depth=frame_save_depth, clock=clock) if frame_source is not None else None
        self._episodes: dict[str, EpisodeRecorder] = {}
        self._recovery_objects: dict[str, str] = {}
        self._recovery_points: dict[str, tuple[float, float]] = {}

    def start(self) -> None:
        if self._stream is not None:
            self._stream.start()
        if self._frame_stream is not None:
            self._frame_stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
        if self._frame_stream is not None:
            self._frame_stream.stop()

    def start_episode(self, job_key: str, goal: str, *, episode_id: Optional[str]=None) -> EpisodeRecorder:
        rec = EpisodeRecorder(episode_id or f'{job_key}:{uuid.uuid4().hex[:8]}', goal, self.buffer, frame_buffer=self.frame_buffer, clock=self._clock, **self._provenance)
        for cam in self._default_cameras:
            rec.add_camera(cam)
        self._episodes[job_key] = rec
        self._recovery_objects.pop(job_key, None)
        self._recovery_points.pop(job_key, None)
        return rec

    def ensure_episode(self, job_key: str, goal: str='', *, episode_id: Optional[str]=None) -> EpisodeRecorder:
        """Return the live recorder, creating it before the supervised stream.

        Local hold-loss can precede the first remote-Judge poll. Creating the
        recorder when the checkpoint arm is armed preserves the pre-failure
        trajectory window; re-arming after handback must not replace it.
        """
        rec = self._episodes.get(job_key)
        return rec if rec is not None else self.start_episode(job_key, goal, episode_id=episode_id)

    def get(self, job_key: str) -> Optional[EpisodeRecorder]:
        return self._episodes.get(job_key)

    def set_gripper_sample_hook(self, fn) -> None:
        """Attach a per-sample hook to the state stream, when a stream is configured."""
        if self._stream is not None:
            self._stream.set_on_sample(fn)

    def recent_gripper_samples(self, n: int=240) -> list[tuple[float, float]]:
        """Return the latest n timestamped gripper-width samples in chronological order."""
        snap = self.buffer.snapshot()
        if not snap:
            return []
        tail = snap[-max(1, int(n)):]
        return [(float(getattr(s, 'ts', 0.0) or 0.0), float(getattr(s, 'gripper_width', 0.0) or 0.0)) for s in tail]

    def set_recovery_object(self, job_key: str, name: Optional[str]) -> None:
        """Stash the latest non-empty Judge-named recovery target for this job."""
        name = (name or '').strip()
        if name:
            self._recovery_objects[job_key] = name

    def get_recovery_object(self, job_key: str) -> str:
        """Latest Judge-named recovery target for this job (``""`` if none yet)."""
        return self._recovery_objects.get(job_key, '')

    def set_recovery_point(self, job_key: str, point: Optional[tuple[float, float]]) -> None:
        """Stash the latest Judge-proposed normalized (x, y) anchor for this job."""
        if point is not None:
            self._recovery_points[job_key] = (float(point[0]), float(point[1]))

    def get_recovery_point(self, job_key: str) -> Optional[tuple[float, float]]:
        """Latest Judge anchor for this job, or ``None`` if none yet."""
        return self._recovery_points.get(job_key)

    def add_judge_tick(self, job_key: str, status_or_verdict: str, *, goal: str='', **kw) -> Optional[JudgeTick]:
        """Append a judge tick, auto-creating the episode on first tick (so no
        separate episode-start hook is required)."""
        rec = self._episodes.get(job_key)
        if rec is None:
            rec = self.start_episode(job_key, goal)
        return rec.add_judge_tick(status_or_verdict, **kw)

    def set_recovery(self, job_key: str, inv: RecoveryInvocation) -> bool:
        """Persist the recovery trace onto this job's episode so a subsequent
        :meth:`build`/dump carries it. Returns False if the episode is gone."""
        rec = self._episodes.get(job_key)
        if rec is None:
            return False
        rec.set_recovery(inv)
        return True

    def build(self, job_key: str) -> Optional[EpisodeRecord]:
        rec = self._episodes.get(job_key)
        return rec.build() if rec is not None else None

    def finalize(self, job_key: str, *, success: Optional[bool]=None) -> Optional[EpisodeRecord]:
        rec = self._episodes.get(job_key)
        if rec is None:
            return None
        rec.stop(success=success)
        return rec.build()

    def discard(self, job_key: str) -> None:
        self._episodes.pop(job_key, None)
        self._recovery_objects.pop(job_key, None)
        self._recovery_points.pop(job_key, None)
