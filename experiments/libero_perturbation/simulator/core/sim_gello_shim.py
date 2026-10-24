"""Expose the LIBERO simulator through camera, control and joint-state services."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_STARVLA_DIR = os.environ.get("STARVLA_DIR")
if _STARVLA_DIR and _STARVLA_DIR not in sys.path:
    sys.path.insert(0, _STARVLA_DIR)

from disturbances import DisturbanceConfig, make_disturbance  # noqa: E402
from sim_world import Mode, SimWorld  # noqa: E402
from terminal_result import TerminalResultWriter  # noqa: E402
from video_io import StreamingMp4Writer  # noqa: E402

# The first env.step initializes task-specific renderer and physics state.
# Job and judge clocks start after this complete-cycle handshake returns.
INITIAL_CYCLE_TIMEOUT_S = 60.0

log = logging.getLogger("sim-shim")


# --------------------------------------------------------------------------- #
# JPEG / PNG encode helpers                                                   #
# --------------------------------------------------------------------------- #
def _encode_jpeg(rgb: np.ndarray) -> bytes:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _encode_depth_png(depth_m: np.ndarray, depth_scale: float) -> bytes:
    import cv2

    # metric meters -> uint16 raw with the advertised depth_scale (meters/unit).
    raw = np.clip(depth_m / max(depth_scale, 1e-9), 0, 65535).astype(np.uint16)
    ok, buf = cv2.imencode(".png", raw)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# debug video recorder                                                        #
# --------------------------------------------------------------------------- #
class VideoRecorder:
    """Best-effort RGB recorder for one shim episode.

    The SimWorld loop owns MuJoCo rendering; this helper only samples the latest
    published camera frame and writes it to disk. If no frame is ready yet, it
    waits instead of touching the env.
    """

    def __init__(self, world: SimWorld, path: str, *, fps: float = 10.0) -> None:
        self.world = world
        self.path = pathlib.Path(path)
        self.fps = max(1e-3, float(fps))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: StreamingMp4Writer | None = None
        self.frames = 0

    def start(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="debug-video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.frames:
            log.info("wrote debug video %s (%d frames)", self.path, self.frames)

    def _run(self) -> None:
        period = 1.0 / self.fps
        writer = StreamingMp4Writer(self.path, fps=self.fps)
        self._writer = writer
        while not self._stop.is_set():
            frame = self.world.get_frame()
            if frame is None:
                time.sleep(min(period, 0.05))
                continue
            try:
                writer.append(frame.rgb)
            except Exception as exc:
                log.warning("could not write debug video frame: %s", exc)
                return
            self.frames = writer.frames
            time.sleep(period)


class EpisodeTimingBoundary:
    """Own the post-start video and wall-clock boundary for one episode."""

    def __init__(self, world, recorder, timing_file: str, process_start_ts: float):
        self.world = world
        self.recorder = recorder
        self.timing_file = pathlib.Path(timing_file) if timing_file else None
        self.process_start_ts = float(process_start_ts)
        self.services_ready_ts = None
        self.episode_start_ts = None
        self.episode_started = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        first_motion = self.world.first_robot_motion_ts
        startup_ms = None
        ready_to_start_ms = None
        episode_to_motion_ms = None
        if self.services_ready_ts is not None:
            startup_ms = (self.services_ready_ts - self.process_start_ts) * 1000.0
        if self.services_ready_ts is not None and self.episode_start_ts is not None:
            ready_to_start_ms = (self.episode_start_ts - self.services_ready_ts) * 1000.0
        if self.episode_start_ts is not None and first_motion is not None:
            episode_to_motion_ms = (first_motion - self.episode_start_ts) * 1000.0
        return {
            "schema": "libero-episode-timing-boundary-v1",
            "process_start_ts": self.process_start_ts,
            "services_ready_ts": self.services_ready_ts,
            "episode_start_ts": self.episode_start_ts,
            "first_robot_motion_ts": first_motion,
            "startup_cost_ms": startup_ms,
            "ready_to_episode_start_ms": ready_to_start_ms,
            "episode_to_first_robot_motion_ms": episode_to_motion_ms,
            "episode_wall_time_origin": "episode_start_ts",
            "video_origin": "episode_start_ts",
            "checkpoint_latency_scope": "warm online inference; startup excluded",
        }

    def persist(self) -> None:
        if self.timing_file is None:
            return
        self.timing_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.timing_file.with_suffix(self.timing_file.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self.timing_file)

    def start(self, services_ready_ts=None) -> dict:
        with self._lock:
            if self.episode_start_ts is not None:
                return {"accepted": False, "error": "episode already started",
                        **self.snapshot()}
            now = time.time()
            ready = now if services_ready_ts is None else float(services_ready_ts)
            if ready < self.process_start_ts or ready > now + 1.0:
                return {"accepted": False, "error": "invalid services_ready_ts"}
            self.services_ready_ts = ready
            self.episode_start_ts = now
            self.world._episode_deadline_wall = now + self.world._episode_wall_cap_s
            if self.recorder is not None:
                self.recorder.start()
            self.episode_started.set()
        self.persist()
        return {"accepted": True, **self.snapshot()}


# --------------------------------------------------------------------------- #
# HTTP control plane (:18080)                                                 #
# --------------------------------------------------------------------------- #
class ControlState:
    def __init__(self, world: SimWorld, episode_boundary=None):
        self.world = world
        self.episode_boundary = episode_boundary
        self.active_request_id: str | None = None
        self.callback_url: str = ""
        self.standby = False
        self.lock = threading.Lock()


def make_control_handler(cs: ControlState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code: int, body: dict):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode() or "{}")
            except Exception:
                return {}

        def do_GET(self):
            if self.path.rstrip("/") == "/healthz":
                self._json(200, {"ok": True})
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            body = self._read_body()
            if self.path.rstrip("/") == "/execute":
                self._handle_execute(body)
            elif self.path.rstrip("/") == "/episode/start":
                if cs.episode_boundary is None:
                    self._json(404, {"accepted": False, "error": "no episode boundary"})
                else:
                    reply = cs.episode_boundary.start(body.get("services_ready_ts"))
                    self._json(200 if reply.get("accepted") else 409, reply)
            elif self.path.startswith("/cancel/"):
                self._handle_cancel(self.path[len("/cancel/"):], body)
            elif self.path.rstrip("/") == "/complete_terminal":
                request_id = str(body.get("request_id", "") or "")
                log.info("/complete_terminal request_id=%s", request_id)
                reply = cs.world.request_complete_terminal(request_id)
                self._json(200 if reply.get("accepted") else 409, reply)
            elif self.path.rstrip("/") == "/complete_terminal/status":
                request_id = str(body.get("request_id", "") or "")
                reply = cs.world.complete_terminal_status(request_id)
                self._json(200 if reply.get("ok") else 404, reply)
            elif self.path.rstrip("/") == "/pregrasp_resume":
                request_id = str(body.get("request_id", "") or "")
                log.info("/pregrasp_resume request_id=%s", request_id)
                reply = cs.world.request_pregrasp_resume(request_id)
                self._json(200 if reply.get("accepted") else 409, reply)
            elif self.path.rstrip("/") == "/pregrasp_resume/status":
                request_id = str(body.get("request_id", "") or "")
                reply = cs.world.pregrasp_resume_status(request_id)
                self._json(200 if reply.get("ok") else 404, reply)
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def _handle_execute(self, body: dict):
            if (cs.episode_boundary is not None
                    and not cs.episode_boundary.episode_started.is_set()):
                self._json(409, {"accepted": False,
                                 "error": "episode start boundary not opened"})
                return
            instruction = body.get("instruction", "")
            inputs = body.get("inputs", {}) or {}
            request_id = body.get("request_id") or f"sim-{int(time.time()*1000)}"
            callback_url = body.get("callback_url", "") or ""
            skip_reset = bool(inputs.get("skip_reset"))
            with cs.lock:
                cs.active_request_id = request_id
                cs.callback_url = callback_url
                cs.standby = False
            log.info("/execute request_id=%s skip_reset=%s instr=%r",
                     request_id, skip_reset, instruction[:60])
            resume_generation = cs.world.submit_execute(
                instruction, inputs, request_id, callback_url
            )
            cycle_ts = cs.world.wait_first_cycle_completed_for_generation(
                resume_generation, timeout_s=INITIAL_CYCLE_TIMEOUT_S
            )
            if cycle_ts is None:
                log.error(
                    "/execute request_id=%s timed out before first complete VLA cycle",
                    request_id,
                )
                cs.world.request_cancel(standby=False)
                cs.world.request_failed_terminal(request_id, "cap_refusal")
                self._json(504, {
                    "accepted": False, "error": "first complete VLA cycle timeout",
                    "request_id": request_id,
                })
                return
            first_action_ts = cs.world.first_action_for_generation(
                resume_generation
            )
            if first_action_ts is None:
                self._json(500, {
                    "accepted": False,
                    "error": "first action generation mismatch",
                    "request_id": request_id,
                    "resume_generation": resume_generation,
                })
                return
            log.info(
                "/execute accepted after first complete VLA cycle "
                "request_id=%s generation=%d action_ts=%.6f cycle_ts=%.6f",
                request_id, resume_generation, first_action_ts, cycle_ts,
            )
            self._json(200, {
                "accepted": True, "job_id": f"sim-job-{request_id}",
                "resume_generation": resume_generation,
                "first_action_returned_ts": first_action_ts,
                "first_cycle_completed_ts": cycle_ts,
            })

        def _handle_cancel(self, request_id: str, body: dict):
            standby = bool(body.get("standby", False))
            return_home = bool(body.get("return_home", False))
            terminal_reason = str(body.get("terminal_reason", "cancel") or "cancel")
            log.info(
                "/cancel/%s standby=%s return_home=%s terminal_reason=%s",
                request_id, standby, return_home, terminal_reason,
            )
            cs.world.request_cancel(standby=standby)
            # Standby is a recovery pause. A plain/non-standby cancel is a final
            # plan cancellation and must drive the simulator to an explicit
            # failed terminal; otherwise cap exhaustion can leave the world
            # alive until EP_TIMEOUT and write success=null.
            terminal = None
            if not standby:
                terminal = cs.world.request_failed_terminal(request_id, terminal_reason)
            with cs.lock:
                cs.standby = standby
            self._json(200, {
                "ok": True, "cancelled": True,
                "return_home": return_home, "standby": standby,
                "terminal": terminal,
            })

    return Handler


# --------------------------------------------------------------------------- #
# HTTP camera plane (:18110)                                                  #
# --------------------------------------------------------------------------- #
def _camera_frame_headers(frame) -> dict[str, str]:
    """Bind JPEG bytes to the producer and simulation frame that created them."""
    headers = {
        'X-CarryOn-History-Session': os.environ['QUALIFICATION_HISTORY_SESSION'],
        "X-CarryOn-Producer-Frame-Seq": str(int(frame.producer_frame_seq)),
        "X-CarryOn-Producer-Frame-Ts": f"{float(frame.ts):.9f}",
        "X-CarryOn-Camera-Role": str(frame.camera_role),
        "X-CarryOn-Camera-Source": str(frame.camera_source),
    }
    if frame.simulation_step_id is not None:
        headers["X-CarryOn-Simulation-Step-Id"] = str(
            int(frame.simulation_step_id)
        )
    if frame.simulation_time_s is not None:
        headers["X-CarryOn-Simulation-Time-S"] = (
            f"{float(frame.simulation_time_s):.9f}"
        )
    return headers


def make_camera_handler(world: SimWorld):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, payload: bytes, headers=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            for name, value in (headers or {}).items():
                self.send_header(str(name), str(value))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, code, body: dict):
            self._send(code, "application/json", json.dumps(body).encode())

        def do_GET(self):
            import urllib.parse as _up

            parsed = _up.urlparse(self.path)
            # Return the simulator object's ground-truth normalized pixel coordinates.
            if parsed.path.rstrip("/") == "/target_px":
                qs = _up.parse_qs(parsed.query)
                name = (qs.get("name", [""])[0] or "").strip()
                if not name:
                    oi = getattr(world, "obj_of_interest", None) or []
                    name = oi[0] if oi else ""
                px = world.object_target_px(name) if name else None
                if px is None:
                    self._send_json(503, {"error": f"no target_px for {name!r}"})
                    return
                self._send_json(200, {"name": name, "px": [px[0], px[1]]})
                return
            if parsed.path.rstrip("/") == "/object_xy":
                qs = _up.parse_qs(parsed.query)
                name = (qs.get("name", [""])[0] or "").strip()
                if not name:
                    oi = getattr(world, "obj_of_interest", None) or []
                    name = oi[0] if oi else ""
                xy = world.object_world_xy(name) if name else None
                if xy is None:
                    self._send_json(503, {"error": f"no object_xy for {name!r}"})
                    return
                self._send_json(200, {"name": name, "xy": [xy[0], xy[1]]})
                return
            # Displace a simulator object by world-frame (dx, dy).
            # Return its new XY, normalized pixel coordinates and in-frame flag.
            if parsed.path.rstrip("/") == "/displace_target":
                qs = _up.parse_qs(parsed.query)
                name = (qs.get("name", [""])[0] or "").strip()
                if not name:
                    oi = getattr(world, "obj_of_interest", None) or []
                    name = oi[0] if oi else ""
                try:
                    dx = float(qs.get("dx", ["0"])[0])
                    dy = float(qs.get("dy", ["0"])[0])
                except (TypeError, ValueError):
                    self._send_json(400, {"error": "bad dx/dy"})
                    return
                # Marshaled onto the loop thread (EGL-affine teleport + re-render).
                res = world.request_displace(name, dx, dy)
                self._send_json(200 if "error" not in res else 500, res)
                return
            parts = parsed.path.strip("/").split("/")
            # /cameras/<name>/<file>
            if len(parts) != 3 or parts[0] != "cameras":
                self._send_json(404, {"error": "not found"})
                return
            _, _name, fname = parts
            if fname == 'history.jpg':
                from qualification_history import history_response
                status,body,headers=history_response(getattr(world,'_qualification_rgb_history',None),parsed.query,_encode_jpeg,_camera_frame_headers)
                self._send(status,'image/jpeg',body,headers=headers)
                return
            frame = world.get_frame()
            if frame is None:
                self._send_json(503, {"error": "no frame yet"})
                return
            try:
                if fname in ("latest.jpg", "latest_raw.jpg"):
                    self._send(
                        200,
                        "image/jpeg",
                        _encode_jpeg(frame.rgb),
                        headers=_camera_frame_headers(frame),
                    )
                elif fname == "latest_depth.png":
                    if frame.depth_m is None:
                        self._send_json(503, {"error": "no depth"})
                        return
                    self._send(200, "image/png",
                               _encode_depth_png(frame.depth_m, frame.intrinsics.get("depth_scale", 0.001)))
                elif fname == "intrinsics.json":
                    self._send_json(200, frame.intrinsics)
                else:
                    self._send_json(404, {"error": "unknown camera file"})
            except Exception as e:  # noqa: BLE001
                self._send_json(500, {"error": str(e)})

    return Handler


# --------------------------------------------------------------------------- #
# ZMQ bridge (6003 REP / 6004 PULL / 6005 PUB)                                #
# --------------------------------------------------------------------------- #
class ZmqBridgeServer:
    def __init__(self, world: SimWorld, *, host="127.0.0.1",
                 rep_port=6003, pull_port=6004, pub_port=6005, pub_hz=15.0):
        import zmq

        self.world = world
        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.rep = ctx.socket(zmq.REP); self.rep.bind(f"tcp://{host}:{rep_port}")
        self.pull = ctx.socket(zmq.PULL); self.pull.bind(f"tcp://{host}:{pull_port}")
        self.pub = ctx.socket(zmq.PUB); self.pub.bind(f"tcp://{host}:{pub_port}")
        self._period = 1.0 / pub_hz
        self._stop = threading.Event()
        self._threads = []

    def start(self):
        for fn, name in ((self._run_pub, "zmq-pub"), (self._run_pull, "zmq-pull"),
                         (self._run_rep, "zmq-rep")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()

    def _run_pub(self):
        while not self._stop.is_set():
            st = self.world.get_bridge_state()
            if st is not None:
                try:
                    self.pub.send_json(st)
                except Exception:
                    pass
            time.sleep(self._period)

    def _run_pull(self):
        poller = self._zmq.Poller()
        poller.register(self.pull, self._zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            if self.pull in socks:
                try:
                    msg = self.pull.recv_json()
                except Exception:
                    continue
                if isinstance(msg, dict) and msg.get("type") == "cmd":
                    self.world.push_joint_command(msg.get("arm", []), msg.get("gripper", 0.0))

    def _run_rep(self):
        poller = self._zmq.Poller()
        poller.register(self.rep, self._zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            if self.rep in socks:
                try:
                    msg = self.rep.recv_json()
                    self.rep.send_json({"status": "READY"})
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# callback poster                                                             #
# --------------------------------------------------------------------------- #
def post_callback(url: str, payload: dict) -> None:
    if not url:
        return
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5.0).read()
    except Exception as e:  # noqa: BLE001
        log.warning("callback POST to %s failed: %s", url, e)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    process_start_ts = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite-name", default="libero_object")
    ap.add_argument(
        "--task-description-override",
        default="",
        help="Adapter-provided official BDDL instruction. Empty preserves suite language.",
    )
    ap.add_argument("--horizon", type=int, default=0,
                    help="override the LIBERO episode horizon (env steps); 0 = suite default. "
                         "Protocol note: changing this REQUIRES rerunning C0 at the same value.")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--percep-resolution", type=int, default=0,
                    help="Overhead-camera render size for Judge+perception; 0=same as --resolution. "
                         "Higher (e.g. 512) improves SAM3 grounding on small LIBERO objects.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--camera-name", default="agentview")
    # disturbance
    ap.add_argument("--disturbance", default="none")
    # Output label (e.g. LIBERO-plus factor "robot"); recorded in the result JSON
    # for per-factor reporting. Defaults to the disturbance kind.
    ap.add_argument("--label", default="")
    ap.add_argument("--offset-x", type=float, default=0.06)
    ap.add_argument("--offset-y", type=float, default=0.0)
    ap.add_argument("--lift-trigger-dz", type=float, default=0.04)
    ap.add_argument("--trigger-step", type=int, default=-1)
    ap.add_argument("--freeze-start-step", type=int, default=30)
    ap.add_argument("--freeze-duration", type=int, default=25)
    # ee_trans_{x,y,z}: FailSafe noise*(rand-0.5) or fixed action_bias per step.
    ap.add_argument("--trans-noise", type=float, default=1.0)
    ap.add_argument("--trans-randomize", action="store_true", default=True)
    ap.add_argument("--no-trans-randomize", dest="trans_randomize", action="store_false")
    ap.add_argument("--action-bias", type=float, default=0.20)
    ap.add_argument("--trans-steps", type=int, default=8)
    # hold_lost: per-step drop probability after lift (1.0 = deterministic).
    ap.add_argument("--drop-prob", type=float, default=1.0)
    ap.add_argument("--drop-scatter", type=float, default=0.0,
                    help="hold_lost: random-direction landing scatter radius (m); 0 = drop in place")
    ap.add_argument("--drop-offset-x", type=float, default=0.0,
                    help="hold_lost OBJECT-RELOCATION: deterministic dx (m, table frame) the "
                         "dropped object is relocated by; overrides --drop-scatter when nonzero")
    ap.add_argument("--drop-offset-y", type=float, default=0.0,
                    help="hold_lost OBJECT-RELOCATION: deterministic dy (m, table frame)")
    ap.add_argument("--rot-phase", default="pre_grasp",
                    help="ee_rot_*: pre_grasp | carry (twist while holding)")
    # vla
    ap.add_argument("--vla-host", default="127.0.0.1")
    ap.add_argument("--vla-port", type=int, default=6694)
    ap.add_argument("--unnorm-key", default=None)
    # ports
    ap.add_argument("--control-port", type=int, default=18080)
    ap.add_argument("--camera-port", type=int, default=18110)
    ap.add_argument("--bridge-host", default="127.0.0.1")
    ap.add_argument("--rep-port", type=int, default=6003)
    ap.add_argument("--pull-port", type=int, default=6004)
    ap.add_argument("--pub-port", type=int, default=6005)
    ap.add_argument("--close-width-m", type=float, default=0.02)
    ap.add_argument("--result-file", default="")
    ap.add_argument("--video-file", default="",
                    help="Optional debug RGB mp4 written from the shim camera stream.")
    ap.add_argument("--video-fps", type=float, default=10.0)
    ap.add_argument(
        "--defer-episode-start", action="store_true",
        help="Wait for POST /episode/start before video and wall-cap clocks begin.",
    )
    ap.add_argument(
        "--timing-file", default="",
        help="Optional atomic episode timing sidecar.",
    )
    ap.add_argument(
        "--keep-alive-after-episode",
        action="store_true",
        help="Keep control/camera/ZMQ endpoints alive after the LIBERO horizon "
        "is reached. Probe scripts use this so CarryOn recovery can finish "
        "after standby cancel without losing the bridge mid-goto_init.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s| %(message)s",
                        datefmt="%H:%M:%S")

    backend = os.environ.get("VLA_BACKEND", "starvla").strip().lower()
    if backend == "pi05":
        from pi05_client import Pi05ModelClient as ModelClient
    elif backend == "starvla":
        from examples.LIBERO.eval_files.model2libero_interface import ModelClient
    else:
        raise SystemExit(
            f"unknown VLA_BACKEND={backend!r}; expected starvla|pi05"
        )
    log.info("VLA_BACKEND=%s", backend)

    world = SimWorld(
        task_suite_name=args.task_suite_name, camera_name=args.camera_name,
        horizon_override=int(args.horizon or 0),
        resolution=args.resolution,
        percep_resolution=(args.percep_resolution or None),
        seed=args.seed, close_width_m=args.close_width_m,
    )
    if backend == "pi05":
        world.vla_client = ModelClient(host=args.vla_host, port=args.vla_port)
    else:
        world.vla_client = ModelClient(
            host=args.vla_host, port=args.vla_port, unnorm_key=args.unnorm_key
        )
    world.disturbance = make_disturbance(DisturbanceConfig(
        kind=args.disturbance, lift_trigger_dz=args.lift_trigger_dz,
        drop_prob=args.drop_prob, drop_scatter_m=args.drop_scatter,
        drop_offset_xy=(args.drop_offset_x, args.drop_offset_y),
        rot_phase=args.rot_phase, action_bias=args.action_bias,
        trans_noise=args.trans_noise, trans_randomize=args.trans_randomize,
        trans_steps=args.trans_steps,
        offset_xy=(args.offset_x, args.offset_y), trigger_step=args.trigger_step,
        freeze_start_step=args.freeze_start_step, freeze_duration=args.freeze_duration,






        seed=args.seed + args.task_id * 1000 + args.episode_idx,
    ))

    log.info("building env %s task=%d ep=%d ...", args.task_suite_name, args.task_id, args.episode_idx)
    # build_env must run on the loop thread (MuJoCo EGL context is thread-affine),
    # so defer it into world.start() and block here until the env is ready.
    world.start_with_env(task_id=args.task_id, episode_idx=args.episode_idx)
    if args.task_description_override:
        world._task_description = args.task_description_override
    world.vla_client.reset(task_description=world._task_description)
    warm_latency_s = world.warm_vla_once()
    world.vla_client.reset(task_description=world._task_description)
    log.info("[preflight] VLA warm inference PASS latency=%.3fs env_steps=0",
             warm_latency_s)
    log.info("env ready: task=%r", world._task_description)

    recorder = VideoRecorder(world, args.video_file, fps=args.video_fps) if args.video_file else None
    timing_file = args.timing_file
    if not timing_file and args.result_file:
        timing_file = args.result_file + ".timing.json"
    episode_boundary = EpisodeTimingBoundary(
        world, recorder, timing_file, process_start_ts
    )
    default_cap = 3600.0 if args.keep_alive_after_episode else 600.0
    try:
        world._episode_wall_cap_s = float(os.environ.get("SHIM_MAX_LIFETIME_S", default_cap))
    except (TypeError, ValueError):
        world._episode_wall_cap_s = default_cap
    if not np.isfinite(world._episode_wall_cap_s) or world._episode_wall_cap_s <= 0:
        raise ValueError("invalid existing episode wall cap")
    cs = ControlState(world, episode_boundary)
    protocol_ledger = pathlib.Path(os.environ["PROTOCOL_RECOVERY_LEDGER"])
    protocol_lock = threading.Lock()
    def protocol_emit(value):
        line = (json.dumps(value, sort_keys=True) + "\n").encode()
        with protocol_lock:
            with protocol_ledger.open("ab") as stream:
                stream.write(line); stream.flush(); os.fsync(stream.fileno())
    protocol_emit({'event':'PROTOCOL_DIRECT_TRIGGER_DISABLED','judge_control_enabled':True})
    done_event = threading.Event()

    def _result_payload():
        return {
                "task_id": args.task_id, "episode_idx": args.episode_idx,
                "init_state_index": args.episode_idx,
                "base_seed": args.seed,
                "disturbance_seed": args.seed + args.task_id * 1000 + args.episode_idx,
                "task_suite": args.task_suite_name,
                "disturbance": args.disturbance,
                "label": args.label or args.disturbance,
                "vla_steps": int(getattr(world, "_step", 0)),
                "max_vla_steps": int(args.horizon),
                "events": list(getattr(world.disturbance, "events", [])),
                "vla_warmup_latency_s": world.vla_warmup_latency_s,
                "first_action_returned_ts": world.episode_first_action_returned_ts,
                "first_cycle_completed_ts": world.episode_first_cycle_completed_ts,
                "first_robot_motion_ts": world.first_robot_motion_ts,
                "episode_timing": episode_boundary.snapshot(),
                "offline_gt_timeline": list(
                    getattr(world, "_offline_gt_timeline", [])
                ),
        }

    terminal_writer = TerminalResultWriter(args.result_file, _result_payload)

    def request_shutdown(_signum=None, _frame=None):
        # run_eval's EP_TIMEOUT uses SIGTERM; an interactive Ctrl-C is a cancel.
        reason = (
            "wall_cap"
            if _signum == signal.SIGTERM and episode_boundary.episode_started.is_set()
            else "cancel"
        )
        terminal_writer.finalize(reason)
        done_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    def on_complete(success: bool, terminal_reason: str):
        if bool(success) and terminal_reason != "normal_success":
            raise ValueError("successful episode must use normal_success")
        if not bool(success) and terminal_reason == "normal_success":
            raise ValueError("failed episode cannot use normal_success")
        terminal_writer.finalize(terminal_reason)
        post_callback(cs.callback_url, {
            "status": "success" if success else "failed",
            "result": {"source": "sim_env", "task_success": bool(success)},
            "meta": {"source": "sim_env"},
        })
        if args.keep_alive_after_episode:
            log.info(
                "episode complete (success=%s); keeping shim alive for recovery probe",
                success,
            )
            return
        done_event.set()

    # The loop thread is already running (started by start_with_env); just attach
    # the completion callback now that it is defined.
    world.on_episode_complete = on_complete

    ctrl_srv = ThreadingHTTPServer(("0.0.0.0", args.control_port), make_control_handler(cs))
    cam_srv = ThreadingHTTPServer(("0.0.0.0", args.camera_port), make_camera_handler(world))
    threading.Thread(target=ctrl_srv.serve_forever, daemon=True).start()
    threading.Thread(target=cam_srv.serve_forever, daemon=True).start()
    bridge = ZmqBridgeServer(world, host=args.bridge_host, rep_port=args.rep_port,
                             pull_port=args.pull_port, pub_port=args.pub_port)
    bridge.start()
    log.info("shim up: control=:%d camera=:%d zmq=%s:%d/%d/%d",
             args.control_port, args.camera_port, args.bridge_host,
             args.rep_port, args.pull_port, args.pub_port)
    if args.defer_episode_start:
        log.info("episode clocks deferred until POST /episode/start")
    else:
        reply = episode_boundary.start(services_ready_ts=time.time())
        if not reply.get("accepted"):
            raise RuntimeError(f"could not open episode boundary: {reply}")
        if recorder is not None:
            log.info("debug video recording to %s at %.1f FPS", args.video_file, args.video_fps)

    try:
        while not done_event.is_set() and not episode_boundary.episode_started.wait(0.1):
            pass
        if (args.defer_episode_start and episode_boundary.episode_started.is_set()
                and recorder is not None):
            log.info("debug video recording to %s at %.1f FPS", args.video_file, args.video_fps)
        # Run until completion or SHIM_MAX_LIFETIME_S. In keep-alive mode the cap
        # includes both policy execution and recovery.
        default_cap = 3600.0 if args.keep_alive_after_episode else 600.0
        try:
            cap_s = float(os.environ.get("SHIM_MAX_LIFETIME_S", default_cap))
        except (TypeError, ValueError):
            cap_s = default_cap
        if (not done_event.is_set()
                and episode_boundary.episode_started.is_set()
                and not done_event.wait(timeout=max(0.0, world._episode_deadline_wall - time.time()))):
            terminal_writer.finalize("wall_cap")
            done_event.set()
    except KeyboardInterrupt:
        terminal_writer.finalize("cancel")
    except BaseException:
        terminal_writer.finalize("exception")
        raise
    finally:
        episode_boundary.persist()
        # A single owner writes the complete terminal triple before EGL teardown.
        # No teardown path is allowed to manufacture or partially rewrite it.
        if terminal_writer.outcome is None:
            terminal_writer.finalize("exception")
        else:
            terminal_writer.commit()
        # Guard every teardown step so an EGL/context free crash can never lose
        # the result that is now already on disk.
        for _name, _fn in (
            ("recorder", (recorder.stop if recorder is not None else None)),
            ("world", world.stop),
            ("bridge", bridge.stop),
            ("ctrl_srv", ctrl_srv.shutdown),
            ("cam_srv", cam_srv.shutdown),
        ):
            if _fn is None:
                continue
            try:
                _fn()
            except Exception as _te:  # noqa: BLE001 - teardown must not lose the result
                log.warning("teardown step %s failed: %s", _name, _te)

    outcome = terminal_writer.outcome
    log.info(
        "shim exiting; success=%s timeout=%s terminal_reason=%s",
        outcome.success, outcome.timeout, outcome.terminal_reason,
    )

if __name__ == "__main__":
    main()
