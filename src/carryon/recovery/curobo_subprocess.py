"""Fail-closed cuRobo planner client using a separately-versioned Python env.

The LIBERO process intentionally imports no cuRobo or CUDA torch symbols.  A
persistent runner in the curobo2 environment owns the heavy planner and exchanges
atomic JSON request/response files with this client.  Planner stdout is evidence,
not protocol, so third-party logging cannot corrupt responses.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class CuroboSubprocessConfig:
    python_exe: str
    runner_path: str
    repo_root: str
    ipc_dir: str
    cuda_visible_devices: str
    planner_config: dict[str, Any]
    startup_timeout_s: float = 240.0
    request_timeout_s: float = 120.0
    poll_interval_s: float = 0.02


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class CuroboSubprocessClient:
    """MotionPlanner-compatible proxy with persistent runner ownership."""

    def __init__(self, config: CuroboSubprocessConfig) -> None:
        self.config = config
        self.last_error: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._log_stream = None
        self._ipc = Path(config.ipc_dir).resolve()
        self._repo = Path(config.repo_root).resolve()
        self._python = Path(config.python_exe).resolve()
        self._runner = Path(config.runner_path).resolve()
        self._closed = False
        atexit.register(self.close)

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    def _log_tail(self, limit: int = 6000) -> str:
        path = self._ipc / "planner_runner.log"
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]

    def _ensure_server(self) -> bool:
        if self._closed:
            self.last_error = "cuRobo subprocess client already closed"
            return False
        if self._proc is not None and self._proc.poll() is None:
            return True
        missing = [str(path) for path in (self._python, self._runner) if not path.is_file()]
        if missing:
            self.last_error = "missing cuRobo subprocess resource(s): " + ", ".join(missing)
            return False
        self._ipc.mkdir(parents=True, exist_ok=False)
        _atomic_json(self._ipc / "PLANNER_CONFIG.json", self.config.planner_config)
        _atomic_json(
            self._ipc / "CLIENT_CONFIG.json",
            {**asdict(self.config), "planner_config": self.config.planner_config},
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._repo) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = str(self.config.cuda_visible_devices)
        cmd = [
            str(self._python),
            str(self._runner),
            "--serve",
            "--ipc-dir", str(self._ipc),
            "--config", str(self._ipc / "PLANNER_CONFIG.json"),
        ]
        self._log_stream = (self._ipc / "planner_runner.log").open("xb")
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self._repo),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            self.last_error = f"cuRobo runner launch failed: {type(exc).__name__}: {exc}"
            self._log_stream.close()
            self._log_stream = None
            return False
        deadline = time.monotonic() + float(self.config.startup_timeout_s)
        ready = self._ipc / "READY.json"
        while time.monotonic() < deadline:
            if ready.is_file():
                payload = json.loads(ready.read_text(encoding="utf-8"))
                if payload.get("ready") is True and payload.get("pid") == self._proc.pid:
                    self.last_error = None
                    return True
                self.last_error = f"invalid cuRobo READY payload: {payload!r}"
                break
            if self._proc.poll() is not None:
                self.last_error = (
                    f"cuRobo runner exited before READY rc={self._proc.returncode}: "
                    f"{self._log_tail()}"
                )
                break
            time.sleep(float(self.config.poll_interval_s))
        else:
            self.last_error = f"cuRobo runner startup timeout after {self.config.startup_timeout_s}s"
        self.close()
        return False

    def _request(self, method: str, args: dict[str, Any]) -> Any:
        if not self._ensure_server():
            return None
        request_id = uuid.uuid4().hex
        request_path = self._ipc / f"REQUEST_{request_id}.json"
        response_path = self._ipc / f"RESPONSE_{request_id}.json"
        _atomic_json(
            request_path,
            {
                "schema": "carryon-curobo-planner-request/v1",
                "request_id": request_id,
                "method": method,
                "args": _plain(args),
                "created_wall_ts": time.time(),
            },
        )
        deadline = time.monotonic() + float(self.config.request_timeout_s)
        while time.monotonic() < deadline:
            if response_path.is_file():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if response.get("request_id") != request_id:
                    self.last_error = "cuRobo response request_id mismatch"
                    return None
                if response.get("ok") is not True:
                    self.last_error = str(response.get("error") or "cuRobo subprocess request failed")
                    return None
                self.last_error = response.get("planner_error")
                return response.get("result")
            if self._proc is None or self._proc.poll() is not None:
                rc = None if self._proc is None else self._proc.returncode
                self.last_error = f"cuRobo runner died during {method} rc={rc}: {self._log_tail()}"
                return None
            time.sleep(float(self.config.poll_interval_s))
        self.last_error = f"cuRobo subprocess timeout method={method} limit={self.config.request_timeout_s}s"
        self.close()
        return None

    def plan_to_q(self, q_start: Sequence[float], q_goal: Sequence[float]):
        return self._request("plan_to_q", {"q_start": q_start, "q_goal": q_goal})

    def plan_to_pose(self, q_start, position, quaternion_wxyz):
        return self._request(
            "plan_to_pose",
            {"q_start": q_start, "position": position, "quaternion_wxyz": quaternion_wxyz},
        )

    def plan_grasp_goalset(
        self, q_start, grasps, *, tcp_offset_m=0.1034, pre_grasp_dis=0.10,
        plan_grasp_to_lift=False,
    ):
        rows = [
            {
                "position": list(grasp.position),
                "quaternion": list(grasp.quaternion),
                "score": grasp.score,
                "source": grasp.source,
            }
            for grasp in grasps
        ]
        return self._request(
            "plan_grasp_goalset",
            {
                "q_start": q_start,
                "grasps": rows,
                "tcp_offset_m": tcp_offset_m,
                "pre_grasp_dis": pre_grasp_dis,
                "plan_grasp_to_lift": plan_grasp_to_lift,
            },
        )

    def ee_pose(self, q):
        result = self._request("ee_pose", {"q": q})
        if result is None:
            return None
        return (result[0], result[1])

    def set_scene_pointcloud(self, points_world, **kwargs) -> bool:
        result = self._request(
            "set_scene_pointcloud", {"points_world": _plain(points_world), "kwargs": kwargs}
        )
        return bool(result)

    def clear_scene(self) -> None:
        self._request("clear_scene", {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
        if self._log_stream is not None and not self._log_stream.closed:
            self._log_stream.close()

