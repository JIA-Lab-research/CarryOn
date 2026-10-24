"""Run module calls in a separate process without modifying the caller's imports."""
from __future__ import annotations

import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
import threading
import signal
import time

from .materialize import runtime_files
from ..services import process_identity,owned_processes,same_process

METHODS = frozenset({"reset", "commit", "decide", "state", "identity", "integration_records", "close"})
REQUIRED_ENV = ("SAFE_SAM3_MODEL", "SAFE_SAM3_PYTHON", "SAFE_SAM3_GPU",
                "SAFE_MOLMO_MODEL", "SAFE_MOLMO_PYTHON", "SAFE_MOLMO_GPU")


def _worker(channel, runtime, diagnostics, environment):
    if os.name=="posix":os.setsid()
    instance = None
    original_post = None
    try:
        identity = runtime_files(runtime)
        os.environ.update(environment)
        if environment.get("CARRYON_ROLE_REQUEST_BUDGET"):
            import requests
            from ..request_budget import BudgetedPost
            from ..files import read_json
            original_post=requests.post
            requests.post=BudgetedPost(read_json(environment["CARRYON_ROLE_REQUEST_BUDGET"]),
                Path(diagnostics).parent/"role_requests",original_post)
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise ValueError("Missing Safety resources: " + ", ".join(missing))
        path = Path(runtime) / "adapter/runtime.py"
        spec = importlib.util.spec_from_file_location("carryon_safety_worker_runtime", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        instance = module.FrozenRuntime(Path(runtime) / "adapter/load_method.py", diagnostics)
        channel.send((True, identity))
        while True:
            method, args = channel.recv()
            if method not in METHODS:
                raise ValueError("Unsupported Safety RPC method")
            result = getattr(instance, method)(*args)
            channel.send((True, result))
            if method == "close":
                instance = None
                break
    except EOFError:
        pass
    except BaseException:
        try:
            channel.send((False, traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if instance is not None:
            instance.close()
        if original_post is not None:
            requests.post=original_post
        channel.close()


class RuntimeClient:
    """Same reset/decide/commit interface, with no in-process module replacement."""
    def __init__(self, runtime, diagnostics, *, environment=None, timeout=600):
        self.receipt = runtime_files(runtime)
        self.timeout = timeout
        self.closed = False
        ctx = multiprocessing.get_context("spawn")
        self.channel, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker,
            args=(child, str(Path(runtime).resolve()), str(Path(diagnostics).resolve()), environment or {}),
            name="carryon-safety-worker")
        self.process.start()
        self.owner=process_identity(self.process.pid) if os.name=="posix" else None
        self.owned_children={};self.stop_watcher=threading.Event();self.watcher=None
        def watch():
            while not self.stop_watcher.wait(.2):self._capture_children()
        if self.owner is not None:
            self.watcher=threading.Thread(target=watch,daemon=True);self.watcher.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self._terminate()
            raise

    def _receive(self):
        if not self.channel.poll(self.timeout):
            raise TimeoutError("Safety worker did not respond within the configured timeout")
        try:
            ok, value = self.channel.recv()
        except EOFError as exc:
            raise RuntimeError("Safety worker exited without a response") from exc
        if not ok:
            raise RuntimeError("Safety worker failed:\n" + value)
        return value

    def _call(self, method, *args):
        if self.closed:
            raise RuntimeError("Safety worker already closed")
        self.channel.send((method, args))
        try:
            return self._receive()
        except BaseException:
            self._terminate()
            raise

    def reset(self, obs):
        return self._call("reset", obs)

    def commit(self, action, obs):
        return self._call("commit", action, obs)

    def decide(self, obs, chunk, step):
        return self._call("decide", obs, chunk, step)

    def state(self):
        return self._call("state")

    def identity(self):
        return {**self._call("identity"), "isolated_worker_pid": self.process.pid,
                "release_runtime": self.receipt}

    def integration_records(self):
        return self._call("integration_records")

    def _capture_children(self):
        for item in owned_processes(self.owner,tuple(self.owned_children.values())):
            self.owned_children[(item["pid"],item["start_ticks"])]=item

    def _terminate(self):
        # Match only this worker's new session and process start times.
        self.stop_watcher.set()
        if self.watcher is not None:self.watcher.join(timeout=2)
        self._capture_children()
        children=list(self.owned_children.values())
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(10)
        for item in children:
            if same_process(item):
                try:os.kill(item["pid"],signal.SIGTERM)
                except ProcessLookupError:pass
        deadline=time.monotonic()+10
        while any(same_process(item) for item in children) and time.monotonic()<deadline:time.sleep(.2)
        for item in children:
            if same_process(item):
                try:os.kill(item["pid"],signal.SIGKILL)
                except ProcessLookupError:pass
        self.channel.close()
        self.closed = True

    def close(self):
        if not self.closed:
            try:
                self._call("close")
                self.process.join(10)
            finally:
                self._terminate()
