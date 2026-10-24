"""Owned local frame services for RoboLab."""
from pathlib import Path
import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RoboLab = Path(__file__).resolve().parent
_SERVERS = {}



def _stop(port):
    proc = _SERVERS.pop(port, None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

def stop_owned():
    for port in list(_SERVERS):
        _stop(port)

atexit.register(stop_owned)

def serve_cell_owned(cell, log, port):
    frames = (RoboLab / "frame_library" / cell).resolve()
    if not frames.is_relative_to((RoboLab / "frame_library").resolve()) or not frames.is_dir():
        raise ValueError("Unknown frame cell")
    _stop(port)
    # A busy port is an error. Never kill another process or read its frames.
    with socket.socket() as check:
        # Match the HTTP server's POSIX reuse policy: a stopped server can leave
        # TIME_WAIT sockets even after its process exits. Active listeners still
        # fail this bind. Windows keeps the exclusive check (reuse differs there).
        if os.name != "nt":
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(("127.0.0.1", port))
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        proc = subprocess.Popen([sys.executable, str(RoboLab / "frame_server.py"), "--frames-dir", str(frames), "--port", str(port)], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    _SERVERS[port] = proc
    expected = (frames / "garage.jpg").read_bytes()
    try:
        for _ in range(20):
            if proc.poll() is not None:
                raise RuntimeError("Frame service exited before readiness")
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/cameras/garage_cam/latest_raw.jpg", timeout=3) as response:
                    received = response.read()
                if received != expected:
                    raise RuntimeError("Frame identity mismatch")
                return None
            except OSError:
                continue
        raise RuntimeError("Frame service readiness timeout")
    except BaseException:
        _stop(port)
        raise
