"""Owned local model processes with explicit readiness and bounded cleanup."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import threading
import urllib.request
from .configuration import validate_public_config
from .files import write_json

def process_identity(pid):
    try:
        parts = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")",1)[1].split()
        return {"pid":pid, "state":parts[0], "ppid":int(parts[1]),
                "session":int(parts[3]), "start_ticks":int(parts[19])}
    except (OSError, ValueError, IndexError):
        return None

def descendants(pid):
    found = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            item = process_identity(int(path.name))
            if item:
                found[item["pid"]] = item
    owned = {pid}
    while True:
        more = {p for p,r in found.items() if r["ppid"] in owned}
        if more <= owned:
            break
        owned |= more
    return [found[p] for p in owned if p != pid and p in found]

def same_process(item):
    current = process_identity(item["pid"])
    return (current is not None and current["state"] != "Z"
            and current["start_ticks"] == item["start_ticks"])

def session_members(owner):
    """Track our new session, including children whose parent has already exited."""
    if owner is None:
        return []
    members = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            item = process_identity(int(path.name))
            if (item and item["pid"] != owner["pid"]
                    and item["session"] == owner["pid"]
                    and item["start_ticks"] >= owner["start_ticks"]):
                members.append(item)
    return members


def owned_processes(owner, previous=()):
    """Find only our process tree, including children that create new sessions."""
    if owner is None:
        return []
    found = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            item = process_identity(int(path.name))
            if item is not None:
                found[item["pid"]] = item
    def current(item):
        live = found.get(item["pid"])
        return live is not None and live["start_ticks"] == item["start_ticks"] and live["state"] != "Z"
    roots = {item["pid"] for item in [owner, *previous] if current(item)}
    roots.update(pid for pid, item in found.items() if item["session"] == owner["pid"]
                 and item["start_ticks"] >= owner["start_ticks"])
    while True:
        children = {pid for pid, item in found.items() if item["ppid"] in roots
                    and item["start_ticks"] >= found[item["ppid"]]["start_ticks"]}
        if children <= roots:
            break
        roots |= children
    return [found[pid] for pid in roots if pid != owner["pid"]]


@contextmanager
def local_service(spec, output):
    validate_public_config(spec)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    port = int(spec["port"])
    if not 1 <= port <= 65535:
        raise ValueError("Invalid model service port")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            sock.bind(("127.0.0.1",port))
        except OSError as exc:
            raise RuntimeError(f"Model service port {port} is occupied; existing processes are untouched") from exc
    argv = list(spec["argv"])
    if not argv or not Path(argv[0]).is_file():
        raise ValueError("Service argv[0] must be an existing Python executable")
    env = dict(os.environ)
    env.pop("PYTHONPATH",None)
    env.pop("PYTHONHOME",None)
    for key in ("OPENROUTER_API_KEY","HF_TOKEN","HUGGING_FACE_HUB_TOKEN"):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    env.update({k:str(v) for k,v in spec.get("environment",{}).items()})
    process = None
    owned = {}
    owner = None
    watcher = None
    stop_watcher = threading.Event()
    def capture():
        if os.name != "nt":
            for item in owned_processes(owner,tuple(owned.values())):
                owned[(item["pid"],item["start_ticks"])] = item
    def watch():
        while not stop_watcher.wait(.2):
            capture()
    with (output/"service.log").open("xb") as log:
        try:
            process = subprocess.Popen(argv,cwd=output,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!="nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            owner = process_identity(process.pid)
            if os.name != "nt":
                watcher = threading.Thread(target=watch,daemon=True)
                watcher.start()
            write_json(output/"PROCESS.json",{"pid":process.pid,"argv":argv,
                       "identity":process_identity(process.pid),"port":port})
            deadline = time.monotonic()+float(spec.get("startup_timeout_s",900))
            ready = False
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError("Model service exited during startup; see "+str(output/"service.log"))
                try:
                    if spec.get("model_id"):
                        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                        with opener.open(f"http://127.0.0.1:{port}/v1/models",timeout=2) as response:
                            body = json.load(response)
                        ready = any(row.get("id")==spec["model_id"] for row in body.get("data",[]))
                    else:
                        with socket.create_connection(("127.0.0.1",port),timeout=1):
                            ready = True
                except (OSError,ValueError):
                    pass
                if ready:
                    break
                time.sleep(.5)
            if not ready:
                raise TimeoutError("Model service did not become ready")
            write_json(output/"READY.json",{"port":port,"model_id":spec.get("model_id")})
            yield process
        finally:
            if process is not None:
                stop_watcher.set()
                if watcher is not None:
                    watcher.join(timeout=2)
                capture()
                owned = list(owned.values())
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=20)
                for item in owned:
                    if same_process(item):
                        try:os.kill(item["pid"],signal.SIGTERM)
                        except ProcessLookupError:pass
                deadline = time.monotonic()+10
                while any(same_process(item) for item in owned) and time.monotonic()<deadline:
                    time.sleep(.2)
                for item in owned:
                    if same_process(item):
                        try:os.kill(item["pid"],signal.SIGKILL)
                        except ProcessLookupError:pass
                write_json(output/"CLEANUP.json",{"parent_stopped":process.poll() is not None,
                    "owned_children":owned,"remaining_owned_pids":[r["pid"] for r in owned if same_process(r)]})
