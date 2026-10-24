"""Linux subprocess ownership, including orphaned children of this session."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from ..files import write_json
from ..services import process_identity,owned_processes,same_process

@contextmanager
def owned(argv, *, cwd, env, log, receipt, stdin=False):
    if os.name!="posix": raise RuntimeError("Recovery simulation requires Linux")
    observed={};stop=threading.Event()
    with Path(log).open("xb") as stream:
        process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,text=True,start_new_session=True)
        identity=process_identity(process.pid)
        if identity is None:
            process.wait();raise RuntimeError("Worker exited before identity capture")
        def capture():
            for child in owned_processes(identity,tuple(observed.values())):observed[(child["pid"],child["start_ticks"])]=child
        def watch():
            while not stop.wait(.2):capture()
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        write_json(Path(receipt).with_suffix(".START.json"),{"identity":identity,"argv":argv})
        try:yield process
        finally:
            stop.set();watcher.join(timeout=2);capture()
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            children=list(observed.values());capture();children=list(observed.values())
            for sig,grace in [(signal.SIGTERM,10),(signal.SIGKILL,5)]:
                for child in children:
                    if same_process(child):
                        try:os.kill(child["pid"],sig)
                        except ProcessLookupError:pass
                deadline=time.monotonic()+grace
                while any(same_process(c) for c in children) and time.monotonic()<deadline:time.sleep(.05)
            remaining=[c["pid"] for c in children if same_process(c)]
            write_json(Path(receipt).with_suffix(".CLEANUP.json"),{"parent_returncode":process.poll(),
                "children":children,"remaining_owned_pids":remaining})
            if remaining:raise RuntimeError("Owned subprocesses remain after cleanup")
