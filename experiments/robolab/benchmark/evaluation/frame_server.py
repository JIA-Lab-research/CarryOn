#!/usr/bin/env python3
"""Serve one still frame per station on the URLs the camera config expects.

Why a still frame is not a shortcut here: the RoboLab routing setting is
decision-layer only. Nothing is picked up, nothing is moved, and the episode
ends at the dispatch. A station's camera therefore shows the same thing for the
whole episode, so serving that station's source frame is what a live
shim would have returned -- with the sim booted, the arm idle, and the scene
untouched.

What it does NOT stand in for: anything that moves. The moment a cell involves
manipulation, or a frame has to change within an episode, this server is the
wrong instrument and a real shim is required.

The frames are the reference sources, resized to the sim camera's own
864x480 -- they were rendered at 4x for the reference isolates, and serving them
at 4x would hand the VLM a sharper picture than the real camera ever produces.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRAMES: dict[str, bytes] = {}
HITS: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        name = self.path.strip("/").split("/")[-1].split(".")[0]
        # /cameras/<cam>/latest_raw.jpg -> <cam>
        parts = [p for p in self.path.strip("/").split("/") if p]
        if len(parts) >= 2 and parts[0] == "cameras":
            name = parts[1]
        if name == "_hits":
            body = json.dumps(HITS, sort_keys=True).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        blob = FRAMES.get(name)
        if blob is None:
            self.send_error(404, f"no frame for camera {name!r}")
            return
        HITS[name] = HITS.get(name, 0) + 1
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *a) -> None:   # keep the agent's stdout readable
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--port", type=int, default=18120)
    a = ap.parse_args()

    d = Path(a.frames_dir)
    for p in sorted(d.glob("*.jpg")):
        cam = f"{p.stem}_cam"
        FRAMES[cam] = p.read_bytes()
    if not FRAMES:
        raise SystemExit(f"no *.jpg under {d}")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"serving {len(FRAMES)} cameras on http://127.0.0.1:{a.port}"
          f"/cameras/<name>/latest_raw.jpg", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
