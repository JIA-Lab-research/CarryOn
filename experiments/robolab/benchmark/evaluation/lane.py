"""Isolate frame ports, callback ports, logs and memory for concurrent runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_PORT = 18121
BASE_CALLBACK = 9000


@dataclass
class Lane:
    lane: "int | None"
    config: Path
    mem: Path
    tlog: Path
    plog: Path
    port: int
    callback_port: int
    substitutions: list = field(default_factory=list)

    def describe(self) -> dict:
        return {"lane": self.lane, "port": self.port,
                "callback_port": self.callback_port,
                "config": str(self.config), "memory_dir": str(self.mem),
                "transport_log": str(self.tlog), "planner_log": str(self.plog),
                "substitutions": self.substitutions}


def setup(lane: "int | None", base_cfg: Path) -> Lane:
    base_cfg = Path(base_cfg)
    base_txt = base_cfg.read_text(encoding="utf-8")
    if lane is None:
        return Lane(None, base_cfg, REPO / "logs" / "robolab" / "memory",
                    REPO / "logs" / "robolab" / "transport.log",
                    REPO / "logs" / "robolab" / "planner.log",
                    BASE_PORT, BASE_CALLBACK, [])
    if lane < 1:
        raise SystemExit("lane must be >= 1")
    port = BASE_PORT + lane
    root = REPO / "logs" / "robolab" / ("lane%d" % lane)
    root.mkdir(parents=True, exist_ok=True)
    subs = [("127.0.0.1:%d" % BASE_PORT, "127.0.0.1:%d" % port, 4),
            ('"./logs/robolab/memory"', '"./logs/robolab/lane%d/memory"' % lane, 1),
            ('"./logs/robolab/planner.log"', '"./logs/robolab/lane%d/planner.log"' % lane, 1),
            ('"./logs/robolab/transport.log"', '"./logs/robolab/lane%d/transport.log"' % lane, 1),
            ('"./logs/robolab/planner_snapshots"', '"./logs/robolab/lane%d/planner_snapshots"' % lane, 1)]
    txt = base_txt
    for old, new, want in subs:
        got = txt.count(old)
        if got != want:
            raise SystemExit("lane config: expected %d x %r in base, found %d"
                             % (want, old, got))
        txt = txt.replace(old, new)
    # Each lane needs its own callback webhook port (see module docstring).
    if txt.startswith("callback:") or ("\ncallback:" in txt):
        raise SystemExit("lane config: base already sets a callback block; merge needed")
    cb = BASE_CALLBACK + lane
    if not txt.endswith("\n"):
        txt += "\n"
    txt += ('callback:\n  port: %d\n  public_url: "http://127.0.0.1:%d"\n' % (cb, cb))
    subs = subs + [("callback.port", str(cb), 1)]
    cfg = root / ("planner_lane%d.yaml" % lane)
    cfg.write_text(txt, encoding="utf-8")
    return Lane(lane, cfg, root / "memory", root / "transport.log",
                root / "planner.log", port, cb,
                [{"from": o, "to": n, "count": w} for o, n, w in subs])
