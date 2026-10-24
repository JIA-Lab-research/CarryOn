"""Pure execution adapter for OpenVLA's native 8-row cache and Shield and response predictor."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np

@dataclass(frozen=True)
class Dispatch:
    action: np.ndarray
    source: str
    cache_index: int | None
    decision: dict[str, Any] | None

class OpenVLAChunkController:
    """Eight-row cache; scan at most five runnable rows; fixed E=3 retreat."""
    def __init__(self, decide: Callable[[np.ndarray], tuple[str, Any, dict[str, Any]]], evade_steps: int = 3):
        if evade_steps != 3: raise ValueError("SHIELD_EVADE_STEPS_MUST_BE_3")
        self._decide = decide; self._evade_steps = evade_steps
        self._cache: list[np.ndarray] = []; self._cursor = 0; self._granted = 0
        self._hold: np.ndarray | None = None; self._hold_left = 0; self._last_tel = None

    @property
    def needs_policy_query(self) -> bool:
        return self._hold_left == 0 and self._cursor >= len(self._cache)

    def install_native_chunk(self, actions: Sequence[Sequence[float]]) -> None:
        arr=np.asarray(actions,dtype=np.float64)
        if arr.shape != (8,7) or not np.isfinite(arr).all(): raise ValueError("OPENVLA_NATIVE_CHUNK_MUST_BE_FINITE_8x7")
        if not self.needs_policy_query: raise RuntimeError("OPENVLA_CACHE_OVERWRITE")
        self._cache=[row.copy() for row in arr]; self._cursor=0; self._granted=0; self._last_tel=None

    def next(self) -> Dispatch | None:
        if self._hold_left:
            self._hold_left -= 1
            return Dispatch(self._hold.copy(),"evade",None,dict(self._last_tel))
        if self.needs_policy_query: return None
        if self._granted == 0:
            scan=np.asarray(self._cache[self._cursor:self._cursor+5],dtype=np.float64)
            mode,evade,tel=self._decide(scan); grant=int(tel.get("grant_steps",-1)); self._last_tel=dict(tel)
            if mode=="pass_through":
                if grant != len(scan): raise RuntimeError("OPENVLA_PASS_GRANT_MISMATCH")
                self._granted=grant
            elif mode=="trim_grant":
                if grant < 1 or grant >= len(scan): raise RuntimeError("OPENVLA_TRIM_GRANT_MISMATCH")
                self._granted=grant
            elif mode=="evade":
                out=np.asarray(evade,dtype=np.float64)
                if out.shape != (7,): raise RuntimeError("OPENVLA_EVADE_SHAPE")
                self._cache=[]; self._cursor=0; self._hold=out.copy(); self._hold_left=self._evade_steps-1
                return Dispatch(out.copy(),"evade",None,dict(tel))
            elif mode=="fail_closed":
                self._cache=[]; self._cursor=0; return None
            else: raise RuntimeError("OPENVLA_UNKNOWN_SHIELD_MODE:"+str(mode))
        idx=self._cursor; self._cursor += 1; self._granted -= 1
        return Dispatch(self._cache[idx].copy(),"nominal",idx,self._last_tel)
