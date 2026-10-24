from __future__ import annotations
from typing import Callable
from ..errors import EndpointNotFound
from .model import RobotEndpoint

class EndpointRegistry:

    def __init__(self) -> None:
        self._by_name: dict[str, RobotEndpoint] = {}
        self._listeners: list[Callable[[], None]] = []

    def register(self, ep: RobotEndpoint) -> None:
        if ep.name in self._by_name:
            raise ValueError(f'endpoint already registered: {ep.name}')
        self._by_name[ep.name] = ep
        self._notify()

    def unregister(self, name: str) -> None:
        if name not in self._by_name:
            raise KeyError(name)
        del self._by_name[name]
        self._notify()

    def get(self, name: str) -> RobotEndpoint:
        ep = self._by_name.get(name)
        if ep is None:
            raise EndpointNotFound(name)
        return ep

    def list(self) -> list[RobotEndpoint]:
        return list(self._by_name.values())

    def snapshot_for_planner_vlm(self) -> list[dict]:
        return [ep.for_planner_vlm() for ep in self._by_name.values()]

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def on_change(self, cb: Callable[[], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass
