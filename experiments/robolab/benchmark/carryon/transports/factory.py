"""Local callback boundary for RoboLab mock endpoints."""
from .mock import MockTransport
from .local import LocalTransport

def build_transports(config, logger=None, *, registry=None, skill_registry=None, local_on_result=None):
    if config.default_transport not in ("mock", "local") or config.ros_force_enable or (registry and any(e.transport not in ("mock", "local") for e in registry.list())):
        raise ValueError("RoboLab permits only mock/local transports")
    out = {"mock": MockTransport(logger=logger)}
    if skill_registry is not None and local_on_result is not None:
        out["local"] = LocalTransport(skill_registry=skill_registry, on_result=local_on_result, logger=logger)
    return out
