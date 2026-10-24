from __future__ import annotations

from typing import Awaitable, Callable, Optional

from ..config import Config
from ..endpoints.registry import EndpointRegistry
from ..logging_util import EventLogger
from .base import Transport
from .http import HTTPTransport
from .mock import MockTransport


def build_transports(
    config: Config,
    logger: Optional[EventLogger] = None,
    *,
    registry: Optional[EndpointRegistry] = None,
    allow_simulation: bool = True,
    skill_registry: Optional["SkillRegistry"] = None,  # noqa: F821
    local_on_result: Optional[Callable[[str, dict], Awaitable[None]]] = None,
) -> dict[str, Transport]:
    """Return a mapping of transport-name -> Transport instance.

    ``logger`` is optional; when supplied every outbound submit / cancel is
    recorded via the shared EventLogger's ``transport_*`` methods.

    HTTP simulator dispatch is included only with allow_simulation. ROS
    requests are rejected. Local and mock endpoints remain available.
    """
    # Omitting HTTP also prevents later registration from bypassing CLI opt-in.
    out: dict[str, Transport] = {"mock": MockTransport(logger=logger)}
    if allow_simulation:
        out["http"] = HTTPTransport(
            default_timeout=config.http_default_timeout,
            auth_token=config.vla_auth_token,
            logger=logger,
        )

    if skill_registry is not None and local_on_result is not None:
        from .local import LocalTransport

        out["local"] = LocalTransport(
            skill_registry=skill_registry,
            on_result=local_on_result,
            logger=logger,
        )

    if config.default_transport == "ros" or config.ros_force_enable or (
        registry is not None and any(ep.transport == "ros" for ep in registry.list())
    ):
        raise ValueError("ROS is not included in the simulation release")

    return out
