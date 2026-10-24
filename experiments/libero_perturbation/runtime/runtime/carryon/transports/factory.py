from __future__ import annotations
from typing import Awaitable, Callable, Optional
from ..config import Config
from ..endpoints.registry import EndpointRegistry
from ..logging_util import EventLogger
from .base import Transport
from .http import HTTPTransport
from .mock import MockTransport

def build_transports(config: Config, logger: Optional[EventLogger]=None, *, registry: Optional[EndpointRegistry]=None, skill_registry: Optional['SkillRegistry']=None, local_on_result: Optional[Callable[[str, dict], Awaitable[None]]]=None) -> dict[str, Transport]:
    """Return a mapping of transport-name -> Transport instance.

    ``logger`` is optional; when supplied every outbound submit / cancel is
    recorded via the shared EventLogger's ``transport_*`` methods.

    The ``ros`` transport is **opt-in**: it is only constructed (and
    therefore only requires ``rclpy``) when the runtime is actually going to
    use it. We treat any of the following as "use ROS":
        * ``config.default_transport == "ros"``
        * any endpoint in ``registry`` declares ``transport: ros``
        * ``config.ros_force_enable`` is true
    This keeps non-ROS users from needing to source a ROS 2 environment.
    """
    out: dict[str, Transport] = {'http': HTTPTransport(default_timeout=config.http_default_timeout, auth_token=config.vla_auth_token, logger=logger), 'mock': MockTransport(logger=logger)}
    if skill_registry is not None and local_on_result is not None:
        from .local import LocalTransport
        out['local'] = LocalTransport(skill_registry=skill_registry, on_result=local_on_result, logger=logger)
    needs_ros = config.default_transport == 'ros' or bool(config.ros_force_enable) or (registry is not None and any((ep.transport == 'ros' for ep in registry.list())))
    if needs_ros:
        raise ValueError('This experiment runtime accepts simulator HTTP and local endpoints only')
    return out
