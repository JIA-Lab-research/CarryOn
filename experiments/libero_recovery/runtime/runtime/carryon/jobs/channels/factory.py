"""Construct result channels for the configured transports."""
from __future__ import annotations
import sys
from ...config import Config
from ...endpoints.registry import EndpointRegistry
from ...logging_util import EventLogger
from .webhook import WebhookChannel

def build_result_channel(config: Config, registry: EndpointRegistry, logger: EventLogger | None=None) -> tuple[WebhookChannel, object | None]:
    """Return ``(result_channel, ros_callback_channel)``.

    The webhook channel is always built. The ROS callback channel is built only
    when ROS is in play (an endpoint declares ``transport: ros``, or the config
    forces it) and the result channel is not pinned to ``webhook-only``. On any
    ROS construction failure we warn and fall back to HTTP (returns ``None``),
    so a misconfigured ROS setup never blocks startup.
    """
    result_channel = WebhookChannel(host=config.callback_host, port=config.callback_port, public_url=config.callback_public_url, auth_token=config.callback_auth_token, logger=logger)
    ros_callback_channel = None
    needs_ros_callback = config.callback_ros_enabled or config.result_channel == 'ros' or config.default_transport == 'ros' or any((ep.transport == 'ros' for ep in registry.list()))
    if needs_ros_callback and config.result_channel != 'webhook-only':
        raise ValueError('This experiment runtime accepts simulator HTTP and local endpoints only')
    return (result_channel, ros_callback_channel)
