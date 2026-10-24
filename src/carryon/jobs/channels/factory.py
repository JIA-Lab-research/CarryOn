"""HTTP callback construction for the simulation release; ROS is excluded."""

from __future__ import annotations

from ...config import Config
from ...endpoints.registry import EndpointRegistry
from ...logging_util import EventLogger
from .webhook import WebhookChannel


def build_result_channel(
    config: Config,
    registry: EndpointRegistry,
    logger: EventLogger | None = None,
) -> tuple[WebhookChannel, object | None]:
    """Return the HTTP callback channel and an unused compatibility slot."""
    result_channel = WebhookChannel(
        host=config.callback_host,
        port=config.callback_port,
        public_url=config.callback_public_url,
        auth_token=config.callback_auth_token,
        logger=logger,
    )

    if config.callback_ros_enabled or config.result_channel == "ros" or config.default_transport == "ros" or any(ep.transport == "ros" for ep in registry.list()):
        raise ValueError("ROS callbacks are not included in the simulation release")
    return result_channel, None
