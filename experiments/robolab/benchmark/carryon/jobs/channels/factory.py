"""Connect local RoboLab callbacks to result intake."""
from .webhook import WebhookChannel

def build_result_channel(config, registry, logger=None):
    if config.callback_ros_enabled or config.result_channel == "ros" or any(e.transport not in ("mock", "local") for e in registry.list()):
        raise ValueError("RoboLab permits only mock/local endpoints")
    return WebhookChannel(host=config.callback_host, port=config.callback_port, public_url=config.callback_public_url, auth_token=config.callback_auth_token, logger=logger), None
