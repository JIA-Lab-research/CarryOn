"""Provider HTTP layer.

The **Module** that speaks a **Provider**'s wire protocol (today: OpenAI-
compatible chat completions). Both the **Planner VLM** (``carryon.planner``)
and the **VLM Judge** (``carryon.judge``) route their HTTP through it. Distinct
from :mod:`carryon.transports`, which talks to a **RobotEndpoint**, not an LLM
gateway.
"""

from .images import (
    DEFAULT_MAX_IMAGE_EDGE_PX,
    downscale_image_bytes,
    fetch_image_bytes,
    fetch_image_url_part,
)
from .openrouter import (
    OPENROUTER_BASE_URL,
    ChatResult,
    OpenRouterChatClient,
    OpenRouterHTTPError,
)
from .scene_capture import SceneCapture, SceneSnapshot

__all__ = [
    "ChatResult",
    "DEFAULT_MAX_IMAGE_EDGE_PX",
    "OPENROUTER_BASE_URL",
    "OpenRouterChatClient",
    "OpenRouterHTTPError",
    "SceneCapture",
    "SceneSnapshot",
    "downscale_image_bytes",
    "fetch_image_bytes",
    "fetch_image_url_part",
]
