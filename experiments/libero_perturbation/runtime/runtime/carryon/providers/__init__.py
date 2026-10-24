"""Provider HTTP layer.

The **Module** that speaks a **Provider**'s wire protocol (today: OpenAI-
compatible chat completions). Both the **Planner VLM** (``carryon.planner``)
and the **VLM Judge** (``carryon.judge``) route their HTTP through it. Distinct
from :mod:`carryon.transports`, which talks to a **RobotEndpoint**, not an LLM
gateway.
"""
from .images import fetch_image_bytes, fetch_image_url_part, fetch_image_url_part_with_metadata
from .openrouter import OPENROUTER_BASE_URL, ChatResult, OpenRouterChatClient, OpenRouterHTTPError
from .scene_capture import SceneCapture, SceneSnapshot
__all__ = ['ChatResult', 'OPENROUTER_BASE_URL', 'OpenRouterChatClient', 'OpenRouterHTTPError', 'SceneCapture', 'SceneSnapshot', 'fetch_image_bytes', 'fetch_image_url_part', 'fetch_image_url_part_with_metadata']
