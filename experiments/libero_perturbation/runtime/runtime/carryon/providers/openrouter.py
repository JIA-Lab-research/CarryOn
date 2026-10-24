"""OpenAI-compatible chat-completions client for OpenRouter.

This **Module** owns the wire protocol — URL, headers, request body, response
shape — for both :class:`carryon.planner.OpenRouterPlannerVLMClient` and
:class:`carryon.judge.OpenRouterVLMJudgeClient`. Callers contribute:

- which model to call,
- how to build the `messages` list (including any pre-fetched image_url parts),
- whether to request strict JSON via ``response_format``,
- how to interpret the returned text.

The seam is :meth:`OpenRouterChatClient.chat`. Everything above that lives in
the Adapter; everything below it is shared here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import httpx
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

class OpenRouterHTTPError(Exception):
    """Non-2xx response from OpenRouter.

    Adapters decide whether to re-raise (Planner) or convert into a
    ``failed`` JudgeDecision (Judge); the Module itself takes no policy.
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f'OpenRouter returned {status_code}: {body[:300]}')
        self.status_code = status_code
        self.body = body

@dataclass
class ChatResult:
    """Result of a single chat-completions call.

    ``text`` is the extracted content from ``choices[0].message.content``
    (both ``str`` and ``list[{type:text, text:...}]`` shapes are flattened).
    ``raw`` is the full parsed response — used by Adapters for ``usage`` and
    for the planner log, but Adapters MUST NOT re-extract ``content`` from
    it (that's the Module's job).
    """
    text: str
    raw: dict = field(default_factory=dict)

class OpenRouterChatClient:

    def __init__(self, *, api_key: str, base_url: str=OPENROUTER_BASE_URL, timeout_s: float=60.0, http_referer: str | None=None, x_title: str | None=None, default_headers: dict[str, str] | None=None, client: httpx.AsyncClient | None=None) -> None:
        if not api_key:
            raise ValueError('OpenRouter: api_key is required')
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.http_referer = http_referer
        self.x_title = x_title
        self.default_headers = dict(default_headers or {})
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def chat(self, *, model: str, messages: list[dict], temperature: float=0.2, max_tokens: int=4096, response_format: dict | None=None) -> ChatResult:
        body: dict[str, Any] = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
        if response_format is not None:
            body['response_format'] = response_format
        response = await self._client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body, headers=self._headers(), timeout=self.timeout_s)
        if response.status_code // 100 != 2:
            raise OpenRouterHTTPError(response.status_code, response.text)
        try:
            data = response.json()
        except Exception as e:
            raise OpenRouterHTTPError(response.status_code, f'invalid JSON: {e}') from e
        text = _extract_content_text(data)
        if text is None:
            raise OpenRouterHTTPError(response.status_code, f'unexpected response shape: {data!r}')
        return ChatResult(text=text, raw=data)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        headers.update(self.default_headers)
        if self.http_referer:
            headers['HTTP-Referer'] = self.http_referer
        if self.x_title:
            headers['X-Title'] = self.x_title
        return headers

def _extract_content_text(data: Any) -> Optional[str]:
    """Pull ``choices[0].message.content`` out of an OpenAI chat-completions
    response. Handles both the plain-string and list-of-parts shapes.
    """
    if not isinstance(data, dict):
        return None
    choices = data.get('choices')
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get('message')
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get('text', '') for p in content if isinstance(p, dict) and isinstance(p.get('text'), str)]
        if parts:
            return ''.join(parts)
    return None
__all__ = ['ChatResult', 'OPENROUTER_BASE_URL', 'OpenRouterChatClient', 'OpenRouterHTTPError']
