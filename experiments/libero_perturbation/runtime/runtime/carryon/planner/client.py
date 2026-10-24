"""Planner VLM model client.

The model **adapter** the :class:`~carryon.planner.planner.Planner` orchestrator
calls. It only knows how to *build messages* (system prompt + camera JSON dump +
response-schema hint) and wrap the returned text in a :class:`PlannerVLMResponse`;
the wire protocol lives in :mod:`carryon.providers`. Mirrors
:mod:`carryon.judge.client` for the supervisor side.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
import httpx
from ..config import Config
from ..errors import PlannerVLMError
from ..providers import OPENROUTER_BASE_URL, OpenRouterChatClient, OpenRouterHTTPError

@dataclass
class PlannerVLMResponse:
    text: str
    raw: Any = None
    usage: dict = field(default_factory=dict)

class PlannerVLMClient(Protocol):

    async def plan_chat(self, messages: list[dict], *, temperature: float=0.2, max_tokens: int=4096, system: Optional[str]=None, response_schema_hint: bool=True) -> PlannerVLMResponse:
        ...

    async def aclose(self) -> None:
        ...

def _response_schema() -> dict[str, Any]:
    return {'goal': 'string', 'subtasks': [{'id': 'string', 'endpoint': 'registered endpoint name', 'instruction': 'free-form natural language', 'inputs': 'object', 'depends_on': ['subtask id']}]}

class MockPlannerVLMClient:
    """Mock top-level VLM planner used by local tests."""

    def __init__(self, *, mode: str='echo', responses: list[str] | None=None, endpoint_names: Optional[Callable[[], list[str]]]=None, cameras: list[dict[str, Any]] | None=None, model: str='mock/vlm-planner') -> None:
        if mode not in ('scripted', 'echo'):
            raise ValueError(f'unknown mock planner VLM mode: {mode!r}')
        self.mode = mode
        self.responses = list(responses or [])
        self._idx = 0
        self._endpoint_names = endpoint_names or (lambda: [])
        self.cameras = list(cameras or [])
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def plan_chat(self, messages: list[dict], *, temperature: float=0.2, max_tokens: int=4096, system: Optional[str]=None, response_schema_hint: bool=True) -> PlannerVLMResponse:
        request = {'model': self.model, 'system_prompt': system or '', 'messages': list(messages), 'cameras': list(self.cameras), 'temperature': temperature, 'max_tokens': max_tokens, 'response_schema_hint': response_schema_hint}
        self.calls.append(request)
        text = self._next_scripted() if self.mode == 'scripted' else self._echo_plan(messages)
        return PlannerVLMResponse(text=text, raw={'mock_planner_vlm_request': request}, usage={})

    async def aclose(self) -> None:
        pass

    def _next_scripted(self) -> str:
        if not self.responses:
            raise RuntimeError('MockPlannerVLMClient(scripted) has no scripted responses')
        response = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return response

    def _last_user_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get('role') == 'user':
                content = message.get('content') or ''
                if isinstance(content, list):
                    content = ''.join((p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text'))
                if not str(content).startswith('Prior results in this session'):
                    return str(content)
        return 'do something'

    def _echo_plan(self, messages: list[dict]) -> str:
        user = self._last_user_text(messages)
        names = self._endpoint_names() or []
        target = names[0] if names else '__no_endpoints_registered__'
        plan = {'goal': user[:120], 'subtasks': [{'id': 't1', 'endpoint': target, 'instruction': user, 'inputs': {'planner_vlm_cameras': [c.get('name') for c in self.cameras]}, 'depends_on': []}]}
        reasoning = f"<reasoning>mock planner VLM echo mode: using {len(self.cameras)} camera input(s) and routing the user's text to endpoint {target!r}.</reasoning>"
        return f'{reasoning}\n{json.dumps(plan, ensure_ascii=False)}'

class OpenRouterPlannerVLMClient:
    """OpenAI-compatible VLM planner client for OpenRouter.

    This Adapter only knows how to **build messages** (system prompt + camera
    JSON dump + response-schema hint) and how to wrap the returned text in
    :class:`PlannerVLMResponse`. The HTTP protocol lives in
    :class:`carryon.providers.OpenRouterChatClient`.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str=OPENROUTER_BASE_URL, temperature: float=0.2, timeout_s: float=60.0, cameras: list[dict[str, Any]] | None=None, http_referer: str | None=None, x_title: str | None=None, client: httpx.AsyncClient | None=None) -> None:
        if not api_key:
            raise PlannerVLMError('planner VLM: api_key is required')
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.default_temperature = temperature
        self.timeout_s = timeout_s
        self.cameras = list(cameras or [])
        self.http_referer = http_referer
        self.x_title = x_title
        self._chat = OpenRouterChatClient(api_key=api_key, base_url=base_url, timeout_s=timeout_s, http_referer=http_referer, x_title=x_title, client=client)

    async def plan_chat(self, messages: list[dict], *, temperature: float=0.2, max_tokens: int=4096, system: Optional[str]=None, response_schema_hint: bool=True) -> PlannerVLMResponse:
        try:
            result = await self._chat.chat(model=self.model, messages=self._messages(messages, system=system, response_schema_hint=response_schema_hint), temperature=temperature, max_tokens=max_tokens)
        except httpx.HTTPError as e:
            raise PlannerVLMError(f'planner VLM http error: {e}') from e
        except OpenRouterHTTPError as e:
            raise PlannerVLMError(f'planner VLM returned {e.status_code}: {e.body[:300]}') from e
        usage = dict(result.raw.get('usage') or {})
        return PlannerVLMResponse(text=result.text, raw=result.raw, usage=usage)

    async def aclose(self) -> None:
        await self._chat.aclose()

    def _messages(self, messages: list[dict], *, system: Optional[str], response_schema_hint: bool=True) -> list[dict]:
        out: list[dict] = []
        system_parts = [system or '']
        if self.cameras:
            system_parts.append('Planner VLM camera inputs available as JSON:\n' + json.dumps(self.cameras, ensure_ascii=False))
        if response_schema_hint:
            system_parts.append('Return optional <reasoning>...</reasoning> followed by strict JSON matching this schema:\n' + json.dumps(_response_schema(), ensure_ascii=False))
        system_content = '\n\n'.join((part for part in system_parts if part))
        if system_content:
            out.append({'role': 'system', 'content': system_content})
        out.extend((dict(m) for m in messages))
        return out

def build_planner_vlm_client(config: Config) -> PlannerVLMClient:
    if config.planner_vlm_provider == 'mock':
        return MockPlannerVLMClient(mode=config.planner_vlm_mock_mode, responses=config.planner_vlm_mock_responses or [], cameras=config.planner_vlm_cameras or [], model=config.planner_vlm_model)
    if config.planner_vlm_provider == 'openrouter':
        if not config.planner_vlm_api_key:
            raise PlannerVLMError('planner_vlm.api_key is required for provider=openrouter')
        return OpenRouterPlannerVLMClient(api_key=config.planner_vlm_api_key, model=config.planner_vlm_model, base_url=config.planner_vlm_base_url, temperature=config.planner_vlm_temperature, timeout_s=config.planner_vlm_timeout_s, cameras=config.planner_vlm_cameras or [], http_referer=config.planner_vlm_http_referer, x_title=config.planner_vlm_x_title)
    raise PlannerVLMError(f'unsupported planner_vlm.provider: {config.planner_vlm_provider}')

def resolve_reflection_vlm(config: Config) -> dict:
    """Effective reflection-VLM settings: each ``reflection_vlm.*`` field falls
    back to its ``planner_vlm.*`` counterpart when unset, so a single planner
    config keeps working while a dedicated reflection model can override it."""
    pick = lambda a, b: a if a is not None else b
    return {'provider': pick(config.reflection_vlm_provider, config.planner_vlm_provider), 'model': pick(config.reflection_vlm_model, config.planner_vlm_model), 'temperature': pick(config.reflection_vlm_temperature, config.planner_vlm_temperature), 'api_key': pick(config.reflection_vlm_api_key, config.planner_vlm_api_key), 'base_url': pick(config.reflection_vlm_base_url, config.planner_vlm_base_url), 'timeout_s': pick(config.reflection_vlm_timeout_s, config.planner_vlm_timeout_s)}

def build_reflection_vlm_client(config: Config) -> PlannerVLMClient:
    """Build a reflection client with recorded frames supplied per request."""
    r = resolve_reflection_vlm(config)
    if r['provider'] == 'mock':
        return MockPlannerVLMClient(mode=config.planner_vlm_mock_mode, responses=config.planner_vlm_mock_responses or [], cameras=[], model=r['model'])
    if r['provider'] == 'openrouter':
        if not r['api_key']:
            raise PlannerVLMError('reflection_vlm.api_key (or planner_vlm.api_key fallback) is required for provider=openrouter')
        return OpenRouterPlannerVLMClient(api_key=r['api_key'], model=r['model'], base_url=r['base_url'], temperature=r['temperature'], timeout_s=r['timeout_s'], cameras=[], http_referer=config.planner_vlm_http_referer, x_title=config.planner_vlm_x_title)
    raise PlannerVLMError(f"unsupported reflection_vlm.provider: {r['provider']}")
