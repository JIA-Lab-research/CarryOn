"""VLM helper that describes a location and judges whether a target is present.

Wraps :class:`carryon.providers.OpenRouterChatClient` with a fixed prompt that
asks the model to (a) enumerate the visible objects and (b) decide whether the
robot's *target* is present. The target can be specified by free text
(``target_text="绿色的杯子"``) and/or a *reference image* (the user's uploaded
photo of their cup), which is passed to the VLM alongside the live camera
frames so it can match by appearance, not just object name.

Returns a structured :class:`SceneDescription` carrying ``objects`` /
``description`` (for ``update_scene`` to persist) plus the match verdict
``found`` / ``matched_object`` / ``confidence``.

Implementation notes:

- For ``provider=mock``, returns a deterministic description and computes
  ``found`` by substring-matching the target against the visible objects, so
  tests / no-API setups exercise the full search loop.
- For ``provider=openrouter``, uses ``response_format=json_object`` to force
  JSON, mirroring the VLM Judge contract.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
import httpx
from ..providers import OPENROUTER_BASE_URL, OpenRouterChatClient, OpenRouterHTTPError, SceneSnapshot
_PROMPT = 'You are a vision module for a mobile robot looking at one location (a shelf, a drawer, a table, etc.).\nYou are given the live camera frame(s) and, optionally, a REFERENCE IMAGE of the specific object the robot is searching for.\nReturn STRICT JSON ONLY with this shape:\n{"objects": ["<lowercase singular noun>", ...], "description": "<one short sentence>", "found": <true|false>, "matched_object": "<the object that matches the target, or empty>", "confidence": <number in [0, 1]>}\nRules:\n- objects MUST be a list of lowercase singular nouns ("cup", not "two cups").\n- Set found=true ONLY if the target object (per the text and/or reference image) is clearly visible at this location.\n- Match on appearance (color, shape) when a reference image is provided; do not rely on object name alone.\n- If you cannot identify anything, return an empty objects list, found=false, and describe what you see.\n- Do not invent items that are not clearly visible.'

@dataclass
class SceneDescription:
    objects: list[str] = field(default_factory=list)
    description: str = ''
    found: bool = False
    matched_object: str = ''
    confidence: float = 0.0
    raw: Any = None

class SceneDescriber(Protocol):

    async def describe(self, snapshots: list[SceneSnapshot], *, target_text: str | None=None, reference_image: SceneSnapshot | None=None, location_name: str | None=None, focus_object: str | None=None, container_name: str | None=None) -> SceneDescription:
        ...

    async def aclose(self) -> None:
        ...

class MockSceneDescriber:
    """Deterministic stand-in for tests / no-API setups."""

    def __init__(self, *, objects: list[str] | None=None, description: str='', locations: dict[str, dict[str, Any]] | None=None) -> None:
        self.objects = list(objects or [])
        self.description = description
        self.locations = dict(locations or {})
        self.calls: list[dict] = []

    async def describe(self, snapshots: list[SceneSnapshot], *, target_text: str | None=None, reference_image: SceneSnapshot | None=None, location_name: str | None=None, focus_object: str | None=None, container_name: str | None=None) -> SceneDescription:
        loc = location_name or container_name
        self.calls.append({'snapshots': len(snapshots), 'target_text': target_text, 'has_reference_image': reference_image is not None, 'focus_object': focus_object, 'location_name': loc})
        fixture = self.locations.get(str(loc or ''))
        if isinstance(fixture, dict):
            raw_objects = fixture.get('objects', self.objects)
            desc = str(fixture.get('description') or self.description or '')
        else:
            raw_objects = self.objects
            desc = self.description
        objs = [str(obj).strip().lower() for obj in raw_objects if str(obj).strip()]
        target = (focus_object or target_text or '').strip().lower()
        matched = ''
        if target:
            for obj in objs:
                if target in obj or obj in target:
                    matched = obj
                    break
        return SceneDescription(objects=objs, description=desc or f"mock describe of {loc or 'location'} ({len(snapshots)} frame(s))", found=bool(matched), matched_object=matched, confidence=1.0 if matched else 0.0, raw={'mock': True})

    async def aclose(self) -> None:
        return None

class OpenRouterSceneDescriber:
    """VLM-backed describer/judge using the same OpenAI-compatible client family
    as Planner / Judge.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str=OPENROUTER_BASE_URL, temperature: float=0.0, max_tokens: int=512, timeout_s: float=30.0, http_referer: str | None=None, x_title: str | None=None, client: httpx.AsyncClient | None=None) -> None:
        if not api_key:
            raise ValueError('scene describer: api_key is required for openrouter')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._chat = OpenRouterChatClient(api_key=api_key, base_url=base_url, timeout_s=timeout_s, http_referer=http_referer, x_title=x_title, client=client)

    async def describe(self, snapshots: list[SceneSnapshot], *, target_text: str | None=None, reference_image: SceneSnapshot | None=None, location_name: str | None=None, focus_object: str | None=None, container_name: str | None=None) -> SceneDescription:
        loc = location_name or container_name
        target = target_text or focus_object
        context_parts: list[dict] = [{'type': 'text', 'text': _build_user_prompt(target, loc)}]
        if reference_image is not None:
            context_parts.append({'type': 'text', 'text': 'REFERENCE IMAGE of the target object:'})
            context_parts.append(reference_image.to_image_url_part())
        for s in snapshots:
            context_parts.append({'type': 'text', 'text': f'Camera {s.camera}:'})
            context_parts.append(s.to_image_url_part())
        if not snapshots:
            context_parts.append({'type': 'text', 'text': '(no camera snapshots available; reply found=false with empty objects)'})
        messages = [{'role': 'system', 'content': _PROMPT}, {'role': 'user', 'content': context_parts}]
        try:
            result = await self._chat.chat(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, response_format={'type': 'json_object'})
        except (httpx.HTTPError, OpenRouterHTTPError) as e:
            return SceneDescription(objects=[], description=f'vlm error: {e}', raw=None)
        try:
            payload = json.loads(result.text)
        except Exception:
            return SceneDescription(objects=[], description=result.text[:200], raw=result.raw)
        return _coerce_description(payload, raw=result.raw)

    async def aclose(self) -> None:
        await self._chat.aclose()

def _build_user_prompt(target: str | None, location_name: str | None) -> str:
    parts: list[str] = []
    if location_name:
        parts.append(f'You are looking at {location_name}.')
    else:
        parts.append('You are looking at one location.')
    if target:
        parts.append(f'The robot is searching for: {target!s}. Decide whether it is present here (found=true/false) and which visible object matches.')
    parts.append('List every object you can see. Reply with STRICT JSON only.')
    return ' '.join(parts)

def _coerce_description(payload: Any, *, raw: Any=None) -> SceneDescription:
    if not isinstance(payload, dict):
        return SceneDescription(objects=[], description=str(payload)[:200], raw=raw)
    objs_raw = payload.get('objects')
    if isinstance(objs_raw, list):
        objs = [str(o).strip().lower() for o in objs_raw if str(o).strip()]
    else:
        objs = []
    desc = str(payload.get('description') or '').strip()
    found = bool(payload.get('found'))
    matched = str(payload.get('matched_object') or '').strip()
    try:
        confidence = float(payload.get('confidence') or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return SceneDescription(objects=objs, description=desc, found=found, matched_object=matched, confidence=max(0.0, min(1.0, confidence)), raw=raw)
