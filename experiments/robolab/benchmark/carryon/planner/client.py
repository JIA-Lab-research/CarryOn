"""Planner VLM model client.

The model **adapter** the :class:`~carryon.planner.planner.Planner` orchestrator
calls. It builds messages (system prompt + live configured-camera snapshots +
response-schema hint) and wraps the returned text in a
:class:`PlannerVLMResponse`; the wire protocol lives in
:mod:`carryon.providers`. Mirrors :mod:`carryon.judge.client` for the
supervisor side.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import httpx

from ..config import Config
from ..errors import PlannerVLMError
from ..providers import (
    DEFAULT_MAX_IMAGE_EDGE_PX,
    OPENROUTER_BASE_URL,
    OpenRouterChatClient,
    OpenRouterHTTPError,
    SceneCapture,
)


# How many camera snapshots one planning turn may attach. The planner picks an
# endpoint and phrases an instruction; a handful of views is enough for that,
# and each extra one is a full image on every turn.
DEFAULT_MAX_IMAGE_ATTACHMENTS = 4


@dataclass
class PlannerVLMResponse:
    text: str
    raw: Any = None
    usage: dict = field(default_factory=dict)


class PlannerVLMClient(Protocol):
    async def plan_chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        response_schema_hint: bool = True,
    ) -> PlannerVLMResponse: ...

    async def aclose(self) -> None: ...


def _response_schema() -> dict[str, Any]:
    return {
        "goal": "string",
        "user_reply": "brief message shown to the user (1-2 sentences)",
        "subtasks": [
            {
                "id": "string",
                "endpoint": "registered endpoint name",
                "instruction": (
                    "required non-empty string: exact skill_library entry, or "
                    "entry with {placeholders} filled; concise free-form only "
                    "if the endpoint has no skill_library"
                ),
                "objective": (
                    "completion criterion for the visual judge; never sent to "
                    "the robot"
                ),
                "inputs": "object",
                "depends_on": ["subtask id"],
            }
        ],
    }


class MockPlannerVLMClient:
    """Mock top-level VLM planner used by local tests."""

    def __init__(
        self,
        *,
        mode: str = "echo",
        responses: list[str] | None = None,
        endpoint_names: Optional[Callable[[], list[str]]] = None,
        cameras: list[dict[str, Any]] | None = None,
        model: str = "mock/vlm-planner",
    ) -> None:
        if mode not in ("scripted", "echo"):
            raise ValueError(f"unknown mock planner VLM mode: {mode!r}")
        self.mode = mode
        self.responses = list(responses or [])
        self._idx = 0
        self._endpoint_names = endpoint_names or (lambda: [])
        self.cameras = list(cameras or [])
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def plan_chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        response_schema_hint: bool = True,
    ) -> PlannerVLMResponse:
        request = {
            "model": self.model,
            "system_prompt": system or "",
            "messages": list(messages),
            "cameras": list(self.cameras),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_schema_hint": response_schema_hint,
        }
        self.calls.append(request)
        text = self._next_scripted() if self.mode == "scripted" else self._echo_plan(messages)
        return PlannerVLMResponse(text=text, raw={"mock_planner_vlm_request": request}, usage={})

    async def aclose(self) -> None:
        pass

    def _next_scripted(self) -> str:
        if not self.responses:
            raise RuntimeError("MockPlannerVLMClient(scripted) has no scripted responses")
        response = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return response

    # Messages the loop injects on the user channel AFTER the user's own text.
    # Echo mode's contract is to echo the USER'S INSTRUCTION, so these are
    # skipped by prefix -- extending the trick already used for prior-results.
    # Without this the mock echoes a control message back as an instruction,
    # and the turn continues on work nobody asked for.
    _INJECTED_USER_PREFIXES = (
        "Prior results in this session",
        "STEPS ALREADY EXECUTED THIS TURN",
        "These step(s) ALREADY EXECUTED in this turn",
    )

    def _last_user_text(self, messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content") or ""
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                if not str(content).startswith(self._INJECTED_USER_PREFIXES):
                    return str(content)
        return "do something"

    def _echo_plan(self, messages: list[dict]) -> str:
        user = self._last_user_text(messages)
        names = self._endpoint_names() or []
        target = names[0] if names else "__no_endpoints_registered__"
        plan = {
            "goal": user[:120],
            "user_reply": f"On it: {user[:80]}",
            "subtasks": [
                {
                    "id": "t1",
                    "endpoint": target,
                    "instruction": user,
                    "inputs": {"planner_vlm_cameras": [c.get("name") for c in self.cameras]},
                    "depends_on": [],
                }
            ],
        }
        reasoning = (
            f"<reasoning>mock planner VLM echo mode: using {len(self.cameras)} camera "
            f"input(s) and routing the user's text to endpoint {target!r}.</reasoning>"
        )
        return f"{reasoning}\n{json.dumps(plan, ensure_ascii=False)}"


class OpenRouterPlannerVLMClient:
    """OpenAI-compatible VLM planner client for OpenRouter.

    This Adapter only knows how to **build messages** (system prompt + current
    camera snapshots + response-schema hint) and how to wrap the returned text
    in :class:`PlannerVLMResponse`. The HTTP protocol lives in
    :class:`carryon.providers.OpenRouterChatClient`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 0.2,
        timeout_s: float = 60.0,
        cameras: list[dict[str, Any]] | None = None,
        http_referer: str | None = None,
        x_title: str | None = None,
        client: httpx.AsyncClient | None = None,
        scene_capture: SceneCapture | None = None,
        max_image_edge_px: int = DEFAULT_MAX_IMAGE_EDGE_PX,
        max_image_attachments: int = DEFAULT_MAX_IMAGE_ATTACHMENTS,
        snapshot_output_dir: str = "",
    ) -> None:
        if not api_key:
            raise PlannerVLMError("planner VLM: api_key is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.default_temperature = temperature
        self.timeout_s = timeout_s
        self.cameras = list(cameras or [])
        self.http_referer = http_referer
        self.x_title = x_title
        # Attachment budget for the camera snapshots this client adds to a
        # planning turn. <=0 disables the respective cap.
        self.max_image_edge_px = int(max_image_edge_px)
        self.max_image_attachments = int(max_image_attachments)
        self.snapshot_output_dir = str(snapshot_output_dir or "")
        self._chat = OpenRouterChatClient(
            api_key=api_key,
            base_url=base_url,
            timeout_s=timeout_s,
            http_referer=http_referer,
            x_title=x_title,
            client=client,
        )
        self._scene_capture = scene_capture or SceneCapture(
            self.cameras,
            http_client=client,
            image_fetch_timeout_s=min(5.0, timeout_s),
        )

    async def plan_chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        response_schema_hint: bool = True,
    ) -> PlannerVLMResponse:
        snapshot_paths: list[str] = []
        try:
            if response_schema_hint:
                messages, snapshot_paths = (
                    await self._attach_current_camera_snapshots_with_paths(messages)
                )
            result = await self._chat.chat(
                model=self.model,
                messages=self._messages(
                    messages,
                    system=system,
                    response_schema_hint=response_schema_hint,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except httpx.HTTPError as e:
            raise PlannerVLMError(f"planner VLM http error: {e}") from e
        except OpenRouterHTTPError as e:
            raise PlannerVLMError(f"planner VLM returned {e.status_code}: {e.body[:300]}") from e
        usage = dict(result.raw.get("usage") or {})
        if snapshot_paths:
            usage["snapshot_paths"] = snapshot_paths
        return PlannerVLMResponse(text=result.text, raw=result.raw, usage=usage)

    async def aclose(self) -> None:
        await self._scene_capture.aclose()
        await self._chat.aclose()

    async def _attach_current_camera_snapshots(
        self, messages: list[dict]
    ) -> list[dict]:
        attached, _ = await self._attach_current_camera_snapshots_with_paths(messages)
        return attached

    async def _attach_current_camera_snapshots_with_paths(
        self, messages: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Attach one current frame per configured camera to the last user turn.

        Reaction prompts already carry before/after image parts. In that case
        those task-specific frames are more useful, and capturing again would
        duplicate image payloads. Camera failures are fail-open: SceneCapture
        drops individual failures and an unexpected capture error leaves the
        original text-only request intact.

        Two caps bound what a planning turn pays for images, because both
        dimensions grow without one: ``max_image_attachments`` bounds the COUNT
        (a rig can register more cameras than the planner needs to choose an
        endpoint, and every extra view is a full image billed on every turn),
        and ``max_image_edge_px`` bounds each frame's SIZE. Cameras are kept in
        configured order, so the cap drops the last-listed views -- list the
        cameras that matter first.
        """
        copied = [dict(message) for message in messages]
        if not self.cameras or _messages_contain_images(copied):
            return copied, []
        try:
            snapshots = await self._scene_capture.capture()
        except Exception:
            return copied, []
        if not snapshots:
            return copied, []
        if self.max_image_attachments > 0:
            snapshots = snapshots[: self.max_image_attachments]

        image_parts = [
            snapshot.to_image_url_part(
                max_edge_px=self.max_image_edge_px or None
            )
            for snapshot in snapshots
        ]
        snapshot_paths = self._persist_snapshot_parts(snapshots, image_parts)

        for index in range(len(copied) - 1, -1, -1):
            if copied[index].get("role") != "user":
                continue
            content = copied[index].get("content") or ""
            if isinstance(content, list):
                parts = [dict(part) if isinstance(part, dict) else part for part in content]
            else:
                parts = [{"type": "text", "text": str(content)}]
            camera_names = ", ".join(snapshot.camera for snapshot in snapshots)
            parts.append(
                {
                    "type": "text",
                    "text": f"Current desktop camera view(s): {camera_names}",
                }
            )
            parts.extend(image_parts)
            copied[index]["content"] = parts
            break
        return copied, snapshot_paths

    def _persist_snapshot_parts(
        self,
        snapshots: list[Any],
        image_parts: list[dict[str, Any]],
    ) -> list[str]:
        """Persist the exact image bytes attached to this planner request."""
        if not self.snapshot_output_dir:
            return []
        try:
            output_dir = Path(self.snapshot_output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return []

        capture_id = uuid.uuid4().hex[:8]
        paths: list[str] = []
        for index, (snapshot, part) in enumerate(zip(snapshots, image_parts), start=1):
            try:
                url = str(((part or {}).get("image_url") or {}).get("url") or "")
                header, encoded = url.split(",", 1)
                if not header.startswith("data:") or ";base64" not in header:
                    continue
                mime = header[5:].split(";", 1)[0].lower()
                extension = {
                    "image/jpeg": "jpg",
                    "image/png": "png",
                    "image/webp": "webp",
                    "image/gif": "gif",
                }.get(mime, "bin")
                captured_at = datetime.fromtimestamp(float(snapshot.ts))
                timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                camera = re.sub(
                    r"[^A-Za-z0-9_.-]+", "_", str(snapshot.camera)
                ).strip("._") or "camera"
                path = output_dir / (
                    f"{timestamp}_{capture_id}_{index:02d}_{camera}.{extension}"
                )
                path.write_bytes(base64.b64decode(encoded, validate=True))
                paths.append(str(path))
            except (OSError, ValueError, TypeError):
                continue
        return paths

    def _messages(
        self,
        messages: list[dict],
        *,
        system: Optional[str],
        response_schema_hint: bool = True,
    ) -> list[dict]:
        out: list[dict] = []
        system_parts = [system or ""]
        if self.cameras:
            system_parts.append(
                "Planner VLM camera inputs available as JSON:\n"
                + json.dumps(self.cameras, ensure_ascii=False)
            )
        if response_schema_hint:
            system_parts.append(
                "Return optional <reasoning>...</reasoning> followed by strict JSON "
                "matching this schema:\n"
                + json.dumps(_response_schema(), ensure_ascii=False)
            )
        system_content = "\n\n".join(part for part in system_parts if part)
        if system_content:
            out.append({"role": "system", "content": system_content})
        out.extend(dict(m) for m in messages)
        return out


def _messages_contain_images(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "image_url"
            for part in content
        ):
            return True
    return False


def build_planner_vlm_client(config: Config) -> PlannerVLMClient:
    if config.planner_vlm_provider == "mock":
        return MockPlannerVLMClient(
            mode=config.planner_vlm_mock_mode,
            responses=config.planner_vlm_mock_responses or [],
            cameras=config.planner_vlm_cameras or [],
            model=config.planner_vlm_model,
        )
    if config.planner_vlm_provider == "openrouter":
        if not config.planner_vlm_api_key:
            raise PlannerVLMError("planner_vlm.api_key is required for provider=openrouter")
        return OpenRouterPlannerVLMClient(
            api_key=config.planner_vlm_api_key,
            model=config.planner_vlm_model,
            base_url=config.planner_vlm_base_url,
            temperature=config.planner_vlm_temperature,
            timeout_s=config.planner_vlm_timeout_s,
            cameras=config.planner_vlm_cameras or [],
            http_referer=config.planner_vlm_http_referer,
            x_title=config.planner_vlm_x_title,
            max_image_edge_px=config.planner_vlm_max_image_edge_px,
            max_image_attachments=config.planner_vlm_max_image_attachments,
            snapshot_output_dir=config.planner_vlm_snapshot_output_dir,
        )
    raise PlannerVLMError(f"unsupported planner_vlm.provider: {config.planner_vlm_provider}")


def resolve_reflection_vlm(config: Config) -> dict:
    """Effective reflection-VLM settings: each ``reflection_vlm.*`` field falls
    back to its ``planner_vlm.*`` counterpart when unset, so a single planner
    config keeps working while a dedicated reflection model can override it."""
    pick = lambda a, b: a if a is not None else b  # noqa: E731
    return {
        "provider": pick(config.reflection_vlm_provider, config.planner_vlm_provider),
        "model": pick(config.reflection_vlm_model, config.planner_vlm_model),
        "temperature": pick(config.reflection_vlm_temperature, config.planner_vlm_temperature),
        "api_key": pick(config.reflection_vlm_api_key, config.planner_vlm_api_key),
        "base_url": pick(config.reflection_vlm_base_url, config.planner_vlm_base_url),
        "timeout_s": pick(config.reflection_vlm_timeout_s, config.planner_vlm_timeout_s),
    }


def build_reflection_vlm_client(config: Config) -> PlannerVLMClient:
    """Build a reflection client with recorded frames supplied per request."""
    r = resolve_reflection_vlm(config)
    if r["provider"] == "mock":
        return MockPlannerVLMClient(
            mode=config.planner_vlm_mock_mode,
            responses=config.planner_vlm_mock_responses or [],
            cameras=[],
            model=r["model"],
        )
    if r["provider"] == "openrouter":
        if not r["api_key"]:
            raise PlannerVLMError(
                "reflection_vlm.api_key (or planner_vlm.api_key fallback) is required for provider=openrouter"
            )
        return OpenRouterPlannerVLMClient(
            api_key=r["api_key"],
            model=r["model"],
            base_url=r["base_url"],
            temperature=r["temperature"],
            timeout_s=r["timeout_s"],
            cameras=[],
            http_referer=config.planner_vlm_http_referer,
            x_title=config.planner_vlm_x_title,
        )
    raise PlannerVLMError(f"unsupported reflection_vlm.provider: {r['provider']}")
