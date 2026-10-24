"""openpi pi0.5 LIBERO client implementing the starVLA ModelClient interface.

The state has eight values: end-effector position (3), axis-angle rotation (3),
and gripper joint positions (2). Callers supply the image orientation; this
client applies resize_with_pad. Actions have seven values: translation (3),
rotation (3), and gripper command (+1 close, -1 open).

step() returns the unchanged continuous action and a sign-consistent raw_action
representation. Invalid state or action dimensions raise an error.
PI05_REPLAN defaults to 5 and PI05_RESIZE defaults to 224."""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

_STATE_DIM = 8
_ACTION_DIM = 7


class Pi05ModelClient:
    """starVLA-ModelClient-shaped façade over an openpi pi0.5 server."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 6699,
        unnorm_key: Optional[str] = None,
        replan: Optional[int] = None,
        resize: Optional[int] = None,
        image_size: Optional[Sequence[int]] = None,
        **_ignored: Any,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.replan = int(os.environ.get("PI05_REPLAN", "5")) if replan is None else int(replan)
        if resize is None:
            resize = int(os.environ.get("PI05_RESIZE", "224"))
        if image_size is not None:
            resize = int(image_size[0])
        self.resize = int(resize)
        # Accepted for signature parity with starVLA's ModelClient. openpi
        # servers ship their own norm stats and unnormalize server-side, so a
        # caller-supplied key has nothing to select here; a non-None value would
        # mean the caller thinks it is steering normalization when it is not.
        if unnorm_key is not None:
            raise ValueError(
                f"pi05 server unnormalizes server-side; unnorm_key={unnorm_key!r} "
                "would be silently ignored"
            )

        self._plan: deque = deque()
        self._task: Optional[str] = None
        self._last_step: Optional[int] = None
        self.infer_calls = 0
        self.steps_served = 0
        print(
            f"*** pi05 client: {host}:{port} replan={self.replan} "
            f"resize={self.resize} ***",
            flush=True,
        )

    # -- lifecycle ------------------------------------------------------- #

    def reset(self, task_description: Optional[str] = None) -> None:
        self._task = task_description
        self._plan.clear()

    # -- inference ------------------------------------------------------- #

    def _pad(self, img) -> np.ndarray:
        arr = np.asarray(img)
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(arr, self.resize, self.resize)
        )

    def _infer(self, example: dict, lang: Optional[str]) -> list:
        images = example.get("image") or []
        if len(images) < 2:
            raise RuntimeError(
                f"pi05 adapter needs [agentview, wrist]; got {len(images)} image(s)"
            )
        state = example.get("state")
        if state is None:
            raise RuntimeError(
                "pi05 adapter needs example['state'] (8-D eef_pos + axis-angle + "
                "gripper_qpos); the caller was not patched to forward it"
            )
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != _STATE_DIM:
            raise RuntimeError(f"expected {_STATE_DIM}-D state, got {state.size}-D")

        element = {
            "observation/image": self._pad(images[0]),
            "observation/wrist_image": self._pad(images[1]),
            "observation/state": state,
            "prompt": str(lang),
        }
        chunk = np.asarray(self.client.infer(element)["actions"])
        self.infer_calls += 1
        if chunk.ndim != 2 or chunk.shape[1] != _ACTION_DIM:
            raise RuntimeError(
                f"pi05 returned action chunk {chunk.shape}; "
                f"LIBERO expects (H, {_ACTION_DIM})"
            )
        return list(chunk[: self.replan])

    def step(self, example: dict, step: int = 0, **_kwargs: Any) -> dict:
        lang = example.get("lang")
        # Reset cached actions when the instruction changes or the step counter resets.
        new_episode = (
            lang != self._task
            or step == 0
            or (self._last_step is not None and step < self._last_step)
        )
        if new_episode:
            self.reset(lang)
        self._last_step = step

        if not self._plan:
            self._plan.extend(self._infer(example, lang))

        action = np.asarray(self._plan.popleft(), dtype=np.float32).reshape(-1)
        if action.size != _ACTION_DIM:
            raise RuntimeError(
                f"pi05 action is {action.size}-D; LIBERO expects {_ACTION_DIM}-D"
            )
        self.steps_served += 1
        grip = float(action[6])
        return {
            # Preferred path: untouched policy output.
            "action": action,
            # Sign-based gripper representation for the raw_action interface.
            "raw_action": {
                "world_vector": action[:3],
                "rotation_delta": action[3:6],
                "open_gripper": np.asarray(
                    [1.0 if grip < 0 else 0.0], dtype=np.float32
                ),
            },
        }
