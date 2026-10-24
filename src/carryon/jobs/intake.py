"""The single funnel every result flows through before reaching JobManager.

Every inbound callback -- HTTP webhook, ROS topic, LocalTransport, MockTransport
auto-callback, and the VLM Judge's terminal decisions -- routes through one
:class:`ResultRouter`. It logs the callback, lets in-flight skill sub-requests
resume (``EndpointInvoker``), and finally hands the result to
``JobManager.on_result``.

Scene memory is written **by the Planner** (via the ``update_scene`` skill),
not here: the search skills only return observations, so this router no longer
sniffs result payloads to auto-persist them.

This used to be four interleaved closures in ``main._amain`` plus a ``[None]``
slot hack: ``build_transports`` needs an intake callback *before* the invoker /
JobManager (which the intake depends on) exist. Expressing the intake as an
object turns that construction-order cycle into ordinary late binding -- the
host constructs the router early with what's available, passes
``router.local_intake`` into ``build_transports``, then assigns ``router.jm`` /
``router.invoker`` once those are built. The methods only dereference those
collaborators at call time, which is long after ``agent.run()`` starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..judge.routing import swallowed_executor_terminal_payload
from ..types import OnResult

if TYPE_CHECKING:
    from ..config import Config
    from ..endpoints.registry import EndpointRegistry
    from ..logging_util import EventLogger
    from ..memory.scene import SceneMemory
    from ..skills import EndpointInvoker
    from .manager import JobManager


class ResultRouter:
    """Routes every transport/skill/judge result into ``JobManager``.

    Constructed with the collaborators available up-front; ``jm`` and
    ``invoker`` are **late-bound** (assigned by the host after they are built,
    before the agent loop runs) so the router can be handed to
    ``build_transports`` while breaking the transports<->invoker cycle.
    """

    def __init__(
        self,
        *,
        tx_logger: "EventLogger",
        registry: "EndpointRegistry",
        config: "Config",
        scene_memory: "SceneMemory | None",
    ) -> None:
        self.tx_logger = tx_logger
        self.registry = registry
        self.config = config
        self.scene_memory = scene_memory
        # Late-bound (see class docstring): set by the host once built.
        self.jm: "JobManager | None" = None
        self.invoker: "EndpointInvoker | None" = None

    async def on_result(self, job_key: str, payload: dict) -> None:
        """The final intake: log, resume skills, route observations, finalize."""
        try:
            self.tx_logger.transport_callback(job_key=job_key, payload=payload or {})
        except Exception:
            pass
        # Skill-issued sub-requests get their callback routed back into the
        # EndpointInvoker future so an in-process skill can resume.
        if self.invoker is not None and self.invoker.has_pending(job_key):
            await self.invoker.feed_result(job_key, payload or {})
            return
        await self.jm.on_result(job_key, payload)

    async def executor_intake(self, job_key: str, payload: dict) -> None:
        """Webhook / ROS path: optionally defer terminal decisions to VLM judge.

        Looks up the owning endpoint (via JobManager) so the per-tag
        ``vlm_judge.applies_to_tags`` whitelist can be honoured: endpoints
        not in scope keep their native executor-terminal semantics.
        """
        endpoint_tags: list[str] = []
        try:
            rec = self.jm.get(job_key)
            if rec is not None:
                ep = self.registry.get(rec.endpoint_name)
                endpoint_tags = list(ep.tags)
        except Exception:
            # Endpoint may have been removed mid-flight; fall back to
            # "no tags", which keeps Judge in charge iff applies_to_tags
            # is empty (legacy default).
            endpoint_tags = []
        replaced = swallowed_executor_terminal_payload(
            vlm_judge_enabled=bool(self.config.vlm_judge_enabled),
            ignore_executor_terminal=bool(self.config.vlm_judge_ignore_executor_terminal),
            payload=payload or {},
            endpoint_tags=endpoint_tags,
            applies_to_tags=self.config.vlm_judge_applies_to_tags,
        )
        if replaced is not None:
            await self.on_result(job_key, replaced)
            return
        await self.on_result(job_key, payload or {})

    async def local_intake(self, job_key: str, payload: dict) -> None:
        """LocalTransport intake -- just the final funnel.

        Safe to hand to ``build_transports`` before ``on_result``'s collaborators
        exist: it dereferences them only at call time.
        """
        await self.on_result(job_key, payload)

    def default_intake(self, judge_swallows_executor_terminal: bool) -> OnResult:
        """The intake external callbacks (webhook/ROS) should use.

        When the VLM judge owns the terminal decision, route through
        :meth:`executor_intake` (which may downgrade executor terminals);
        otherwise straight to :meth:`on_result`.
        """
        return self.executor_intake         if judge_swallows_executor_terminal else self.on_result
