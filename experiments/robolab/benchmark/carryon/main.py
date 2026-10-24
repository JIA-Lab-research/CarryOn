"""CLI entry point: ``python -m carryon [--config path]``"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .agent.commands import CommandHandler
from .agent.loop import AgentLoop
from .agent.notifier import Notifier
from .config import Config
from .endpoints import load_endpoints_yaml
from .executor.executor import Executor
from .jobs.channels.factory import build_result_channel
from .jobs.intake import ResultRouter
from .jobs.manager import JobManager
from .memory.factory import build_memory
from .planner.factory import build_planner_stack
from .providers import SceneCapture
from .skills import build_endpoint_skill_stack, build_skill_registry
from .transports.factory import build_transports


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="carryon", description="High-level robot dispatch agent.")
    p.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config YAML (default: $CARRYON_CONFIG or ./config.yaml).",
    )
    return p.parse_args(argv)


async def _amain(config_path: str | None) -> int:
    try:
        config = Config.from_file(config_path)
    except FileNotFoundError as e:
        sys.stderr.write(f"[config error] {e}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"[config error] {type(e).__name__}: {e}\n")
        return 2

    # infra
    registry = load_endpoints_yaml(config.endpoints_config)

    # Fail fast on camera-scoping typos before any service starts: endpoint
    # ``cameras:`` (judge scoping) names must exist in the configured camera
    # set. The judge layer fails open on unknown names at runtime, so a typo
    # must never survive startup; valid configs behave exactly as before.
    from .camera_scoping import validate_camera_scoping

    _judge_cams = None
    if config.vlm_judge_enabled:
        raise ValueError("RoboLab evaluation requires recovery and Judge disabled")
    try:
        validate_camera_scoping(
            judge_cameras=_judge_cams,
            endpoints=registry.list(),
        )
    except ValueError as e:
        sys.stderr.write(f"[config error] {e}\n")
        return 2

    # Transport I/O tracing (JSONL + stdout one-liner per outbound/inbound call)
    from .logging_util import EventLogger
    tx_logger = EventLogger(
        path=config.transport_log_path,
        stdout=bool(config.verbose and config.transport_log_stdout),
        channel="transport",
    )

    # Three-tier memory (short-term / scene / user); tiers gated by config.
    _mem = build_memory(config)
    memory = _mem.short_term
    scene_memory = _mem.scene
    user_memory = _mem.user
    notifier = Notifier(verbose=bool(config.verbose))

    # Build local skills first; attach endpoint-backed skills after transports and result routing.
    skill_registry = build_skill_registry(
        config, scene_memory=scene_memory, user_memory=user_memory
    )

    # The result router is the single funnel every transport/skill/judge result
    # flows through (see ResultRouter). Built here -- before transports -- because
    # build_transports needs an intake callback, while the router's own
    # collaborators (JobManager, EndpointInvoker) are built later and assigned
    # back onto it (late binding breaks the transports<->invoker cycle).
    router = ResultRouter(
        tx_logger=tx_logger,
        registry=registry,
        config=config,
        scene_memory=scene_memory,
    )

    transports = build_transports(
        config, logger=tx_logger, registry=registry,
        skill_registry=skill_registry, local_on_result=router.local_intake,
    )

    # result channel(s): always an HTTP webhook, plus a coexisting ROS callback
    # channel when any endpoint speaks ROS (see build_result_channel).
    result_channel, ros_callback_channel = build_result_channel(
        config, registry, logger=tx_logger
    )

    async def _noop_final(_job_key: str) -> None:
        pass

    async def _noop_status(_job_key: str, _payload: dict) -> None:
        pass

    jm = JobManager(
        registry=registry,
        transports=transports,
        memory=memory,
        on_final=_noop_final,
        on_status_event=_noop_status,
        tick_interval_s=config.job_tick_interval_s,
        judge_release_timeout_s=config.judge_release_timeout_s,
    )
    router.jm = jm
    judge_supervisor = None

    # Planner stack: VLM client (+ I/O logger), ContextBuilder, Planner
    # orchestrator, and the post-plan user-memory hooks. See build_planner_stack.
    pstack = build_planner_stack(
        config,
        registry=registry,
        memory=memory,
        scene_memory=scene_memory,
        user_memory=user_memory,
        tx_logger=tx_logger,
    )
    planner = pstack.planner
    cb = pstack.context_builder
    planner_vlm = pstack.planner_vlm

    executor = Executor(
        registry=registry,
        transports=transports,
        memory=memory,
        job_manager=jm,
        result_channel=result_channel,
        config=config,
        notifier=notifier,
        planner=planner,
        on_plan_finished=pstack.on_plan_finished,
    )

    # SceneCapture is shared by Reactor (for "after" failure snapshots) and
    # the VLM Judge supervisor (for "before" / subtask-start snapshots).
    # Camera list defers to the planner VLM config; the Judge supervisor will
    # also pass its own camera list to its judge client independently.
    scene_capture: SceneCapture | None = None
    if config.planner_vlm_cameras:
        scene_capture = SceneCapture(list(config.planner_vlm_cameras))

    check_capture = None
    check_chat = None
    # check_endpoint: registered ONLY when the endpoint catalog declares it.
    # The catalog is the same switch ContextBuilder._check_endpoint_registered
    # uses for the policy clause, so the skill and the prompt that describes it
    # can never disagree -- and a deployment that does not declare it (the real
    # robot) gets neither. It needs the planner VLM and a camera, so it is
    # registered here rather than in build_skill_registry.
    if "check_endpoint" in registry.names():
        # Its own camera list when one is given. Anything in planner_vlm.cameras
        # is attached to EVERY planning round, so a deployment with one camera
        # per place must not put them there -- the planner would see every place
        # at once and the skill would have nothing left to answer. Falling back
        # to the planner's list keeps a single-workspace robot working as before.
        check_cams = list(config.check_endpoint_cameras
                          if config.check_endpoint_cameras is not None
                          else (config.planner_vlm_cameras or []))
        if not check_cams:
            raise SystemExit(
                "check_endpoint is declared in the endpoint catalog but no "
                "cameras are configured for it (check_endpoint.cameras and "
                "planner_vlm.cameras are both empty); it cannot answer a "
                "presence question without a frame."
            )
        check_capture = (scene_capture
                         if config.check_endpoint_cameras is None
                         else SceneCapture(check_cams))
        # NOT planner_vlm: that is the planner CLIENT (it turns a request into a
        # plan) and has no .chat seam. check_endpoint wants the raw chat client
        # the judge also uses. Building our own rather than reaching into the
        # planner's internals keeps the two independent.
        if config.planner_vlm_provider != "openrouter":
            raise SystemExit(
                f"check_endpoint currently needs an OpenAI-compatible chat "
                f"client; planner_vlm.provider is "
                f"{config.planner_vlm_provider!r}. Refusing to guess a seam.")
        from .providers import OpenRouterChatClient
        check_chat = OpenRouterChatClient(
            api_key=config.planner_vlm_api_key or "",
            base_url=config.planner_vlm_base_url,
            timeout_s=config.planner_vlm_timeout_s,
        )
        from .skills.check_endpoint import CheckEndpointSkill
        skill_registry.register(CheckEndpointSkill(
            registry=registry,
            cameras=check_cams,
            scene_capture=check_capture,
            chat_client=check_chat,
            temperature=config.planner_vlm_temperature,
            model=config.planner_vlm_model,
            max_edge_px=config.planner_vlm_max_image_edge_px,
        ))
        tx_logger.emit("check_endpoint_registered", skill="check_endpoint",
                       cameras=[c.get("name") for c in check_cams],
                       planner_ambient_cameras=[c.get("name") for c in
                                                (config.planner_vlm_cameras or [])])

    # Reactor: closes the loop endpoint events -> planner VLM -> next dispatch.
    from .executor.reactor import Reactor
    reactor = Reactor(
        planner=planner,
        executor=executor,
        memory=memory,
        notifier=notifier,
        scene_capture=scene_capture,
    )
    executor.reactor = reactor

    jm.on_final = executor.on_job_final
    jm.on_status_event = executor.on_status_event

    # When enabled, a judge failure triggers recovery before normal release.
    # Load recovery dependencies lazily; leave the hook unset if assembly fails.
    recovery_recorder_registry = None  # Set only when recovery is enabled.
    recovery_bridges: list = []        # zmq clients to close on shutdown
    if config.recovery.enabled:
        raise ValueError("RoboLab evaluation requires recovery and Judge disabled")

    # MockTransport's optional auto-callback (wired below) simulates a VLA
    # webhook by feeding router.on_result, so the full pipeline can be exercised
    # without a real VLA.
    from .transports.mock import MockTransport  # local import

    # ---- EndpointInvoker. Late-binds onto the router so LocalTransport's
    # intake (router.local_intake, already handed to build_transports) resolves
    # to the fully-wired funnel. See skills/factory.py.
    invoker = build_endpoint_skill_stack(
        config,
        registry=registry,
        transports=transports,
        result_channel=result_channel,
    )
    router.invoker = invoker

    for tr in transports.values():
        if (
            isinstance(tr, MockTransport)
            and config.mock_auto_callback_after_s > 0
            and not config.vlm_judge_enabled
        ):
            tr.set_auto_callback(router.on_result, config.mock_auto_callback_after_s)

    if config.vlm_judge_enabled:
        raise ValueError("RoboLab evaluation requires recovery and Judge disabled")

    commands = CommandHandler(
        registry=registry, job_manager=jm, memory=memory,
        scene_memory=scene_memory, user_memory=user_memory,
        default_user_id=config.user_id,
    )


    _judge_swallows_executor_terminal = (
        bool(config.vlm_judge_enabled)
        and bool(config.vlm_judge_ignore_executor_terminal)
    )

    # External callbacks (webhook/ROS) route through the router so the
    # EndpointInvoker / SceneMemory hooks see every result; when the VLM judge
    # owns the terminal decision, route via executor_intake (which may downgrade
    # executor terminals to progress).
    default_intake = router.default_intake(_judge_swallows_executor_terminal)

    agent = AgentLoop(
        memory=memory,
        planner=planner,
        executor=executor,
        context_builder=cb,
        job_manager=jm,
        result_channel=result_channel,
        notifier=notifier,
        endpoint_registry=registry,
        command_handler=commands,
        config=config,
        result_intake=default_intake,
    )


    # Start the optional ROS callback channel before running the loop;
    # ROS endpoints will publish their results onto its topic.
    if ros_callback_channel is not None:
        try:
            await ros_callback_channel.start(default_intake)
        except Exception as e:
            sys.stderr.write(
                f"[warn] ROS result channel start failed: {e}\n"
            )
            ros_callback_channel = None

    try:
        await agent.run()
    finally:
        if ros_callback_channel is not None:
            try:
                await ros_callback_channel.stop()
            except Exception:
                pass
        if judge_supervisor is not None:
            try:
                await judge_supervisor.stop()
            except Exception:
                pass
        if check_chat is not None:
            try:
                await check_chat.aclose()
            except Exception:
                pass
        for _cap in (scene_capture, check_capture):
            if _cap is not None:
                try:
                    await _cap.aclose()
                except Exception:
                    pass
        for tr in transports.values():
            try:
                await tr.aclose()
            except Exception:
                pass
        try:
            await planner_vlm.aclose()
        except Exception:
            pass
        # Tear down recovery resources so the process can exit on Ctrl-C: stop the
        # state-recorder stream and close the zmq bridge sockets (LINGER=0), else
        # the shared zmq Context.instance() term blocks forever on shutdown.
        if recovery_recorder_registry is not None:
            try:
                recovery_recorder_registry.stop()
            except Exception:
                pass
        for _br in recovery_bridges:
            try:
                _br.close()
            except Exception:
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    import os

    args = _parse_args(argv)
    # Snapshot the terminal mode so the hard-exit below can restore it. The
    # REPL's input() reader (readline) puts the TTY into its own line mode and
    # normally restores it when input() returns; because that reader thread is
    # still parked in input() when we hard-exit, the restore never runs and the
    # shell is left with echo OFF after Ctrl-C. We re-apply the saved attrs.
    _tty_fd = None
    _tty_attrs = None
    try:
        import termios

        if sys.stdin is not None and sys.stdin.isatty():
            _tty_fd = sys.stdin.fileno()
            _tty_attrs = termios.tcgetattr(_tty_fd)
    except Exception:
        _tty_fd = None
        _tty_attrs = None
    code = 0
    try:
        code = asyncio.run(_amain(args.config))
    except KeyboardInterrupt:
        code = 130
    finally:
        # The REPL's stdin reader runs in the default ThreadPoolExecutor on a
        # NON-daemon thread blocked in input(). On Ctrl-C the main coroutine
        # unwinds and _amain's finally tears down resources, but that worker
        # thread stays parked in input() -- and Python's atexit join of it
        # (concurrent.futures.thread._python_exit -> t.join()) then blocks
        # forever, so the process never dies (Ctrl-C "doesn't work"). Cleanup
        # has already run, so flush and hard-exit to skip the hanging join.
        if _tty_fd is not None and _tty_attrs is not None:
            try:
                import termios

                termios.tcsetattr(_tty_fd, termios.TCSADRAIN, _tty_attrs)
            except Exception:
                pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(code if isinstance(code, int) else 0)
    return code


if __name__ == "__main__":
    sys.exit(main())
