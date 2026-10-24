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
from .judge.factory import build_judge_supervisor
from .memory.factory import build_memory
from .planner.factory import build_planner_stack
from .providers import SceneCapture
from .skills import build_endpoint_skill_stack, build_skill_registry
from .transports.factory import build_transports

def _parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='carryon', description='High-level robot dispatch agent.')
    p.add_argument('-c', '--config', default=None, help='Path to config YAML (default: $CARRYON_CONFIG or ./config.yaml).')
    return p.parse_args(argv)

async def _amain(config_path: str | None) -> int:
    try:
        config = Config.from_file(config_path)
    except FileNotFoundError as e:
        sys.stderr.write(f'[config error] {e}\n')
        return 2
    except Exception as e:
        sys.stderr.write(f'[config error] {type(e).__name__}: {e}\n')
        return 2
    registry = load_endpoints_yaml(config.endpoints_config)
    from .logging_util import EventLogger
    tx_logger = EventLogger(path=config.transport_log_path, stdout=config.transport_log_stdout, channel='transport')
    _mem = build_memory(config)
    memory = _mem.short_term
    scene_memory = _mem.scene
    user_memory = _mem.user
    notifier = Notifier()
    skill_registry = build_skill_registry(config, scene_memory=scene_memory)
    router = ResultRouter(tx_logger=tx_logger, registry=registry, config=config, scene_memory=scene_memory)
    transports = build_transports(config, logger=tx_logger, registry=registry, skill_registry=skill_registry, local_on_result=router.local_intake)
    result_channel, ros_callback_channel = build_result_channel(config, registry, logger=tx_logger)

    async def _noop_final(_job_key: str) -> None:
        pass

    async def _noop_status(_job_key: str, _payload: dict) -> None:
        pass
    jm = JobManager(registry=registry, transports=transports, memory=memory, on_final=_noop_final, on_status_event=_noop_status, tick_interval_s=config.job_tick_interval_s, judge_release_timeout_s=config.judge_release_timeout_s)
    router.jm = jm
    judge_supervisor = None
    pstack = build_planner_stack(config, registry=registry, memory=memory, scene_memory=scene_memory, user_memory=user_memory)
    planner = pstack.planner
    cb = pstack.context_builder
    planner_vlm = pstack.planner_vlm
    executor = Executor(registry=registry, transports=transports, memory=memory, job_manager=jm, result_channel=result_channel, config=config, notifier=notifier, planner=planner, on_plan_finished=pstack.on_plan_finished)
    scene_capture: SceneCapture | None = None
    if config.planner_vlm_cameras:
        scene_capture = SceneCapture(list(config.planner_vlm_cameras))
    from .executor.reactor import Reactor
    reactor = Reactor(planner=planner, executor=executor, memory=memory, notifier=notifier, scene_capture=scene_capture)
    executor.reactor = reactor
    jm.on_final = executor.on_job_final
    jm.on_status_event = executor.on_status_event
    recovery_recorder_registry = None
    recovery_bridges: list = []
    recovery_gello = None
    if config.recovery.enabled:
        from .recovery.factory import build_recovery
        _recovery_wiring = build_recovery(config, jm)
        recovery_recorder_registry = _recovery_wiring.recorder_registry
        recovery_bridges = _recovery_wiring.bridges
        recovery_gello = _recovery_wiring.gello
    from .transports.mock import MockTransport
    invoker = build_endpoint_skill_stack(config, skill_registry=skill_registry, registry=registry, transports=transports, result_channel=result_channel, scene_memory=scene_memory, scene_capture=scene_capture)
    router.invoker = invoker
    for tr in transports.values():
        if isinstance(tr, MockTransport) and config.mock_auto_callback_after_s > 0 and (not config.vlm_judge_enabled):
            tr.set_auto_callback(router.on_result, config.mock_auto_callback_after_s)
    if config.vlm_judge_enabled:
        try:
            judge_supervisor, scene_capture = build_judge_supervisor(config, scene_capture=scene_capture, reactor=reactor, memory=memory, executor=executor, recovery_recorder_registry=recovery_recorder_registry, on_result=router.on_result, gello=recovery_gello, job_lookup=jm.get)
        except Exception as e:
            sys.stderr.write(f'[config error] VLM judge: {type(e).__name__}: {e}\n')
            return 2
    commands = CommandHandler(registry=registry, job_manager=jm, memory=memory, scene_memory=scene_memory, user_memory=user_memory, default_user_id=config.user_id)
    _judge_swallows_executor_terminal = bool(config.vlm_judge_enabled) and bool(config.vlm_judge_ignore_executor_terminal)
    default_intake = router.default_intake(_judge_swallows_executor_terminal)
    agent = AgentLoop(memory=memory, planner=planner, executor=executor, context_builder=cb, job_manager=jm, result_channel=result_channel, notifier=notifier, endpoint_registry=registry, command_handler=commands, config=config, result_intake=default_intake)
    if ros_callback_channel is not None:
        try:
            await ros_callback_channel.start(default_intake)
        except Exception as e:
            sys.stderr.write(f'[warn] ROS result channel start failed: {e}\n')
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
        if scene_capture is not None:
            try:
                await scene_capture.aclose()
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

def main(argv: list[str] | None=None) -> int:
    import os
    args = _parse_args(argv)
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
if __name__ == '__main__':
    sys.exit(main())
