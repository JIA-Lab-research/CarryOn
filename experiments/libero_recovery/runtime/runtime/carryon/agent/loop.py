from __future__ import annotations
import asyncio
import sys
from typing import Optional
from ..config import Config
from ..endpoints import EndpointRegistry
from ..errors import PlanError, CarryOnError
from ..executor.executor import Executor
from ..jobs.channels.base import OnResult, ResultChannel
from ..jobs.manager import JobManager
from ..memory import Memory
from ..planner.planner import Planner
from ..types import Message
from .commands import CommandHandler
from .context import ContextBuilder
from .notifier import Notifier

class AgentLoop:

    def __init__(self, *, memory: Memory, planner: Planner, executor: Executor, context_builder: ContextBuilder, job_manager: JobManager, result_channel: ResultChannel, notifier: Notifier, endpoint_registry: EndpointRegistry, command_handler: CommandHandler, config: Config, result_intake: Optional[OnResult]=None) -> None:
        self.memory = memory
        self.planner = planner
        self.executor = executor
        self.cb = context_builder
        self.jm = job_manager
        self.rc = result_channel
        self.notifier = notifier
        self.registry = endpoint_registry
        self.commands = command_handler
        self.config = config
        self._result_intake: Optional[OnResult] = result_intake
        self._latest_plan_id: Optional[str] = None
        self._bg_tasks: set[asyncio.Task] = set()
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        intake = self._result_intake if self._result_intake is not None else self.jm.on_result
        await self.rc.start(intake)
        await self.jm.start()

    async def stop(self) -> None:
        self._stopping.set()
        grace = float(self.config.shutdown_grace_s or 0.0)
        if grace > 0:
            deadline = asyncio.get_running_loop().time() + grace
            while self.jm.plans and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
        for plan_id in list(self.jm.plans.keys()):
            try:
                await self.jm.cancel_plan(plan_id)
            except Exception:
                pass
        await self.jm.stop()
        await self.rc.stop()
        for t in list(self._bg_tasks):
            t.cancel()
        for t in list(self._bg_tasks):
            try:
                await t
            except BaseException:
                pass

    async def run(self) -> None:
        await self.start()
        from ..logging_util import set_repl_prompt
        set_repl_prompt(self.notifier.prompt)
        try:
            sys.stdout.write(f"[callback server] listening on {self.config.callback_host}:{self.config.callback_port}\n[endpoints] loaded {len(self.registry.list())}: {', '.join(self.registry.names()) or '(none)'}\n")
            if self.config.planner_log_path:
                sys.stdout.write(f'[planner log] {self.config.planner_log_path} (provider={self.config.planner_vlm_provider} model={self.config.planner_vlm_model})\n')
            cameras = self.config.planner_vlm_cameras or []
            sys.stdout.write(f'[planner vision] provider={self.config.planner_vlm_provider} model={self.config.planner_vlm_model} cameras={len(cameras)}\n')
            if self.config.vlm_judge_enabled:
                judge_settings = self.config.vlm_judge_settings or {}
                judge_provider = (judge_settings.get('vlm') or {}).get('provider', '?')
                sys.stdout.write(f'[vlm judge] enabled provider={judge_provider}\n')
                applies = list(self.config.vlm_judge_applies_to_tags)
                if applies:
                    sys.stdout.write(f'[vlm judge] applies only to endpoints with any of tags={applies}\n')
                else:
                    sys.stdout.write('[vlm judge] applies to all endpoints (set vlm_judge.applies_to_tags to restrict)\n')
                if self.config.vlm_judge_ignore_executor_terminal:
                    sys.stdout.write('[vlm judge] executor terminal callbacks ignored (only judge closes jobs' + (' for endpoints in scope' if applies else '') + ')\n')
            sys.stdout.flush()
            while not self._stopping.is_set():
                try:
                    line = await self._read_line()
                except (EOFError, KeyboardInterrupt):
                    break
                if line is None:
                    break
                s = line.strip()
                if not s:
                    continue
                if s in ('exit', 'quit'):
                    break
                if CommandHandler.is_command(s):
                    out = await self.commands.dispatch(s)
                    if out:
                        sys.stdout.write(out + '\n')
                        sys.stdout.flush()
                    continue
                task = asyncio.create_task(self._safe_handle(s))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
        finally:
            sys.stdout.write(f'[shutdown] cancelling {len(self.jm.plans)} in-flight plans...\n')
            sys.stdout.flush()
            set_repl_prompt('')
            await self.stop()

    async def _read_line(self) -> Optional[str]:
        loop = asyncio.get_running_loop()
        prompt = self.notifier.prompt
        try:
            return await loop.run_in_executor(None, _blocking_input, prompt)
        except (EOFError, KeyboardInterrupt):
            raise

    async def handle(self, user_input: str) -> None:
        self.memory.append(Message(role='user', content=user_input))
        self.cb.mark_task_start()
        try:
            plan = await self.planner.plan(user_input)
        except PlanError as e:
            await self.notifier.info(f'[plan error] {e}')
            return
        self._latest_plan_id = plan.id
        self.memory.append(Message(role='assistant', content=f'plan {plan.id}: goal={plan.goal}; subtasks={[s.id for s in plan.subtasks]}', plan_id=plan.id, metadata={'kind': 'plan_created'}))
        await self.executor.submit_plan(plan)
        roots = [s.id for s in plan.subtasks if not s.depends_on]
        eps = sorted({s.endpoint for s in plan.subtasks if s.id in roots})
        await self.notifier.info(f"[{plan.id}] dispatched to: {', '.join(eps)}")

    async def _safe_handle(self, user_input: str) -> None:
        try:
            await self.handle(user_input)
        except CarryOnError as e:
            await self.notifier.info(f'[error] {e}')
        except Exception as e:
            await self.notifier.info(f'[error] {type(e).__name__}: {e}')

def _blocking_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise
