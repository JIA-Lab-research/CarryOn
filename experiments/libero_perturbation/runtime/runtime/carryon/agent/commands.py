from __future__ import annotations
import shlex
from typing import TYPE_CHECKING
from ..endpoints import EndpointRegistry, RobotEndpoint
from ..errors import CommandError
from ..memory import Memory, SceneMemory, UserMemory
if TYPE_CHECKING:
    from ..jobs.manager import JobManager
HELP_TEXT = 'Available commands:\n  /help                                     show this help\n  /list                                     list registered endpoints\n  /describe <name>                          print full endpoint info\n  /register <name> <transport> <address> "<description>" [tag1,tag2]\n  /unregister <name>\n  /jobs [plan_id]                           list in-flight jobs\n  /cancel <plan_id>                         cancel a plan\'s in-flight jobs\n  /memory [n]                               print last n memory entries (default 10)\n  /scene [clear]                            show / clear scene memory (markdown)\n  /remember_image [user_id] <path> [key]    attach a reference image to entity:cup (upload-button placeholder)\n  /user [user_id] [clear]                   list / clear stored user facts\n'

class CommandHandler:

    def __init__(self, registry: EndpointRegistry, job_manager: 'JobManager | None', memory: Memory, scene_memory: 'SceneMemory | None'=None, user_memory: 'UserMemory | None'=None, default_user_id: str='default') -> None:
        self.registry = registry
        self.jobs = job_manager
        self.memory = memory
        self.scene_memory = scene_memory
        self.user_memory = user_memory
        self.default_user_id = default_user_id

    @staticmethod
    def is_command(line: str) -> bool:
        return line.strip().startswith('/')

    async def dispatch(self, line: str) -> str:
        line = line.strip()
        try:
            parts = shlex.split(line)
        except ValueError as e:
            return f'[cmd error] parse: {e}'
        if not parts:
            return ''
        cmd = parts[0].lstrip('/').lower()
        args = parts[1:]
        handler = getattr(self, f'_cmd_{cmd}', None)
        if handler is None:
            return f'[cmd error] unknown command: /{cmd}. try /help'
        try:
            return await handler(args)
        except CommandError as e:
            return f'[cmd error] {e}'
        except Exception as e:
            return f'[cmd error] {type(e).__name__}: {e}'

    async def _cmd_help(self, args: list[str]) -> str:
        return HELP_TEXT

    async def _cmd_list(self, args: list[str]) -> str:
        eps = self.registry.list()
        if not eps:
            return '(no endpoints registered)'
        lines = [f"{'name':<20} {'transport':<10} {'tags':<24} description"]
        for ep in eps:
            tags = ','.join(ep.tags)
            desc = (ep.description or '').splitlines()[0][:60]
            lines.append(f'{ep.name:<20} {ep.transport:<10} {tags:<24} {desc}')
        return '\n'.join(lines)

    async def _cmd_describe(self, args: list[str]) -> str:
        if not args:
            raise CommandError('usage: /describe <name>')
        ep = self.registry.get(args[0])
        return f"name:                 {ep.name}\ntransport:            {ep.transport}\naddress:              {ep.address}\ntags:                 {', '.join(ep.tags)}\ncancel_path:          {ep.cancel_path}\nreturn_home_on_cancel:{ep.return_home_on_cancel}\ndescription:          {ep.description}"

    async def _cmd_register(self, args: list[str]) -> str:
        if len(args) < 4:
            raise CommandError('usage: /register <name> <transport> <address> "<description>" [tag1,tag2]')
        name, transport, address, description = (args[0], args[1], args[2], args[3])
        tags: list[str] = []
        if len(args) >= 5:
            tags = [t for t in args[4].split(',') if t]
        ep = RobotEndpoint(name=name, transport=transport, address=address, description=description, tags=tags)
        self.registry.register(ep)
        return f'[registered] {name}'

    async def _cmd_unregister(self, args: list[str]) -> str:
        if not args:
            raise CommandError('usage: /unregister <name>')
        name = args[0]
        self.registry.unregister(name)
        return f'[unregistered] {name}'

    async def _cmd_jobs(self, args: list[str]) -> str:
        if self.jobs is None:
            return '(no job manager attached)'
        plan_id = args[0] if args else None
        records = self.jobs.list_plan_jobs(plan_id) if plan_id else self.jobs.list_jobs()
        if not records:
            return '(no jobs)'
        lines = [f"{'job_key':<28} {'endpoint':<18} {'status':<10} attempts"]
        for r in records:
            lines.append(f'{r.job_key:<28} {r.endpoint_name:<18} {r.status:<10} {r.attempts}')
        return '\n'.join(lines)

    async def _cmd_cancel(self, args: list[str]) -> str:
        if self.jobs is None:
            return '(no job manager attached)'
        if not args:
            raise CommandError('usage: /cancel <plan_id>')
        plan_id = args[0]
        n = await self.jobs.cancel_plan(plan_id)
        return f'[cancelled] {plan_id}: {n} jobs'

    async def _cmd_memory(self, args: list[str]) -> str:
        try:
            n = int(args[0]) if args else 10
        except ValueError:
            raise CommandError('usage: /memory [n]')
        msgs = self.memory.messages()[-n:]
        if not msgs:
            return '(memory empty)'
        return '\n'.join((f'[{m.role}] {m.content[:200]}' for m in msgs))

    async def _cmd_scene(self, args: list[str]) -> str:
        if self.scene_memory is None:
            return '(scene memory disabled)'
        if args and args[0] == 'clear':
            self.scene_memory.clear()
            return '[scene] cleared'
        rendered = self.scene_memory.render()
        if not rendered.strip():
            return '(scene memory empty)'
        return rendered.rstrip()

    async def _cmd_remember_image(self, args: list[str]) -> str:
        """Stand-in for a future UI 'upload file' button.

        Usage: /remember_image [user_id] <path> [key]
        Stores the image on disk and records an entity:cup reference fact whose
        value is the stored path. ``user_id`` defaults to the session user;
        ``key`` defaults to ``reference_image`` (category fixed to entity:cup).
        """
        from ..memory import UserFact
        if self.user_memory is None:
            return '(user memory disabled)'
        if not args:
            raise CommandError('usage: /remember_image [user_id] <path> [key]')
        import os
        path_idx = next((i for i, a in enumerate(args) if os.path.isfile(a)), None)
        if path_idx is None:
            raise CommandError(f'no readable image file in: {args}')
        path = args[path_idx]
        user_id = args[0] if path_idx > 0 else self.default_user_id
        key = args[path_idx + 1] if len(args) > path_idx + 1 else 'reference_image'
        try:
            stored = self.user_memory.store_reference_image(user_id, path)
        except (FileNotFoundError, OSError) as e:
            raise CommandError(f'store image failed: {e}')
        self.user_memory.add(UserFact(id='', user_id=user_id, category='entity:cup', key=key, value=stored, source='/remember_image'))
        return f'[remember_image] stored: {user_id}.entity:cup.{key}={stored}'

    async def _cmd_user(self, args: list[str]) -> str:
        if self.user_memory is None:
            return '(user memory disabled)'
        if args and args[-1] == 'clear':
            user_id = args[0] if len(args) > 1 else self.default_user_id
            self.user_memory.clear(user_id)
            return f'[user] cleared {user_id}'
        user_id = args[0] if args else None
        snap = self.user_memory.snapshot_for_planner(user_id)
        if not snap:
            return '(no user facts recorded)'
        lines = []
        for uid, categories in sorted(snap.items()):
            lines.append(f'[{uid}]')
            for category, kv in sorted(categories.items()):
                for k, v in sorted(kv.items()):
                    lines.append(f'  {category}.{k} = {v}')
        return '\n'.join(lines)
