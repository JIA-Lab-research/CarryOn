from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from ..endpoints import EndpointRegistry, RobotEndpoint
from ..errors import CommandError
from ..memory import Memory, SceneMemory, UserMemory


if TYPE_CHECKING:
    from ..jobs.manager import JobManager


HELP_TEXT = """\
Available commands:
  /help                                     show this help
  /list                                     list registered endpoints
  /describe <name>                          print full endpoint info
  /register <name> <transport> <address> "<description>" [tag1,tag2]
  /unregister <name>
  /jobs [plan_id]                           list in-flight jobs
  /cancel <plan_id>                         cancel a plan's in-flight jobs
  /memory [n]                               print last n memory entries (default 10)
  /scene [clear]                            show / clear scene memory (markdown)
  /remember_image [user_id] <path> <entity[.key]>  attach a reference image to an entity (debug backdoor; entity is required)
  /user [user_id] [clear]                   list / clear stored user facts

Attachments: put @image:/path/to/pic.jpg (or @image:"path with spaces") anywhere
in a normal message to attach the image. When the post-plan summarizer distills
a durable entity fact grounded in that message ("this is my lizard" + photo),
the image is stored automatically as the entity's reference image.
"""


class CommandHandler:
    def __init__(
        self,
        registry: EndpointRegistry,
        job_manager: "JobManager | None",
        memory: Memory,
        scene_memory: "SceneMemory | None" = None,
        user_memory: "UserMemory | None" = None,
        default_user_id: str = "default",
    ) -> None:

        self.registry = registry
        self.jobs = job_manager
        self.memory = memory
        self.scene_memory = scene_memory
        self.user_memory = user_memory
        self.default_user_id = default_user_id


    @staticmethod
    def is_command(line: str) -> bool:
        return line.strip().startswith("/")

    async def dispatch(self, line: str) -> str:
        line = line.strip()
        try:
            parts = shlex.split(line)
        except ValueError as e:
            return f"[cmd error] parse: {e}"
        if not parts:
            return ""
        cmd = parts[0].lstrip("/").lower()
        args = parts[1:]
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return f"[cmd error] unknown command: /{cmd}. try /help"
        try:
            return await handler(args)
        except CommandError as e:
            return f"[cmd error] {e}"
        except Exception as e:  # last-resort; do not crash REPL
            return f"[cmd error] {type(e).__name__}: {e}"

    # ---- handlers ----

    async def _cmd_help(self, args: list[str]) -> str:
        return HELP_TEXT

    async def _cmd_list(self, args: list[str]) -> str:
        eps = self.registry.list()
        if not eps:
            return "(no endpoints registered)"
        lines = [f"{'name':<20} {'transport':<10} {'tags':<24} description"]
        for ep in eps:
            tags = ",".join(ep.tags)
            desc = (ep.description or "").splitlines()[0][:60]
            lines.append(f"{ep.name:<20} {ep.transport:<10} {tags:<24} {desc}")
        return "\n".join(lines)

    async def _cmd_describe(self, args: list[str]) -> str:
        if not args:
            raise CommandError("usage: /describe <name>")
        ep = self.registry.get(args[0])
        return (
            f"name:                 {ep.name}\n"
            f"transport:            {ep.transport}\n"
            f"address:              {ep.address}\n"
            f"tags:                 {', '.join(ep.tags)}\n"
            f"cancel_path:          {ep.cancel_path}\n"
            f"return_home_on_cancel:{ep.return_home_on_cancel}\n"
            f"description:          {ep.description}"
        )

    async def _cmd_register(self, args: list[str]) -> str:
        if len(args) < 4:
            raise CommandError(
                'usage: /register <name> <transport> <address> "<description>" [tag1,tag2]'
            )
        name, transport, address, description = args[0], args[1], args[2], args[3]
        tags: list[str] = []
        if len(args) >= 5:
            tags = [t for t in args[4].split(",") if t]
        ep = RobotEndpoint(
            name=name,
            transport=transport,
            address=address,
            description=description,
            tags=tags,
        )
        self.registry.register(ep)
        return f"[registered] {name}"

    async def _cmd_unregister(self, args: list[str]) -> str:
        if not args:
            raise CommandError("usage: /unregister <name>")
        name = args[0]
        self.registry.unregister(name)
        return f"[unregistered] {name}"

    async def _cmd_jobs(self, args: list[str]) -> str:
        if self.jobs is None:
            return "(no job manager attached)"
        plan_id = args[0] if args else None
        records = (
            self.jobs.list_plan_jobs(plan_id) if plan_id else self.jobs.list_jobs()
        )
        if not records:
            return "(no jobs)"
        lines = [f"{'job_key':<28} {'endpoint':<18} {'status':<10} attempts"]
        for r in records:
            lines.append(
                f"{r.job_key:<28} {r.endpoint_name:<18} {r.status:<10} {r.attempts}"
            )
        return "\n".join(lines)

    async def _cmd_cancel(self, args: list[str]) -> str:
        if self.jobs is None:
            return "(no job manager attached)"
        if not args:
            raise CommandError("usage: /cancel <plan_id>")
        plan_id = args[0]
        n = await self.jobs.cancel_plan(plan_id)
        return f"[cancelled] {plan_id}: {n} jobs"

    async def _cmd_memory(self, args: list[str]) -> str:
        try:
            n = int(args[0]) if args else 10
        except ValueError:
            raise CommandError("usage: /memory [n]")
        msgs = self.memory.messages()[-n:]
        if not msgs:
            return "(memory empty)"
        return "\n".join(
            f"[{m.role}] {m.content[:200]}" for m in msgs
        )

    async def _cmd_scene(self, args: list[str]) -> str:
        if self.scene_memory is None:
            return "(scene memory disabled)"
        if args and args[0] == "clear":
            self.scene_memory.clear()
            return "[scene] cleared"
        rendered = self.scene_memory.render()
        if not rendered.strip():
            return "(scene memory empty)"
        return rendered.rstrip()

    async def _cmd_remember_image(self, args: list[str]) -> str:
        """Debug backdoor for attaching a reference image to an entity fact.

        Usage: /remember_image [user_id] <path> <entity[.key]>
        Stores the image on disk and records an ``entity:<name>`` reference
        fact whose value is the stored path. ``user_id`` defaults to the
        session user. The last argument names the entity (and optionally the
        fact key): ``lizard`` -> entity:lizard / reference_image;
        ``entity:lizard.photo`` -> entity:lizard / photo. The entity is
        REQUIRED -- there is no default object (a hardcoded default entity
        silently misfiles other objects and is a personalization
        anti-pattern); only the fact key defaults, to ``reference_image``.

        The primary ingestion path is automatic: user messages carrying image
        attachments are distilled by the post-plan UserSummarizer (ADR 0008);
        this command remains as a manual/debug channel.
        """
        from ..memory import UserFact
        from ..memory.user import normalize_category

        if self.user_memory is None:
            return "(user memory disabled)"
        if not args:
            raise CommandError(
                "usage: /remember_image [user_id] <path> <entity[.key]>"
            )
        # Resolve which arg is the path: the first existing file path.
        import os

        path_idx = next(
            (i for i, a in enumerate(args) if os.path.isfile(a)), None
        )
        if path_idx is None:
            raise CommandError(f"no readable image file in: {args}")
        path = args[path_idx]
        user_id = args[0] if path_idx > 0 else self.default_user_id
        # Derive the category (and optional key) from the argument after the
        # path: "<entity>", "entity:<name>", or "<entity>.<key>".
        if len(args) <= path_idx + 1:
            raise CommandError(
                "missing entity: usage /remember_image [user_id] <path> <entity[.key]>"
            )
        spec = args[path_idx + 1]
        entity, _, key = spec.partition(".")
        key = key.strip() or "reference_image"
        entity = entity.strip()
        if not entity:
            raise CommandError(
                "missing entity: usage /remember_image [user_id] <path> <entity[.key]>"
            )
        category = (
            normalize_category(entity) if ":" in entity else f"entity:{entity}"
        )
        try:
            stored = self.user_memory.store_reference_image(user_id, path)
        except (FileNotFoundError, OSError) as e:
            raise CommandError(f"store image failed: {e}")
        self.user_memory.add(UserFact(
            id="", user_id=user_id, category=category, key=key,
            value=stored, source="/remember_image",
        ))
        return f"[remember_image] stored: {user_id}.{category}.{key}={stored}"

    async def _cmd_user(self, args: list[str]) -> str:
        if self.user_memory is None:
            return "(user memory disabled)"
        if args and args[-1] == "clear":
            user_id = args[0] if len(args) > 1 else self.default_user_id
            self.user_memory.clear(user_id)
            return f"[user] cleared {user_id}"
        user_id = args[0] if args else None
        snap = self.user_memory.snapshot_for_planner(user_id)
        if not snap:
            return "(no user facts recorded)"
        lines = []
        for uid, categories in sorted(snap.items()):
            lines.append(f"[{uid}]")
            for category, kv in sorted(categories.items()):
                for k, v in sorted(kv.items()):
                    lines.append(f"  {category}.{k} = {v}")
        return "\n".join(lines)

