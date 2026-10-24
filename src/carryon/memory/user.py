"""UserMemory: the single durable, per-user profile of facts about a user.

One **markdown** file per user (``<store_dir>/<user_id>.md``) holds facts
grouped by ``category`` (a ``## <category>`` section) with one ``- key: value``
line per fact, across two user-centric namespaces:

- ``entity:*`` -- attributes of objects the user owns or refers to
  (``entity:cup / color = 绿色``, ``entity:cup / reference_image = <path>``).
- ``profile:*`` -- durable facts about the user themselves: identity, habits,
  and preferences (``profile:home / city = 北京``,
  ``profile:eating_habit / diet = 素食``,
  ``profile:cooking / boiled_egg = "2个，煮3分钟"``). The ``profile:avoid:*``
  sub-namespace holds durable *negative* preferences -- standing prohibitions
  the user does NOT want (``profile:avoid:dishwasher / cup = "别把我的杯子放进洗碗机"``);
  the planner surfaces them as avoid-constraints it must respect.

The single distillation writer is the post-plan
:class:`carryon.memory.user_summarizer.UserSummarizer`, which applies a
durability filter (one-off, session-scoped wishes never enter the profile).
The planner can also write durable facts mid-interaction via the
``update_user_memory`` skill (same store; later writes overwrite the same
keys). Reference images arrive through the same distillation pass: a user message
may carry image attachments (``@image:<path>`` in the CLI ->
``Message.metadata["attachments"]``), and when the summarizer distills a
durable ``entity:*`` fact grounded in such a message it stores the image as
the entity's ``reference_image``. ``/remember_image`` remains as a
manual/debug upload channel. The profile is exposed to the Planner VLM as *soft context* via
:meth:`render` (the markdown is injected verbatim); the planner infers how to
apply it -- resolving owner references (``抓A的杯子`` -> ``抓绿色的杯子``) and
deriving task behavior from profile facts (素食 -> 三明治不放肉).

Adding a new value for the same ``(category, key)`` overwrites the old one; the
markdown file holds only the latest value per key (no history). Use
:meth:`remove` / :meth:`remove_many` (or ``delete: true`` via the
``update_user_memory`` skill) to retract obsolete keys when a preference is
superseded.

Reference images (the "upload button" path) are stored on disk under
``<store_dir>/images/<user_id>/`` and only their path is written as a fact
value, keeping the markdown small.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .markdown_store import MarkdownMemory

NAMESPACES = ("entity", "profile")

# Negative preferences live under the ``profile:avoid`` category (or a
# ``profile:avoid:<topic>`` sub-namespace). They ride the exact same
# UserFact / markdown / durability-filter machinery as positive profile
# facts -- only the summarizer prompt (which distils them) and the planner
# context block (which frames them as avoid-constraints) are avoid-aware.
AVOID_CATEGORY_PREFIX = "profile:avoid"


def is_avoid_category(category: "str | None") -> bool:
    """True when ``category`` is a negative-preference (avoid) category."""
    cat = normalize_category(category)
    return cat == AVOID_CATEGORY_PREFIX or cat.startswith(AVOID_CATEGORY_PREFIX + ":")


@dataclass
class UserFact:
    id: str
    user_id: str
    category: str
    key: str
    value: Any
    source: Optional[str] = None
    extracted_at: float = field(default_factory=time.time)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UserFact":
        return cls(
            id=str(data.get("id") or _new_fact_id()),
            user_id=normalize_user_id(data.get("user_id")),
            category=normalize_category(data.get("category")),
            key=str(data.get("key") or "").strip(),
            value=data.get("value"),
            source=data.get("source"),
            extracted_at=float(data.get("extracted_at") or time.time()),
            confidence=float(data.get("confidence") or 0.0),
        )


class UserMemory:
    def __init__(self, store_dir: str, *, default_user_id: str = "default") -> None:
        self.store_dir = store_dir
        self.default_user_id = normalize_user_id(default_user_id)
        # uid -> {category: {key: value}}  (latest value per key)
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        self._stores: dict[str, MarkdownMemory] = {}
        self._lock = threading.RLock()

    # ----- public API -----

    def add(self, fact: UserFact) -> None:
        if not fact.category or not fact.key:
            raise ValueError("UserFact requires category and key")
        self.add_many([fact])

    def add_many(self, facts: list[UserFact]) -> None:
        kept = [f for f in facts if f.category and f.key]
        if not kept:
            return
        with self._lock:
            touched: set[str] = set()
            for fact in kept:
                uid = normalize_user_id(fact.user_id)
                category = normalize_category(fact.category)
                key = str(fact.key).strip()
                if not category or not key:
                    continue
                self._load_user_locked(uid)
                self._data.setdefault(uid, {}).setdefault(category, {})[key] = fact.value
                touched.add(uid)
            for uid in touched:
                self._save_user_locked(uid)

    def remove(self, user_id: str, category: str, key: str) -> bool:
        """Delete one ``(category, key)``. Returns True if it existed."""
        return self.remove_many([(user_id, category, key)]) > 0

    def remove_many(self, items: list[tuple[str, str, str]]) -> int:
        """Delete several ``(user_id, category, key)`` facts. Returns count removed."""
        removed = 0
        with self._lock:
            touched: set[str] = set()
            for user_id, category, key in items:
                uid = normalize_user_id(user_id)
                cat = normalize_category(category)
                k = str(key or "").strip()
                if not cat or not k:
                    continue
                self._load_user_locked(uid)
                bucket = self._data.get(uid, {}).get(cat)
                if not bucket or k not in bucket:
                    continue
                del bucket[k]
                if not bucket:
                    self._data[uid].pop(cat, None)
                removed += 1
                touched.add(uid)
            for uid in touched:
                self._save_user_locked(uid)
        return removed

    def query(self, user_id: str) -> dict[str, dict[str, Any]]:
        """Latest value per ``(category, key)`` for one user: ``{category: {key: value}}``."""
        uid = normalize_user_id(user_id)
        with self._lock:
            self._load_user_locked(uid)
            return {cat: dict(kv) for cat, kv in self._data.get(uid, {}).items() if kv}

    def list_user_ids(self) -> list[str]:
        with self._lock:
            users = set(self._data.keys())
            if self.store_dir and os.path.isdir(self.store_dir):
                for name in os.listdir(self.store_dir):
                    if name.endswith(".md"):
                        users.add(normalize_user_id(os.path.splitext(name)[0]))
            return sorted(users)

    def snapshot_for_planner(
        self, user_id: str | None = None
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """``{user_id: {category: {key: value}}}`` for the prompt / CLI."""
        if user_id is not None:
            snap = self.query(user_id)
            return {normalize_user_id(user_id): snap} if snap else {}
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for uid in self.list_user_ids():
            snap = self.query(uid)
            if snap:
                out[uid] = snap
        return out

    def render(self, user_id: str | None = None) -> str:
        """Markdown view for injection into the Planner system prompt.

        One ``# user: <id>`` heading per user, followed by that user's
        ``## <category>`` sections.
        """
        if user_id is not None:
            uid = normalize_user_id(user_id)
            with self._lock:
                self._load_user_locked(uid)
                body = self._stores.get(uid)
            rendered = body.render() if body is not None else ""
            return f"# user: {uid}\n{rendered}".rstrip() + "\n" if rendered else ""
        blocks: list[str] = []
        for uid in self.list_user_ids():
            rendered = self.render(uid)
            if rendered:
                blocks.append(rendered.rstrip())
        return "\n\n".join(blocks) + "\n" if blocks else ""

    def store_reference_image(self, user_id: str, src_path: str) -> str:
        """Copy an image file into the per-user image dir; return its stored path.

        This is the on-disk side of the ``/remember_image`` "upload button":
        the caller then records a fact whose value is the returned path.
        """
        uid = normalize_user_id(user_id)
        if not src_path or not os.path.isfile(src_path):
            raise FileNotFoundError(f"reference image not found: {src_path!r}")
        img_dir = os.path.join(self.store_dir, "images", _safe_filename(uid))
        os.makedirs(img_dir, exist_ok=True)
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        # The ms timestamp alone is NOT unique: one summarizer batch stores
        # several reference images within the same millisecond, and the
        # second copy silently OVERWROTE the first (two entities then shared
        # one image file). Suffix a uuid fragment to make the name unique.
        import uuid as _uuid
        dst = os.path.join(
            img_dir,
            f"img_{int(time.time() * 1000)}_{_uuid.uuid4().hex[:6]}{ext}")
        shutil.copyfile(src_path, dst)
        return dst

    def all(self, user_id: str | None = None) -> list[UserFact]:
        """Reconstruct UserFacts from the markdown (no stored history/ids)."""
        with self._lock:
            uids = (
                [normalize_user_id(user_id)] if user_id is not None
                else self.list_user_ids()
            )
            facts: list[UserFact] = []
            for uid in uids:
                self._load_user_locked(uid)
                for category, kv in self._data.get(uid, {}).items():
                    for key, value in kv.items():
                        facts.append(UserFact(
                            id=_new_fact_id(), user_id=uid, category=category,
                            key=key, value=value,
                        ))
            return facts

    def clear(self, user_id: str | None = None) -> None:
        uid = normalize_user_id(user_id or self.default_user_id)
        with self._lock:
            self._data[uid] = {}
            self._store_for_locked(uid).clear()

    # ----- internals -----

    def _store_for_locked(self, user_id: str) -> MarkdownMemory:
        uid = normalize_user_id(user_id)
        store = self._stores.get(uid)
        if store is None:
            store = MarkdownMemory(self._path_for_user(uid))
            self._stores[uid] = store
        return store

    def _load_user_locked(self, user_id: str) -> None:
        uid = normalize_user_id(user_id)
        if uid in self._data:
            return
        store = self._store_for_locked(uid)
        data: dict[str, dict[str, Any]] = {}
        for category, body in store.sections().items():
            cat = normalize_category(category)
            if not cat:
                continue
            kv = _parse_kv_lines(body)
            if kv:
                data[cat] = kv
        self._data[uid] = data

    def _save_user_locked(self, user_id: str) -> None:
        if not self.store_dir:
            return
        uid = normalize_user_id(user_id)
        sections = {
            category: _render_kv_lines(kv)
            for category, kv in self._data.get(uid, {}).items()
            if kv
        }
        self._store_for_locked(uid).replace_all(sections)

    def _path_for_user(self, user_id: str) -> str:
        return os.path.join(self.store_dir, f"{_safe_filename(user_id)}.md")


def _render_kv_lines(kv: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in kv.items():
        lines.append(f"- {key}: {_dump_value(value)}")
    return "\n".join(lines)


def _parse_kv_lines(body: str) -> dict[str, Any]:
    kv: dict[str, Any] = {}
    for line in (body or "").splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        rest = line[1:].strip()
        if ":" not in rest:
            continue
        key, _, raw = rest.partition(":")
        key = key.strip()
        if key:
            kv[key] = _load_value(raw.strip())
    return kv


def _dump_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _load_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def normalize_category(category: str | None) -> str:
    raw = str(category or "").strip()
    if not raw:
        return ""
    if ":" in raw:
        ns, _, rest = raw.partition(":")
        ns = ns.strip().lower()
        rest = rest.strip()
        if ns in NAMESPACES and rest:
            return f"{ns}:{rest}"
        # Legacy ``task:*`` sections (pre profile-merge) fold into profile:*.
        if ns == "task" and rest:
            return f"profile:{rest}"
        return raw
    # Bare category without a namespace prefix -> a fact about the user.
    return f"profile:{raw}"


def normalize_user_id(user_id: str | None) -> str:
    uid = str(user_id or "default").strip()
    uid = re.sub(r"^(?:请)?(?:记住|记一下|记得)", "", uid).strip()
    uid = re.sub(r"^(?:remember|note\s+that|note)\s+", "", uid, flags=re.IGNORECASE).strip()
    return uid or "default"


def _safe_filename(user_id: str) -> str:
    safe = re.sub(r"[<>:\"/\\|?*\x00-\x1f\s]+", "_", normalize_user_id(user_id)).strip("._")
    return safe or "default"


def _new_fact_id() -> str:
    return f"uf-{uuid.uuid4().hex[:10]}"
