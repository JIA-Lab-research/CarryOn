"""UpdateUserMemorySkill: planner-driven write into UserMemory.

Lets the Planner persist a durable user fact mid-interaction -- e.g. the user
says they are vegetarian, or that their cup is green -- without waiting for the
post-plan UserSummarizer. Prefer this skill over inventing robot busywork when
the turn is only about remembering something.

The post-plan :class:`~carryon.memory.user_summarizer.UserSummarizer` remains
as a backup distillation path; it sees the on-disk profile and drops facts
already covered (including ``profile:avoid`` nesting aliases).

Inputs (single fact)::

    {"category": "profile:eating_habit", "key": "diet",
     "value": "vegetarian; does not eat meat"}

Inputs (many facts, including a retract)::

    {"facts": [{"category": "profile:eating_habit", "key": "diet",
                "value": "meat eater"},
               {"category": "profile:avoid:food", "key": "greens",
                "value": "does not eat greens/lettuce"},
               {"category": "entity:cup", "key": "color", "value": "green"},
               {"category": "profile:avoid:food", "key": "meat",
                "delete": true}]}

``delete: true`` (or ``value: null``) removes an obsolete ``(category, key)``.
``user_id`` defaults to the configured session user when omitted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import ToolResult

if TYPE_CHECKING:
    from ..memory import UserMemory


def _is_delete(item: dict) -> bool:
    if item.get("delete") is True:
        return True
    # Explicit JSON null means retract; missing value is an incomplete write.
    return "value" in item and item.get("value") is None


class UpdateUserMemorySkill:
    name = "update_user_memory"

    def __init__(
        self, *, user_memory: "UserMemory", default_user_id: str = "default"
    ) -> None:
        self._user = user_memory
        self._default_user_id = default_user_id

    async def run(self, instruction: str, inputs: dict) -> ToolResult:
        # Imported lazily to avoid a hard import cycle at module load.
        from ..memory import UserFact

        inputs = inputs or {}
        raw_facts = inputs.get("facts")
        if not isinstance(raw_facts, list) or not raw_facts:
            raw_facts = [inputs]

        facts: list = []
        deletions: list[tuple[str, str, str]] = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "").strip()
            key = str(item.get("key") or "").strip()
            if not category or not key:
                continue
            uid = str(item.get("user_id") or self._default_user_id)
            if _is_delete(item):
                deletions.append((uid, category, key))
                continue
            if "value" not in item:
                continue
            facts.append(
                UserFact(
                    id="",
                    user_id=uid,
                    category=category,
                    key=key,
                    value=item.get("value"),
                    source="update_user_memory",
                    confidence=1.0,
                )
            )

        if not facts and not deletions:
            return ToolResult(
                ok=False,
                error=(
                    "update_user_memory requires category + key + value "
                    "(or delete:true / value:null to retract), or "
                    "inputs.facts: list of those objects"
                ),
            )
        if facts:
            self._user.add_many(facts)
        removed = self._user.remove_many(deletions) if deletions else 0
        parts: list[str] = [
            f"{f.user_id}.{f.category}.{f.key}={f.value}" for f in facts
        ]
        for uid, category, key in deletions:
            parts.append(f"{uid}.{category}.{key}=<deleted>")
        return ToolResult(
            ok=True,
            data={
                "stored": len(facts),
                "deleted": removed,
                "summary": ", ".join(parts),
            },
        )


__all__ = ["UpdateUserMemorySkill"]
