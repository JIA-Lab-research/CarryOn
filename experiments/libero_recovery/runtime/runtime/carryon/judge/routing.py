"""Route endpoint callbacks when the visual judge owns terminal state.

The meta.source value vlm_judge_intake identifies judge-owned payloads consumed
by JobManager."""
from __future__ import annotations
from typing import Any, Iterable, Optional
from ..types import is_terminal_status

def endpoint_under_judge(endpoint_tags: Iterable[str] | None, applies_to_tags: Iterable[str] | None) -> bool:
    """Return True if VLM Judge should supervise an endpoint with these tags.

    Semantics: when ``applies_to_tags`` is empty / unset, Judge applies to
    every endpoint (legacy behaviour). Otherwise Judge applies only when
    ``endpoint_tags`` intersects ``applies_to_tags`` (ANY-of whitelist).
    """
    whitelist = list(applies_to_tags or [])
    if not whitelist:
        return True
    tags = set(endpoint_tags or [])
    return any((t in tags for t in whitelist))

def swallowed_executor_terminal_payload(*, vlm_judge_enabled: bool, ignore_executor_terminal: bool, payload: dict[str, Any], endpoint_tags: Iterable[str] | None=None, applies_to_tags: Iterable[str] | None=None) -> Optional[dict[str, Any]]:
    """If this callback should not close the job, return a progress-shaped dict.

    When VLM judge is enabled and ``ignore_executor_terminal`` is true, any
    inbound payload whose ``status`` is terminal but ``meta.source`` is not
    ``vlm_judge`` is converted into a non-terminal ``progress`` event so
    JobManager keeps the job running until the judge reports a terminal state.

    The downgrade is additionally gated by the tag whitelist (see
    :func:`endpoint_under_judge`): if the endpoint is not in scope for Judge,
    the executor's terminal callback is forwarded unchanged so its native
    success/failed semantics keep working.

    Returns ``None`` when the original payload should be forwarded unchanged.
    """
    if not vlm_judge_enabled or not ignore_executor_terminal:
        return None
    if not endpoint_under_judge(endpoint_tags, applies_to_tags):
        return None
    p = payload or {}
    meta = p.get('meta') or {}
    if not isinstance(meta, dict) or meta.get('source') == 'vlm_judge':
        return None
    st = str(p.get('status') or '').lower()
    if not is_terminal_status(st):
        return None
    return {'status': 'progress', 'terminal': False, 'detail': f'executor reported terminal status={st!r}; ignored while awaiting VLM judge', 'result': p.get('result'), 'error': p.get('error'), 'meta': {'source': 'vlm_judge_intake', 'kind': 'ignored_executor_terminal', 'original_status': st}}
