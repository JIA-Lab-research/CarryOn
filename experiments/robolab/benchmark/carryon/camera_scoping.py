"""Validate and resolve the cameras declared for each endpoint."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _camera_names(cameras: Iterable[Mapping[str, Any]] | None) -> set[str]:
    """Extract the non-empty ``name`` fields from a configured camera list."""
    return {
        str(c.get("name") or "")
        for c in (cameras or [])
        if isinstance(c, Mapping)
    } - {""}


def validate_camera_scoping(
    *,
    judge_cameras: list[dict[str, Any]] | None,
    endpoints: Iterable[Any] = (),
) -> None:
    """Fail fast on camera-scoping names that match no configured camera.

    Endpoint ``extra.cameras`` names are checked against ``judge_cameras``
    (the set :func:`cameras_for_endpoint` filters at runtime).

    A consumer with NO configured cameras is skipped: its scoping lists are
    inert config (that consumer never snapshots), not a runtime hazard.

    Raises ``ValueError`` listing EVERY offending entry (endpoint name +
    unknown camera name), so one startup failure shows all typos.
    """
    problems: list[str] = []

    judge_names = _camera_names(judge_cameras)
    if judge_names:
        for ep in endpoints:
            wanted = (getattr(ep, "extra", None) or {}).get("cameras")
            if not isinstance(wanted, list):
                continue
            for n in wanted:
                if str(n) not in judge_names:
                    problems.append(
                        f"endpoint {getattr(ep, 'name', '?')!r}: cameras entry "
                        f"{n!r} matches no configured judge camera "
                        f"(configured: {sorted(judge_names)})"
                    )

    if problems:
        raise ValueError(
            "camera scoping references unknown camera name(s):\n  - "
            + "\n  - ".join(problems)
        )


__all__ = ["validate_camera_scoping"]
