"""Geometric and record-completeness checks for recovery evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .geometry import GeoCheck, GeometryValidator
from .schema import EpisodeRecord


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"          # not applicable to this record (e.g. no recovery ran)
    PENDING_IMPL = "pending_impl"  # geometry backend not wired yet


class CheckSeverity(str, Enum):
    BLOCKING = "blocking"        # must PASS to freeze a policy
    WARNING = "warning"          # advisory; recorded but non-blocking


class CheckTier(str, Enum):
    DATA = "data"
    GEOMETRY = "geometry"


@dataclass
class CheckSpec:
    """Description of a record check, keyed by its identifier."""

    id: str
    title: str
    tier: CheckTier
    severity: CheckSeverity
    description: str
    fn: Callable[..., "CheckResult"]
    # Geometry-tier checks receive the optional GeometryValidator as a 2nd arg.
    needs_geometry: bool = False


@dataclass
class CheckResult:
    id: str
    status: CheckStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        # PENDING_IMPL / SKIPPED do not count as failures.
        return self.status in (CheckStatus.PASS, CheckStatus.SKIPPED, CheckStatus.PENDING_IMPL)


@dataclass
class ValidationReport:
    episode_id: str
    results: list[CheckResult] = field(default_factory=list)

    def by_id(self, check_id: str) -> Optional[CheckResult]:
        return next((r for r in self.results if r.id == check_id), None)

    @property
    def blocking_failures(self) -> list[CheckResult]:
        blocking_ids = {s.id for s in CHECKS if s.severity is CheckSeverity.BLOCKING}
        return [r for r in self.results if r.id in blocking_ids and r.status is CheckStatus.FAIL]

    @property
    def freezable(self) -> bool:
        """Return whether the results contain no blocking failures."""
        return not self.blocking_failures

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        verdict = "FREEZABLE" if self.freezable else "BLOCKED"
        return f"[{self.episode_id}] {verdict} ({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _ok(spec_id: str, detail: str = "") -> CheckResult:
    return CheckResult(spec_id, CheckStatus.PASS, detail)


def _fail(spec_id: str, detail: str) -> CheckResult:
    return CheckResult(spec_id, CheckStatus.FAIL, detail)


def _skip(spec_id: str, detail: str) -> CheckResult:
    return CheckResult(spec_id, CheckStatus.SKIPPED, detail)


def _pending(spec_id: str, detail: str) -> CheckResult:
    return CheckResult(spec_id, CheckStatus.PENDING_IMPL, detail)


def _is_sorted(values: list[float]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def _nearest_sample_dt(rec: EpisodeRecord, ts: float) -> Optional[float]:
    if not rec.trajectory:
        return None
    return min(abs(s.ts - ts) for s in rec.trajectory)


# --------------------------------------------------------------------------- #
# Data-tier checks (implemented)                                               #
# --------------------------------------------------------------------------- #
def _check_timestamps_monotonic(rec: EpisodeRecord) -> CheckResult:
    cid = "timestamps_monotonic"
    if not _is_sorted([s.ts for s in rec.trajectory]):
        return _fail(cid, "trajectory timestamps are not non-decreasing")
    if not _is_sorted([j.ts for j in rec.judge_timeline]):
        return _fail(cid, "judge_timeline timestamps are not non-decreasing")
    if rec.ts_end < rec.ts_start:
        return _fail(cid, "ts_end < ts_start")
    return _ok(cid)


def _check_calibration_present(rec: EpisodeRecord) -> CheckResult:
    cid = "calibration_present"
    if not rec.cameras:
        return _fail(cid, "no cameras recorded; perception checks impossible")
    bad = [c.name for c in rec.cameras
           if len(c.intrinsics) != 3 or len(c.extrinsics) != 4]
    if bad:
        return _fail(cid, f"malformed intrinsics/extrinsics for: {bad}")
    return _ok(cid, f"{len(rec.cameras)} camera(s) calibrated")


def _check_first_fail_present(rec: EpisodeRecord) -> CheckResult:
    cid = "first_fail_present"
    if rec.recovery is None:
        return _skip(cid, "no recovery invocation in this episode")
    if rec.first_fail_ts is None:
        return _fail(cid, "recovery ran but first_fail_ts is missing")
    if not (rec.ts_start <= rec.first_fail_ts <= rec.ts_end):
        return _fail(cid, "first_fail_ts outside [ts_start, ts_end]")
    return _ok(cid)


def _check_rewind_target_before_fail(rec: EpisodeRecord) -> CheckResult:
    cid = "rewind_target_before_fail"
    if rec.recovery is None:
        return _skip(cid, "no recovery invocation")
    rt = rec.recovery.rewind_target_ts
    if rt is None:
        return _skip(cid, "no rewind_target_ts recorded")
    if rec.first_fail_ts is None:
        return _fail(cid, "cannot compare: first_fail_ts missing")
    if rt >= rec.first_fail_ts:
        return _fail(cid, f"rewind target ts {rt} is not strictly before first fail {rec.first_fail_ts}")
    return _ok(cid)


def _check_rewind_target_in_buffer(rec: EpisodeRecord) -> CheckResult:
    """The rewind target must correspond to a physically-visited sample, so it is
    reachable by construction. We verify the target timestamp has a nearby
    trajectory sample (within 50 ms)."""
    cid = "rewind_target_in_buffer"
    if rec.recovery is None:
        return _skip(cid, "no recovery invocation")
    rt = rec.recovery.rewind_target_ts
    if rt is None:
        return _skip(cid, "no rewind_target_ts recorded")
    dt = _nearest_sample_dt(rec, rt)
    if dt is None:
        return _fail(cid, "trajectory ring buffer is empty")
    if dt > 0.05:
        return _fail(cid, f"nearest trajectory sample is {dt*1000:.0f} ms from rewind target")
    return _ok(cid, f"nearest sample within {dt*1000:.0f} ms")


def _check_grasp_in_workspace(rec: EpisodeRecord) -> CheckResult:
    """Cheap data-tier sanity bound on the perceived grasp (full reachability is
    a geometry-tier check). Default FR3 tabletop workspace box; override via
    ``metadata['workspace_aabb'] = [[xmin,ymin,zmin],[xmax,ymax,zmax]]``."""
    cid = "grasp_in_workspace"
    if rec.recovery is None or rec.recovery.perceived_grasp is None:
        return _skip(cid, "no perceived grasp")
    aabb = rec.metadata.get("workspace_aabb") or [[0.2, -0.5, 0.0], [0.8, 0.5, 0.6]]
    lo, hi = aabb[0], aabb[1]
    p = rec.recovery.perceived_grasp.position
    inside = all(lo[i] <= p[i] <= hi[i] for i in range(3))
    if not inside:
        return _fail(cid, f"grasp {p} outside workspace aabb {aabb}")
    return _ok(cid)


def _check_frames_cover_keyframes(rec: EpisodeRecord) -> CheckResult:
    """There must be at least one RGB frame near the first-fail moment and near
    the rewind target, otherwise reflection / replay has no visual evidence."""
    cid = "frames_cover_keyframes"
    if rec.recovery is None:
        return _skip(cid, "no recovery invocation")
    if not rec.frames:
        return _fail(cid, "no frames recorded")

    def has_rgb_near(ts: Optional[float]) -> bool:
        if ts is None:
            return True
        return any(f.rgb_path and abs(f.ts - ts) <= 0.1 for f in rec.frames)

    missing = []
    if not has_rgb_near(rec.first_fail_ts):
        missing.append("first_fail_ts")
    if not has_rgb_near(rec.recovery.rewind_target_ts):
        missing.append("rewind_target_ts")
    if missing:
        return _fail(cid, f"no RGB frame within 100 ms of: {missing}")
    return _ok(cid)


# --------------------------------------------------------------------------- #
# Geometry-tier checks (real when a GeometryValidator is injected, else pending)#
# --------------------------------------------------------------------------- #
def _from_geo(cid: str, geo: GeoCheck) -> CheckResult:
    """Map a backend :class:`GeoCheck` to a :class:`CheckResult`."""
    if geo.ok is True:
        return _ok(cid, geo.detail)
    if geo.ok is False:
        return _fail(cid, geo.detail)
    return _pending(cid, geo.detail)


def _check_grasp_reachability(rec: EpisodeRecord, geometry: Optional[GeometryValidator]) -> CheckResult:
    cid = "grasp_reachability"
    if rec.recovery is None or rec.recovery.perceived_grasp is None:
        return _skip(cid, "no perceived grasp")
    if geometry is None:
        return _pending(cid, "needs pyroki/curobo IK on recorded grasp + pre-grasp pose")
    return _from_geo(cid, geometry.grasp_reachable(rec))


def _check_rewind_path_collision_free(rec: EpisodeRecord, geometry: Optional[GeometryValidator]) -> CheckResult:
    cid = "rewind_path_collision_free"
    if rec.recovery is None or rec.recovery.rewind_target_q is None:
        return _skip(cid, "no rewind target joint state")
    if geometry is None:
        return _pending(cid, "needs curobo collision plan on recorded depth point cloud")
    return _from_geo(cid, geometry.rewind_path_collision_free(rec))


def _check_perception_determinism(rec: EpisodeRecord, geometry: Optional[GeometryValidator]) -> CheckResult:
    cid = "perception_determinism"
    if rec.recovery is None or rec.recovery.perceived_grasp is None:
        return _skip(cid, "no perceived grasp")
    if geometry is None:
        return _pending(cid, "needs re-running perception twice on recorded RGB-D and comparing within tol")
    return _from_geo(cid, geometry.perception_repeatable(rec))


# --------------------------------------------------------------------------- #
# The checklist registry                                                       #
# --------------------------------------------------------------------------- #
CHECKS: list[CheckSpec] = [
    CheckSpec("timestamps_monotonic", "Timestamps monotonic & consistent",
              CheckTier.DATA, CheckSeverity.BLOCKING,
              "Trajectory and judge streams are non-decreasing and ts_end>=ts_start.",
              _check_timestamps_monotonic),
    CheckSpec("calibration_present", "Camera calibration present & well-formed",
              CheckTier.DATA, CheckSeverity.BLOCKING,
              "Every camera carries a 3x3 intrinsics and 4x4 world_from_camera extrinsics.",
              _check_calibration_present),
    CheckSpec("first_fail_present", "First-fail timestamp present & in range",
              CheckTier.DATA, CheckSeverity.BLOCKING,
              "If recovery ran, first_fail_ts exists within the episode window.",
              _check_first_fail_present),
    CheckSpec("rewind_target_before_fail", "Rewind target precedes first fail",
              CheckTier.DATA, CheckSeverity.BLOCKING,
              "Rewind target timestamp is strictly before the first detected failure.",
              _check_rewind_target_before_fail),
    CheckSpec("rewind_target_in_buffer", "Rewind target is a visited state",
              CheckTier.DATA, CheckSeverity.BLOCKING,
              "Rewind target maps to an actually-recorded trajectory sample (<=50 ms).",
              _check_rewind_target_in_buffer),
    CheckSpec("frames_cover_keyframes", "RGB frames cover key moments",
              CheckTier.DATA, CheckSeverity.BLOCKING,
              "An RGB frame exists near first-fail and near the rewind target.",
              _check_frames_cover_keyframes),
    CheckSpec("grasp_in_workspace", "Perceived grasp within workspace box",
              CheckTier.DATA, CheckSeverity.WARNING,
              "Cheap AABB bound on the grasp position before full IK reachability.",
              _check_grasp_in_workspace),
    CheckSpec("grasp_reachability", "Grasp + pre-grasp IK feasible",
              CheckTier.GEOMETRY, CheckSeverity.BLOCKING,
              "IK converges within joint limits for grasp and pre-grasp poses (no execution).",
              _check_grasp_reachability, needs_geometry=True),
    CheckSpec("rewind_path_collision_free", "Rewind path collision-free",
              CheckTier.GEOMETRY, CheckSeverity.BLOCKING,
              "A collision-free joint path exists from failed state to rewind target.",
              _check_rewind_path_collision_free, needs_geometry=True),
    CheckSpec("perception_determinism", "Perception is reproducible",
              CheckTier.GEOMETRY, CheckSeverity.WARNING,
              "Re-running perception on recorded RGB-D yields a consistent grasp within tolerance.",
              _check_perception_determinism, needs_geometry=True),
]


def run_checks(
    rec: EpisodeRecord,
    *,
    only_tier: Optional[CheckTier] = None,
    geometry: Optional[GeometryValidator] = None,
) -> ValidationReport:
    """Check an episode record, optionally restricted to one tier.

    The optional geometry backend evaluates geometry-tier checks. Checks that
    cannot run without it report PENDING_IMPL."""
    report = ValidationReport(episode_id=rec.episode_id)
    for spec in CHECKS:
        if only_tier is not None and spec.tier is not only_tier:
            continue
        try:
            result = spec.fn(rec, geometry) if spec.needs_geometry else spec.fn(rec)
            report.results.append(result)
        except Exception as e:  # a buggy check must not crash the gate
            report.results.append(_fail(spec.id, f"check raised {type(e).__name__}: {e}"))
    return report


def checklist() -> list[dict]:
    """Render check identifiers, tiers, severities and descriptions."""
    return [
        {
            "id": s.id,
            "title": s.title,
            "tier": s.tier.value,
            "severity": s.severity.value,
            "description": s.description,
        }
        for s in CHECKS
    ]
