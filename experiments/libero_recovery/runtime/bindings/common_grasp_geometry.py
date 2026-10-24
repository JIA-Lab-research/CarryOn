"""Offline pure geometry; no detector/factory/runtime hooks or external calls.

World XYZ metres; quaternion WXYZ maps hand to world. Hand +Y opens;
hand +Z approaches world -Z. Centre is the midpoint of both observed PCA bounds. Caller owns cloud selection,
effective contact-TCP z, calibrated model opening limit and runtime safety.
"""
import hashlib
import math
import struct
import sys
SCHEMA = 'common-grasp-geometry/v2'
NUMERIC_REL_TOL = 64 * sys.float_info.epsilon

def _finite_vector(value, length):
    result = tuple((float(v) for v in value))
    if len(result) != length or not all((math.isfinite(v) for v in result)):
        raise ValueError('invalid vector')
    return result

def _canonical_axis(axis):
    x, y = axis
    if x < 0.0 or (x == 0.0 and y < 0.0):
        x, y = (-x, -y)
    return (x, y)

def _topdown_quaternion(axis):
    x, y = axis
    angle = 0.5 * math.atan2(x, -y)
    return [0.0, math.cos(angle), math.sin(angle), 0.0]

def _observed_envelope_axes(pc, ambiguous):
    """Return one observed support frame, or a reason to retain the PCA axes."""
    if ambiguous:
        return (None, {'envelope_status': 'PCA_FALLBACK', 'fallback_reason': 'NUMERICALLY_ISOTROPIC_NO_UNIQUE_FRAME'})

    def fallback(reason):
        return (None, {'envelope_status': 'PCA_FALLBACK', 'fallback_reason': reason})
    ox, oy = min(((p[0], p[1]) for p in pc))
    scale = max((max(abs(p[0] - ox), abs(p[1] - oy)) for p in pc))
    if not math.isfinite(scale) or scale <= 0:
        return fallback('NO_FINITE_POSITIVE_ENVELOPE')
    points = sorted(set((((p[0] - ox) / scale, (p[1] - oy) / scale) for p in pc)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = ([], [])
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return fallback('NUMERICALLY_DEGENERATE_OBSERVED_HULL')
    rectangles = []
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        dx, dy = (b[0] - a[0], b[1] - a[1])
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        u = _canonical_axis((dx / length, dy / length))
        v = _canonical_axis((-u[1], u[0]))
        bounds = []
        for axis in (u, v):
            projections = [p[0] * axis[0] + p[1] * axis[1] for p in hull]
            bounds.append((min(projections), max(projections)))
        area = (bounds[0][1] - bounds[0][0]) * (bounds[1][1] - bounds[1][0])
        if math.isfinite(area) and area > 0:
            rectangles.append((area, (u, v), tuple(bounds)))
    if not rectangles:
        return fallback('NO_FINITE_POSITIVE_ENVELOPE')
    minimum = min((row[0] for row in rectangles))
    ties = [row for row in rectangles if row[0] - minimum <= NUMERIC_REL_TOL * max(minimum, sys.float_info.min)]
    ref = ties[0][1][0]
    for _, axes, _ in ties:
        u = axes[0]
        same_frame = max(abs(u[0] * ref[0] + u[1] * ref[1]), abs(u[0] * ref[1] - u[1] * ref[0])) >= 1 - NUMERIC_REL_TOL
        if not same_frame:
            return fallback('NUMERICALLY_TIED_DISTINCT_MIN_AREA_FRAMES')
    _, axes, _ = min(ties, key=lambda row: (row[1], row[2]))
    return (axes, {'envelope_status': 'MIN_AREA_OBSERVED_XY_RECTANGLE', 'hull_vertex_count': len(hull), 'minimum_observed_area_normalized': minimum, 'envelope_normalization_scale_m': scale})

def build_candidates(cloud, *, contact_tcp_z, max_opening_m):
    """Return envelope-axis candidates with the unchanged PCA-bounds centre.

    Exact XYZ triples required; >=3 finite points, numerically full-rank XY.
    Isotropic covariance retains deterministic world X/Y with an ambiguity flag. Each axis midpoint is (observed min+max)/2, transformed to world
    once for both directions. Axes ordered by observed span; no other centre
    algorithm, object branch, percentile, pose correction or z rule. Only axes
    use the minimum-area observed XY rectangle, with numeric PCA fallbacks.
    The retained PCA centre can still depend on observed point density. Rotation
    equivariance of centre/axes is not promised when PCA is isotropic.
    Machine-precision rank/tie tolerances are not physical tuning thresholds.
    Input z passes through unchanged; opening estimates do not certify full
    object fit, contact-band fit, collision freedom or runtime permission.
    """
    ev = {'schema': SCHEMA, 'reference': 'contact-TCP', 'frame': 'world', 'opening_axis_hand': '+Y', 'approach_axis_hand': '+Z', 'approach_axis_world': [0.0, 0.0, -1.0], 'width_scope': 'selected observed cloud only; partial visibility may underestimate object extent', 'collision_feasibility': 'UNKNOWN', 'runtime_installed': False}
    try:
        z = float(contact_tcp_z)
        opening = float(max_opening_m)
        if not math.isfinite(z) or not math.isfinite(opening) or opening <= 0:
            raise ValueError('invalid scalar')
    except (ValueError, TypeError, OverflowError):
        ev['failure'] = 'INVALID_CALLER_GEOMETRY'
        return ([], ev)
    try:
        pc = tuple((_finite_vector(point, 3) for point in cloud))
    except (ValueError, TypeError, OverflowError):
        ev['failure'] = 'INVALID_CLOUD'
        return ([], ev)
    ev.update(n=len(pc), contact_tcp_z=z, max_opening_m=opening)
    if len(pc) < 3:
        ev['failure'] = 'INSUFFICIENT_POINTS'
        return ([], ev)
    ev['cloud_sha256_float64_xyz'] = hashlib.sha256(b''.join((struct.pack('<ddd', *p) for p in pc))).hexdigest()
    origin = pc[0]
    try:
        mx = math.fsum((p[0] - origin[0] for p in pc)) / len(pc)
        my = math.fsum((p[1] - origin[1] for p in pc)) / len(pc)
        xy = tuple(((p[0] - origin[0] - mx, p[1] - origin[1] - my) for p in pc))
        scale = max((max(abs(x), abs(y)) for x, y in xy))
        if not math.isfinite(scale) or scale == 0:
            raise ValueError('zero or non-finite XY scale')
        normalized = tuple(((x / scale, y / scale) for x, y in xy))
        a = math.fsum((x * x for x, y in normalized)) / len(pc)
        b = math.fsum((x * y for x, y in normalized)) / len(pc)
        d = math.fsum((y * y for x, y in normalized)) / len(pc)
        gap = math.hypot(a - d, 2 * b)
        lam_max = (a + d + gap) / 2
        lam_min = (a + d - gap) / 2
        if lam_min <= NUMERIC_REL_TOL * lam_max:
            ev['failure'] = 'DEGENERATE_XY_RANK'
            return ([], ev)
        ambiguous = gap <= NUMERIC_REL_TOL * (a + d)
        angle = 0.0 if ambiguous else 0.5 * math.atan2(2 * b, a - d)
        major = _canonical_axis((math.cos(angle), math.sin(angle)))
        minor = _canonical_axis((-major[1], major[0]))
        centered_bounds = []
        for axis in (major, minor):
            projections = tuple((x * axis[0] + y * axis[1] for x, y in xy))
            centered_bounds.append((min(projections), max(projections)))
        mid_major, mid_minor = (lo + (hi - lo) / 2 for lo, hi in centered_bounds)
        center = (origin[0] + mx + major[0] * mid_major + minor[0] * mid_minor, origin[1] + my + major[1] * mid_major + minor[1] * mid_minor)
        if not all((math.isfinite(v) for v in center)):
            raise ValueError('non-finite computed centre')
        ev.update(center_xy=list(center), pca_centered_bounds_major_minor_m=[list(v) for v in centered_bounds], pca_midpoints_major_minor_m=[mid_major, mid_minor])
        envelope_axes, envelope_ev = _observed_envelope_axes(pc, ambiguous)
        ev.update(envelope_ev)
        axes = (('pca_minor', minor), ('pca_major', major))
        if envelope_axes is not None:
            axes = tuple((('observed_xy_min_area_rectangle', axis) for axis in envelope_axes))
        rows = []
        for name, axis in axes:
            projections = tuple(((p[0] - center[0]) * axis[0] + (p[1] - center[1]) * axis[1] for p in pc))
            lo, hi = (min(projections), max(projections))
            span = hi - lo
            required = 2 * max(abs(lo), abs(hi))
            if not all((math.isfinite(v) for v in (lo, hi, span, required))):
                raise ValueError('non-finite projection')
            rows.append((span, name, axis, lo, hi, required))
    except (ValueError, OverflowError):
        ev['failure'] = 'DEGENERATE_OR_UNREPRESENTABLE_XY'
        return ([], ev)
    rows.sort(key=(lambda row: (row[0], row[2])) if envelope_axes is not None else lambda row: row[0])
    candidates = []
    for index, (span, axis_source, axis, lo, hi, required) in enumerate(rows):
        candidates.append({'axis_choice': 'short' if index == 0 else 'long', 'axis_source': axis_source, 'position': [center[0], center[1], z], 'quaternion_wxyz': _topdown_quaternion(axis), 'opening_axis_world': [axis[0], axis[1], 0.0], 'predicted_width_m': span, 'projected_bounds_relative_center_m': [lo, hi], 'required_symmetric_opening_m': required, 'max_opening_m': opening, 'opening_limit_status': 'PASS_ESTIMATE' if required <= opening else 'INFEASIBLE_KNOWN_CLOUD', 'collision_feasibility': 'UNKNOWN'})
    ev.update(axis_rule='orthogonal observed XY minimum-area rectangle; short/long ordered by observed span' if envelope_axes is not None else 'orthogonal XY PCA frame, short/long ordered by observed span', geometry_revision='observed-envelope-axes-fixed-pca-center/v1', covariance_isotropic_at_numeric_tolerance=ambiguous, numeric_relative_tolerance=NUMERIC_REL_TOL, xyz_frozen_for_both=True, max_candidate_count=2, center_algorithm='midpoint of observed min/max on both XY PCA axes, transformed once to world', z_rule='caller supplied; no compensation or retry delta')
    return (candidates, ev)
