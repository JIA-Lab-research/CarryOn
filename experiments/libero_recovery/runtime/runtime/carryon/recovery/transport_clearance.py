"""Isolated common pre-place transport, with explicit path and arrival checks."""
from __future__ import annotations
import json
import math

def _finite_vector(value, count):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('invalid transport vector')
    result = [float(x) for x in value]
    if not all((math.isfinite(x) for x in result)):
        raise ValueError('non-finite transport vector')
    return result

def _quat(value):
    q = _finite_vector(value, 4)
    n = math.sqrt(sum((x * x for x in q)))
    if n <= 0:
        raise ValueError('zero transport quaternion')
    return [x / n for x in q]

def _angle(a, b):
    dot = min(1.0, max(0.0, abs(sum((x * y for x, y in zip(_quat(a), _quat(b)))))))
    return 2.0 * math.acos(dot)

def _limits(backend):
    values = {}
    for name in ('transport_position_tol_m', 'transport_orientation_tol_rad'):
        raw = getattr(backend.cfg, name, None)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or (not math.isfinite(raw)) or (raw <= 0):
            raise ValueError('transport limits unconfigured: ' + name)
        values[name] = float(raw)
    return values

def _contact_pose(backend, pose):
    from .primitives import grasp_local_standoff
    if pose is None:
        raise ValueError('transport FK unavailable')
    hand, quat = pose
    hand = _finite_vector(hand, 3)
    quat = _quat(quat)
    tcp = float(backend.cfg.tcp_offset_m)
    if not math.isfinite(tcp):
        raise ValueError('invalid TCP offset')
    return (_finite_vector(list(grasp_local_standoff(hand, quat, -tcp)), 3), quat)

def _current_contact(backend):
    q = list(backend.current_q())
    return (q, _contact_pose(backend, backend.planner.ee_pose(q)))

def _endpoint(backend, target, reference_quat, limits):
    q, (point, quat) = _current_contact(backend)
    position_error = math.sqrt(sum(((a - b) ** 2 for a, b in zip(point, target))))
    angle_error = _angle(quat, reference_quat)
    return {'ok': position_error <= limits['transport_position_tol_m'] and angle_error <= limits['transport_orientation_tol_rad'], 'measured_q': q, 'measured_world_contact': point, 'measured_quaternion_wxyz': quat, 'position_error_m': position_error, 'orientation_error_rad': angle_error}

def audit_command_path(backend, stream_waypoints, guard):
    """Check clearance along the densified motion trajectory before execution."""
    try:
        if not stream_waypoints:
            raise ValueError('empty final transport stream')
        if not callable(getattr(backend.planner, 'ee_poses', None)):
            raise ValueError('batch transport FK unavailable')
        measured_q = list(backend.current_q())
        qs = [measured_q] + [list(q) for q in stream_waypoints]
        poses = backend.planner.ee_poses(qs)
        if not isinstance(poses, (list, tuple)) or len(poses) != len(qs):
            raise ValueError('transport FK count mismatch')
        contacts = [_contact_pose(backend, pose) for pose in poses]
        points = [x[0] for x in contacts]
        angles = [_angle(x[1], guard['reference_quat']) for x in contacts]
        tol = guard['limits']['transport_position_tol_m']
        phase = guard['phase']
        if phase == 'lateral':
            deviations = [abs(p[2] - guard['target'][2]) for p in points]
        elif phase in ('ascent', 'descent'):
            deviations = [math.hypot(p[0] - guard['target'][0], p[1] - guard['target'][1]) for p in points]
            lower = min(guard['start_contact'][2], guard['target'][2]) - tol
            upper = max(guard['start_contact'][2], guard['target'][2]) + tol
            if any((not lower <= p[2] <= upper for p in points)):
                raise ValueError(phase + ' exceeds frozen vertical interval')
        else:
            raise ValueError('unsupported transport phase')
        if max(deviations) > tol:
            raise ValueError('transport path violates phase position constraint')
        if max(angles) > guard['limits']['transport_orientation_tol_rad']:
            raise ValueError('transport path violates fixed orientation')
        terminal_error = math.sqrt(sum(((a - b) ** 2 for a, b in zip(points[-1], guard['target']))))
        if terminal_error > tol:
            raise ValueError('transport path endpoint outside tolerance')
        return {'ok': True, 'phase': phase, 'checked_command_count': len(stream_waypoints), 'measured_start_included': True, 'min_contact_z_m': min((p[2] for p in points)), 'max_contact_z_m': max((p[2] for p in points)), 'max_phase_error_m': max(deviations), 'max_orientation_error_rad': max(angles), 'terminal_error_m': terminal_error}
    except Exception as exc:
        return {'ok': False, 'phase': guard.get('phase'), 'reason': str(exc), 'checked_before_any_phase_command': True}

def execute_high_transport(backend, original_contact_target):
    """Reach shared source/destination height, then carry and descend."""
    from .primitives import MotionResult, grasp_local_standoff
    review = {'schema': 'common-high-transport/v2', 'phases': [], 'old_path_fallback': False}
    backend._last_high_transport_review = review

    def finish(ok, reason, final_q=None):
        review.update(ok=bool(ok), reason=reason)
        return MotionResult(ok=bool(ok), final_q=final_q, reason=reason + ' | transport_review=' + json.dumps(review, sort_keys=True))
    try:
        if not callable(getattr(backend.planner, 'plan_transport_segment', None)):
            raise ValueError('constrained transport planner unavailable')
        if getattr(backend, '_transport_path_guard', None) is not None:
            raise ValueError('nested transport execution refused')
        limits = _limits(backend)
        original = _finite_vector(original_contact_target, 3)
        q_start, (start, reference_quat) = _current_contact(backend)
        transport_z = max(start[2], original[2])
        ascent = [start[0], start[1], transport_z]
        lateral = [original[0], original[1], transport_z]
        phases = []
        if original[2] > start[2]:
            phases.append(('ascent', ascent))
        phases.extend((('lateral', lateral), ('descent', original)))
        review.update(limits=limits, initial_world_contact=start, reference_quaternion_wxyz=reference_quat, lateral_target_world_contact=lateral, original_preplace_world_contact=original, transport_world_contact_z=transport_z, ascent_target_world_contact=ascent if phases[0][0] == 'ascent' else None)
        grip = float(backend._hold_gripper_width())
        if not math.isfinite(grip) or grip < 0:
            raise ValueError('invalid held close command')
        for phase, target in phases:
            q_start, (measured_start, measured_quat) = _current_contact(backend)
            if phase == 'descent':
                arrival = _endpoint(backend, lateral, reference_quat, limits)
                review['lateral_arrival_before_descent'] = arrival
                if not arrival['ok']:
                    return finish(False, 'measured high-point arrival rejected', arrival['measured_q'])
                q_start = arrival['measured_q']
                measured_start = arrival['measured_world_contact']
            hand = list(grasp_local_standoff(target, reference_quat, float(backend.cfg.tcp_offset_m)))
            path = backend.planner.plan_transport_segment(q_start, hand, reference_quat, phase=phase)
            if not path:
                return finish(False, 'constrained ' + phase + ' planning failed')
            entry = {'phase': phase, 'target_world_contact': target, 'target_world_hand': hand, 'gripper_close_command_m': grip}
            review['phases'].append(entry)
            backend._transport_path_guard = {'phase': phase, 'start_contact': measured_start, 'target': target, 'reference_quat': reference_quat, 'limits': limits, 'entry': entry}
            try:
                motion = backend._stream_joint_path(path, grip, verify=True)
            finally:
                backend._transport_path_guard = None
            entry['motion_ok'] = bool(motion.ok)
            entry['motion_reason'] = motion.reason
            if not motion.ok:
                return finish(False, phase + ' execution rejected', motion.final_q)
            arrival = _endpoint(backend, target, reference_quat, limits)
            entry['measured_arrival'] = arrival
            if not arrival['ok']:
                return finish(False, phase + ' measured arrival rejected', arrival['measured_q'])
        return finish(True, 'constrained high transport completed', arrival['measured_q'])
    except Exception as exc:
        return finish(False, 'high transport failed closed: ' + str(exc))
