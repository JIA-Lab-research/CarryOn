"""Passive same-attempt evidence. No additional FK, perception or robot calls."""
import copy
import functools
import hashlib
import json
import math
import time
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clone(x):
    return copy.deepcopy(x)

def unknown(reason):
    return {'status': 'unknown', 'reason': reason}

def tcp_pose(pose, offset):
    import numpy as np
    from scipy.spatial.transform import Rotation
    hand, q = pose
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return (np.asarray(hand, dtype=float) + float(offset) * rot[:, 2]).tolist()

def visible_cloud(backend, candidate):
    """Use the current perception's existing, world-frame GraspGen input only."""
    import numpy as np
    d = getattr(backend, 'last_perceive_debug', {}) or {}
    archive = (d.get('graspgen') or {}).get('raw_archive')
    if not isinstance(archive, dict):
        return unknown('same-perception raw archive unavailable')
    try:
        ref = archive['point_cloud']
        assert sha(ref['path']) == ref['sha256'], 'pointcloud SHA mismatch'
        pc = np.load(ref['path'], allow_pickle=False)
        assert pc.ndim == 2 and pc.shape[1] == 3 and len(pc) and np.isfinite(pc).all()
        reground = d.get('all_candidate_reground') or {}
        return {'status': 'recorded', 'coordinate_frame': 'planner world; existing _object_point_cloud_world transform', 'units': 'm', 'phase': 'perception before this candidate pool was planned', 'source': ref, 'archive_started_wall_ts': archive['started_wall_ts'], 'algorithm_mask_id_not_identity_truth': reground.get('selected_candidate_id'), 'candidate_origin': candidate['original_pool_index'], 'point_count': len(pc), 'xyz_min_m': pc.min(0).tolist(), 'xyz_max_m': pc.max(0).tolist(), 'xyz_median_m': np.median(pc, axis=0).tolist(), 'limitation': 'Visible masked geometry only; not complete object surface, correct identity, contact or correct-height truth.'}
    except Exception as exc:
        return unknown(type(exc).__name__ + ': ' + str(exc))

def install_motion_capture(base):
    original = base._goalset_with_retry
    assert not getattr(original, '__parameter_evidence__', False)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        planner = self.planner
        had_stream = '_stream_joint_path' in self.__dict__
        own_stream = self.__dict__.get('_stream_joint_path')
        stream = self._stream_joint_path
        captures = []
        descent = [None]

        class PlannerView:

            def __getattr__(proxy, name):
                return getattr(planner, name)

            def ee_pose(proxy, q):
                value = planner.ee_pose(q)
                captures.append({'q': list(q), 'pose': clone(value), 'wall_ts': time.time(), 'after_descent': descent[0] is not None})
                return value

        def stream_capture(path, width, **kw):
            start = time.time()
            value = stream(path, width, **kw)
            if kw.get('motion_segment') == 'descent':
                descent[0] = {'planned_q': list(path[-1]), 'actual_q': None if value.final_q is None else list(value.final_q), 'started_wall_ts': start, 'finished_wall_ts': time.time(), 'command_width_m': width, 'motion_ok': bool(value.ok)}
            return value
        self.planner = PlannerView()
        self._stream_joint_path = stream_capture
        try:
            return original(self, *args, **kwargs)
        finally:
            self.planner = planner
            if had_stream:
                self._stream_joint_path = own_stream
            else:
                del self.__dict__['_stream_joint_path']
            debug = getattr(self, 'last_goalset_execution_debug', {}) or {}
            binding = clone(debug.get('executed_selection'))
            if binding is not None:
                ev = {'schema': 'carryon-same-attempt-evidence/v1', 'binding': binding, 'coordinate_frame': 'planner world', 'position_units': 'm', 'orientation': 'Panda hand quaternion wxyz; fixed accepted axes/TCP calibration', 'planned_contact_TCP_xyz_m': binding['executed_candidate']['position'], 'planned_quaternion_wxyz': binding['executed_candidate']['quaternion'], 'predicted_width_m': binding['executed_candidate'].get('predicted_width_m'), 'descent': descent[0] or unknown('physical descent not reached'), 'actual_post_descent': unknown('existing FK call unavailable'), 'additional_robot_or_FK_calls': 0, 'visible_geometry': visible_cloud(self, binding)}
                try:
                    d = descent[0]
                    if d is not None and d['actual_q'] is not None:
                        matches = [r for r in captures if r['after_descent'] and r['q'] == d['actual_q'] and r['pose']]
                        if matches:
                            r = matches[0]
                            tip = tcp_pose(r['pose'], self.cfg.tcp_offset_m)
                            target = ev['planned_contact_TCP_xyz_m']
                            ev['actual_post_descent'] = {'status': 'recorded', 'phase': 'after descent, before close/lift/retreat', 'wall_ts': r['wall_ts'], 'joint_q': r['q'], 'FK_hand_xyz_m': r['pose'][0], 'FK_hand_quaternion_wxyz': r['pose'][1], 'TCP_xyz_m': tip, 'tcp_offset_m': self.cfg.tcp_offset_m, 'signed_actual_minus_target_TCP_xyz_m': [a - b for a, b in zip(tip, target)]}
                except Exception as exc:
                    ev['actual_post_descent'] = unknown(type(exc).__name__ + ': ' + str(exc))
                self.last_topdown_family_debug['parameter_attempt_evidence'] = ev
    wrapped.__parameter_evidence__ = True
    base._goalset_with_retry = wrapped
    return (original, wrapped)

def family(item):
    d = getattr(item, 'data', None) or {}
    if 'fresh_attempts' in d:
        return d['fresh_attempts'][-1].get('topdown_family', {})
    return d.get('topdown_family', {})

def node_at(steps, path):
    parts = [int(x) for x in path.split('.')]
    items = steps
    for i, part in enumerate(parts):
        item = items[part]
        if i + 1 < len(parts):
            items = getattr(item, 'children', [])
    return item

def widths_only(data):
    """Never feed raw bridge XYZ or lift-endpoint height as contact coordinates."""
    allow = {'command_width', 'w_cmd', 'w_meas', 'width_m', 'measured_width_m', 'grasp_contact', 'contact_ok', 'stable_ok', 'trace_ok', 'width_ok', 'ts', 'sample_count', 'delta_width_m', 'all_contact_true'}
    if isinstance(data, list):
        return [widths_only(v) for v in data]
    if not isinstance(data, dict):
        return data
    return {k: widths_only(v) for k, v in data.items() if k in allow or k in {'state_after', 'lift_width_trace', 'samples'}}

def assemble_evidence(backend, steps, attribution):
    view = attribution.failure_view(steps)
    chosen = view['unresolved_nohold']
    path = chosen['goto_path'] if chosen else (view['last_goto'] or {}).get('path')
    if path is None:
        return unknown('no bound grasp operation')
    item = node_at(steps, path)
    target = attribution.physical_target_record(item)
    ev = clone(family(item).get('parameter_attempt_evidence'))
    if ev is None:
        return unknown('same-attempt passive capture absent; historical or incomplete record')
    if target is None or ev['planned_contact_TCP_xyz_m'] != target['target']:
        return unknown('capture and failure-attribution physical target disagree')
    ev['failure_goto_path'] = path
    ev['physical_target_binding'] = target
    ev['close'] = unknown('not reached or absent')
    ev['holding'] = unknown('not reached or absent')
    if chosen:
        close = node_at(steps, chosen['close_path'])
        hold = chosen['assert_item']
        raw = (getattr(hold, 'data', {}) or {}).get('holding_check', {})
        hb = raw.get('executed_selection')
        if not isinstance(hb, dict) or hb != ev['binding']:
            return unknown('holding and planned candidate bindings disagree')
        ev['close'] = {'phase': 'close after this descent', 'ok': bool(close.ok), 'values': widths_only((getattr(close, 'data', {}) or {}).get('gripper', {}))}
        ev['holding'] = {'phase': 'after closure and lift, not contact-target height', 'ok': bool(hold.ok), 'values': widths_only(raw), 'raw_bridge_positions_deliberately_excluded': True}
    ev['memory_application'] = clone(getattr(backend, '_row3_last_parameter_event', None)) or unknown('no application event')
    return ev

def install_attribution_capture():
    import failure_evidence_attribution as attribution
    cls = attribution.FailureEvidenceAttributionBackend
    original = cls.reflect_parameter_memory_at_recovery_end
    assert not getattr(original, '__parameter_evidence__', False)

    @functools.wraps(original)
    def wrapped(self, steps):
        reflector = self.jam_z_reflector
        if reflector is not None:
            reflector.attempt_evidence = assemble_evidence(self, steps, attribution)
        return original(self, steps)
    wrapped.__parameter_evidence__ = True
    cls.reflect_parameter_memory_at_recovery_end = wrapped
    return (original, wrapped)
