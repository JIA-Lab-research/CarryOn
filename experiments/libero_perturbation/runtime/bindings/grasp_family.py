"""Isolated named-class P family entry; no install or I/O at import time."""
from runtime_paths import bound_path, resource
import copy
import hashlib
import math
from pathlib import Path
from common_grasp_geometry import build_candidates

def install(backend_class, exact_executor, *, archive_root):
    old = backend_class.goto_grasp_topdown_family
    source = Path(old.__code__.co_filename)
    base = old.__globals__['memory_module']
    from carryon.recovery.schema import GraspProposal as proposal_cls
    if 'common-native-exact-family/draft-v1' not in exact_executor.__code__.co_consts:
        raise RuntimeError('exact native executor not installed at named binding')
    if exact_executor is old:
        raise RuntimeError('exact executor must not be the old P consumer')

    def prepare_common_geometry_family(self, *, cloud, seed_z_m, max_opening_m, object_key, family_id):
        key = base.normalize_key(object_key)
        if not key or key != object_key or (not family_id):
            raise ValueError('explicit normalized object key and family id required')
        previous = getattr(self, '_common_geometry_family', None)
        if previous and previous['family_id'] == family_id:
            raise RuntimeError('family already prepared; do not reload P for retry')
        memory = base.load_parameter_memory()
        row = memory.get(key)
        z = float(seed_z_m if row is None else row['corrected_z_m'])
        if not math.isfinite(z):
            raise ValueError('nonfinite effective z')
        candidates, geometry = build_candidates(cloud, contact_tcp_z=z, max_opening_m=max_opening_m)
        pool = [proposal_cls(position=list(c['position']), quaternion=list(c['quaternion_wxyz']), score=0.0, source='common_geometry_v2/' + c['axis_choice'], predicted_width_m=c['predicted_width_m']) for c in candidates]
        from selection_origin_binding import proposal
        record = {'family_id': family_id, 'object_key': key, 'seed_z_m': float(seed_z_m), 'effective_z_m': z, 'P_applied': row is not None, 'P_record': copy.deepcopy(row), 'P_read_count': 1, 'geometry': geometry, 'candidates': copy.deepcopy(candidates), 'proposal_snapshots': [proposal(g) for g in pool]}
        self._common_geometry_family = record
        self._row3_last_parameter_event = copy.deepcopy(record)
        return (pool, copy.deepcopy(record))

    def goto_grasp_topdown_family(self, frozen_target, fallback_grasps, **kwargs):
        from selection_origin_binding import proposal
        family = getattr(self, '_common_geometry_family', None)
        if not family:
            raise RuntimeError('common family must be prepared before execution')
        snapshot = proposal(frozen_target)
        if snapshot not in family['proposal_snapshots']:
            raise RuntimeError('candidate changed after common family freeze')
        return exact_executor(self, frozen_target, fallback_grasps, **kwargs)
    backend_class._common_archive_root = Path(archive_root)
    backend_class.prepare_common_geometry_family = prepare_common_geometry_family
    backend_class.goto_grasp_topdown_family = goto_grasp_topdown_family
    from common_cloud_continuity import install as install_continuity
    install_continuity(backend_class)
    return {'installed_isolated_candidate': True, 'P_resolution': 'once_before_family', 'second_direction_z': 'same_frozen_z', 'old_loader_mutated': False}
