"""Isolated selected-mask cloud continuity candidate. No I/O/network/GPU at import.
New uncalibrated general hyperparameters: radius = 3 * median(nonzero NN),
unique component containing strictly more than half the original points.
Not a semantic selector, not a learned parameter, and never changes input z.
"""
import copy
import hashlib
import numpy as np
SCHEMA = 'common-cloud-continuity/v1'
RADIUS_NN_MULTIPLIER = 3.0
MAJORITY_FRACTION = 0.5

def select_component(cloud):
    pc = np.asarray(cloud, dtype=np.float64)
    if pc.ndim != 2 or pc.shape[1] != 3 or (not np.isfinite(pc).all()):
        raise ValueError('existing geometry requires finite Nx3 cloud')
    evidence = dict(schema=SCHEMA, input_count=len(pc), input_sha256_float64=hashlib.sha256(pc.tobytes()).hexdigest(), radius_nn_multiplier=RADIUS_NN_MULTIPLIER, majority_fraction=MAJORITY_FRACTION, parameter_status='NEW_UNCALIBRATED_GENERAL_HYPERPARAMETERS', selected=False, selection_source='strict_majority_of_original_selected_mask_cloud', kept_indices=list(range(len(pc))), component_labels=None, z_policy='unchanged caller seed/effective z', object_identity_used=False)

    def unchanged(reason):
        evidence.update(status='UNKNOWN_ORIGINAL_CLOUD', reason=reason, retained_count=len(pc))
        return (pc.copy(), evidence)
    if len(pc) < 3:
        return unchanged('existing_geometry_will_handle_insufficient_points')
    unique, inverse = np.unique(pc, axis=0, return_inverse=True)
    evidence['unique_point_count'] = len(unique)
    if len(unique) < 2:
        return unchanged('no_nonzero_neighbor_distance')
    dist = np.linalg.norm(unique[:, None, :] - unique[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)[inverse]
    median = float(np.median(nearest))
    radius = RADIUS_NN_MULTIPLIER * median
    if not np.isfinite(radius) or radius <= 0:
        return unchanged('no_finite_positive_neighbor_scale')
    links = dist <= radius
    labels = np.full(len(unique), -1, dtype=int)
    next_label = 0
    for start in range(len(unique)):
        if labels[start] != -1:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            node = stack.pop()
            unvisited = np.flatnonzero(links[node] & (labels == -1))
            labels[unvisited] = next_label
            stack.extend(unvisited.tolist())
        next_label += 1
    point_labels = labels[inverse]
    sizes = np.bincount(point_labels, minlength=next_label)
    eligible = np.flatnonzero(sizes > MAJORITY_FRACTION * len(pc))
    evidence.update(nonzero_nn_median_m=median, graph_radius_m=radius, component_sizes_original_points=sizes.tolist(), component_labels=point_labels.tolist())
    if len(eligible) != 1:
        return unchanged('no_unique_strict_majority_component')
    selected = int(eligible[0])
    kept = np.flatnonzero(point_labels == selected)
    evidence.update(status='SELECTED_MAJORITY_COMPONENT', selected=True, selected_component=selected, kept_indices=kept.tolist(), retained_count=len(kept), removed_count=len(pc) - len(kept), changed=len(kept) != len(pc))
    return (pc[kept].copy(), evidence)

def build_geometry(cloud, *, contact_tcp_z, max_opening_m, geometry_module):
    retained, evidence = select_component(cloud)
    candidates, geometry = geometry_module.build_candidates(retained, contact_tcp_z=contact_tcp_z, max_opening_m=max_opening_m)
    return (candidates, dict(continuity=evidence, geometry=geometry))

def install(backend_class):
    """Wrap the named existing family construction seam; no consumer/loader edits."""
    old = backend_class.prepare_common_geometry_family
    if getattr(old, '_common_cloud_continuity', False):
        raise RuntimeError('continuity already installed')

    def prepare(self, *, cloud, seed_z_m, max_opening_m, object_key, family_id):
        retained, evidence = select_component(cloud)
        pool, family = old(self, cloud=retained, seed_z_m=seed_z_m, max_opening_m=max_opening_m, object_key=object_key, family_id=family_id)
        family['cloud_continuity'] = copy.deepcopy(evidence)
        self._common_geometry_family['cloud_continuity'] = copy.deepcopy(evidence)
        self._row3_last_parameter_event['cloud_continuity'] = copy.deepcopy(evidence)
        return (pool, family)
    prepare._common_cloud_continuity = True
    backend_class.prepare_common_geometry_family = prepare
    return dict(installed_isolated_candidate=True, schema=SCHEMA, z_modified=False)
