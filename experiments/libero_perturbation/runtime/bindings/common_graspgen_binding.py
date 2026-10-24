"""Isolated common selected-mask/depth to native geometry binding.
Existing selector preserved; GraspGen class used only for selected cloud acquisition.
P is read once in the family entry; no GraspGen candidate generation.
"""
import functools
import inspect

def configure_inner(inner):
    original = inner.install_candidate_presentation

    @functools.wraps(original)
    def install():
        original()
        import carryon.main as mm
        v2 = inspect.getclosurevars(mm.main).nonlocals['mask_main']
        v1 = inspect.getclosurevars(v2).nonlocals['selection_main']
        cells = inspect.getclosurevars(v1).nonlocals
        bind(cells['AllCandidateBackend'], cells['AllCandidateSelector'], inner.selection_module)
    inner.install_candidate_presentation = install

def bind(backend_class, selector_class, selection_module):
    import hashlib
    import numpy as np
    from carryon.recovery.geometry import GraspGenDetector
    assert backend_class.perceive_grasps.__name__ == 'perceive_grasps_v7'

    def selector_segmenter(selector):
        for detector in list(getattr(selector, '_row3_detector_sources', [])):
            segmenter = getattr(detector, 'segmenter', None)
            if segmenter is not None:
                return segmenter
        raise RuntimeError('all-candidate SAM3 segmenter binding unavailable')

    def perceive_grasps_common_v1(self, object_hint: str, *, top_k: int=15, source: str='topdown', target_xy=None, target_px=None, grounding=None, sam_text_prompts=None):
        selector = selection_module.memory_module._selector_for_backend(self)
        key = selection_module.memory_module.normalize_key(selector.memory.target_name_from_instruction(object_hint) if selector is not None else object_hint)
        self._last_grasp_object_key = key or None
        if not isinstance(selector, selector_class):
            self.last_perceive_reason = 'all-candidate selector V2 unavailable'
            self._mark_selection_fail_closed(self.last_perceive_reason)
            return []
        if bool(getattr(self, '_row3_selection_fail_closed', False)):
            self.last_perceive_reason = str(getattr(self, '_row3_selection_fail_reason', 'selection fail-closed'))
            return []
        self.last_perceive_debug = {}
        inputs = self._fetch_perception_inputs()
        if inputs is None:
            self._mark_selection_fail_closed(getattr(self, 'last_perceive_reason', 'perception input fetch failed'))
            return []
        rgb_jpeg, depth_m, intrinsics = inputs
        self._last_scene_perception = (depth_m, intrinsics)
        selector._row3_exclude_masks = list(getattr(self, 'identity_rejected', None) or [])
        selected = selector.select_mask(object_hint, rgb_jpeg, selector_segmenter(selector), prompt='object', scene_bundle_identity=getattr(self, '_last_scene_bundle_identity', None))
        if selected is None:
            self.last_perceive_reason = selector.last_error or 'all-candidate selection failed closed'
            self.last_perceive_debug = {'all_candidate_reground': dict(selector.last_evidence or {}), 'selected_mask_injected_into_graspgen': False}
            self._mark_selection_fail_closed(self.last_perceive_reason)
            return []
        from pathlib import Path
        import json
        from common_perception_adapter import prepare_detector_input
        detector = getattr(self, 'detector', None)
        if not isinstance(detector, GraspGenDetector) or not callable(getattr(detector, '_object_point_cloud_world', None)):
            self.last_perceive_reason = 'selected-cloud producer unavailable'
            self._mark_selection_fail_closed(self.last_perceive_reason)
            return []
        bound_intrinsics = dict(intrinsics)
        bound_intrinsics.pop('segmap', None)
        bound_intrinsics['mask'] = np.asarray(selected, dtype=np.int32)
        try:
            prepared = prepare_detector_input(detector, archive_root=self._common_archive_root, rgb=rgb_jpeg, depth_m=depth_m, intrinsics=bound_intrinsics, object_hint=object_hint, target_xy=target_xy, target_px=target_px, grounding=grounding, sam_text_prompts=sam_text_prompts, scene_reference=getattr(self, 'scene_reference', None), exclude_masks=list(getattr(self, 'identity_rejected', None) or []), camera_source={'camera_name': self.cfg.grasp_camera, 'scene_bundle_identity': getattr(self, '_last_scene_bundle_identity', None)})
            grasps, family = self.prepare_common_geometry_family(cloud=prepared.cloud, seed_z_m=prepared.seed_z_m, max_opening_m=float(self.cfg.gripper_open_width), object_key=key, family_id=prepared.evidence['prepare_id'])
            directory = Path(prepared.evidence['archive']['dir'])
            with (directory / 'SELECTED_MASK.npy').open('xb') as f:
                np.save(f, bound_intrinsics['mask'], allow_pickle=False)
            (directory / 'CURRENT_RGB.bin').write_bytes(rgb_jpeg)
            evidence = dict(selector.last_evidence or {})
            evidence.update({'execution_status': 'COMMON_GEOMETRY_RETURNED', 'selected_mask_injected_into_graspgen': False, 'graspgen_called': False, 'selected_mask_injected_into_common_cloud': True, 'generator': 'common_grasp_geometry/v2', 'candidate_count': len(grasps), 'generator_input_mask_sha256': hashlib.sha256(bound_intrinsics['mask'].tobytes()).hexdigest(), 'common_perception': prepared.evidence, 'common_family': family})
            (directory / 'FAMILY.json').write_text(json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False), encoding='utf8')
            selector.last_evidence = evidence
            selector._prepare_image_directory()
            self.last_perceive_debug = {'all_candidate_reground': evidence, 'common_geometry': family, 'common_perception': prepared.evidence}
            if not grasps:
                self.last_perceive_reason = 'common geometry returned no candidates'
            return grasps
        except Exception as exc:
            self.last_perceive_reason = f'common selected-cloud preparation failed: {type(exc).__name__}: {exc}'
            self._mark_selection_fail_closed(self.last_perceive_reason)
            self.last_perceive_debug = {'graspgen_called': False, 'common_geometry_failed': self.last_perceive_reason}
            return []
    backend_class.perceive_grasps = perceive_grasps_common_v1
