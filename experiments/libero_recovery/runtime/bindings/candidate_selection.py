"""Select recovery targets from semantic memory and current visual candidates."""
from __future__ import annotations
from runtime_paths import bound_path, resource
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
REPO = Path(str(bound_path('runtime')))
BASE = REPO / 'logs/recovery'
BASE_ADAPTER = Path(__file__).with_name('semantic_recovery.py')
ALL_CANDIDATE_PROMPT = Path(__file__).with_name('candidate_selection_prompt.txt')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _load_base_adapter():
    if not BASE_ADAPTER.is_file():
        raise RuntimeError('semantic recovery adapter file is missing')
    if not ALL_CANDIDATE_PROMPT.is_file():
        raise RuntimeError('candidate selection prompt file is missing')
    spec = importlib.util.spec_from_file_location('carryon_semantic_recovery_adapter', BASE_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError('semantic recovery adapter import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
memory_module = _load_base_adapter()
ENTRIES = memory_module.ENTRIES
Z_PRIORS_M = memory_module.Z_PRIORS_M
Z_CORRECTED_M = memory_module.Z_CORRECTED_M
PROMPT_BYTES = ALL_CANDIDATE_PROMPT.read_text(encoding='utf-8')

def _threshold_snapshot(segmenter) -> dict:
    config = getattr(segmenter, 'config', None)
    names = ('threshold', 'confidence_threshold', 'retry_threshold', 'mask_threshold')
    return {name: getattr(config, name) for name in names if config is not None and hasattr(config, name)}

def _strict_selection(text: str, count: int) -> tuple[int, str, str]:
    """Accept exactly SELECT + valid id + MEDIUM/HIGH; reject everything else."""
    if not isinstance(text, str) or re.fullmatch('\\s*\\{.*\\}\\s*', text, re.DOTALL) is None:
        raise ValueError('response must be exactly one JSON object')
    payload = json.loads(text)
    expected_keys = {'decision', 'candidate_id', 'confidence', 'evidence'}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError('selection schema keys mismatch')
    decision = payload['decision']
    confidence = payload['confidence']
    evidence = payload['evidence']
    cid = payload['candidate_id']
    if decision not in {'SELECT', 'REFUSE', 'AMBIGUOUS'}:
        raise ValueError('decision is not SELECT|REFUSE|AMBIGUOUS')
    if confidence not in {'LOW', 'MEDIUM', 'HIGH'}:
        raise ValueError('confidence is not LOW|MEDIUM|HIGH')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('evidence must be a non-empty string')
    if decision != 'SELECT':
        raise ValueError(f'VLM_{decision}')
    if confidence == 'LOW':
        raise ValueError('VLM_LOW_CONFIDENCE')
    if isinstance(cid, bool) or not isinstance(cid, int) or (not 1 <= cid <= count):
        raise ValueError('candidate_id missing/out_of_range')
    return (cid, confidence, evidence.strip())

class _FailClosedConfigProxy:

    def __init__(self, base, backend) -> None:
        self._base = base
        self._backend = backend

    def __getattr__(self, name):
        if name == 'handback_on_failure' and bool(getattr(self._backend, '_row3_selection_fail_closed', False)):
            return False
        return getattr(self._base, name)

def install_all_candidate_adapter() -> None:
    if os.environ.get('CARRYON_RECOVERY_PIPELINE') != 'semantic':
        raise RuntimeError('all-candidate adapter requires CARRYON_RECOVERY_PIPELINE=semantic')
    if os.environ.get('CARRYON_SEMANTIC_MEMORY_SNAPSHOT'):
        raise RuntimeError('legacy semantic snapshot must remain disabled')
    if os.environ.get('CARRYON_POSITIONFIX_MEMORY'):
        raise RuntimeError('Position-fix memory is not part of this recovery configuration')
    if os.environ.get('CARRYON_VISUAL_OBJECT_MEMORY_MANIFEST') != str(memory_module.R7):
        raise RuntimeError('Reference registry environment mismatch')
    import cv2
    import numpy as np
    import carryon.main as main_module
    import carryon.recovery as recovery
    import carryon.recovery.integration as integration
    import carryon.recovery.orchestrator as orchestrator
    import carryon.recovery.visual_object_memory as vom
    from carryon.recovery.schema import GraspProposal
    true_main = main_module.main
    base_selector = vom.VisualMemorySomSelector
    base_backend = recovery.GelloRecoveryBackend
    base_orchestrator = orchestrator.RecoveryOrchestrator

    class AllCandidateSelector(base_selector):
        """Entry-hit identity selector; every hit invokes SoM even for K=1."""

        def __init__(self, *args, **kwargs):
            if kwargs.get('semantic_memory') is not None:
                raise RuntimeError('legacy semantic selector must remain disabled')
            super().__init__(*args, **kwargs)
            self._row3_exclude_masks = []
            self._row3_last_object_key = None
            self._row3_forced_count = 0

        @staticmethod
        def _neutral_36px_som(rgb, masks):
            canvas = AllCandidateSelector._rgb_array(rgb).copy()
            mapping = []
            shapes = ('circle', 'square', 'diamond', 'triangle', 'hexagon', 'cross', 'star', 'octagon')
            for candidate_id, raw in enumerate(masks, 1):
                mask = np.asarray(raw, dtype=bool)
                if mask.shape != canvas.shape[:2] or not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                cx = int(round(float(xs.mean())))
                cy = int(round(float(ys.mean())))
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, contours, -1, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.drawContours(canvas, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)
                shape = shapes[(candidate_id - 1) % len(shapes)]
                radius = 18
                if shape == 'circle':
                    points = cv2.ellipse2Poly((cx, cy), (radius, radius), 0, 0, 360, 15)
                elif shape == 'diamond':
                    points = np.asarray([[cx, cy - radius], [cx + radius, cy], [cx, cy + radius], [cx - radius, cy]], dtype=np.int32)
                else:
                    sides = {'triangle': 3, 'square': 4, 'hexagon': 6, 'octagon': 8, 'cross': 12, 'star': 10}.get(shape, 6)
                    vertices = []
                    for index in range(sides):
                        rr = radius
                        if shape in {'cross', 'star'} and index % 2:
                            rr = radius * (0.42 if shape == 'star' else 0.5)
                        angle = (-90.0 + 360.0 * index / sides) * np.pi / 180.0
                        vertices.append([int(round(cx + rr * np.cos(angle))), int(round(cy + rr * np.sin(angle)))])
                    points = np.asarray(vertices, dtype=np.int32)
                cv2.polylines(canvas, [points], True, (0, 0, 0), 7, cv2.LINE_AA)
                cv2.polylines(canvas, [points], True, (255, 255, 255), 3, cv2.LINE_AA)
                label = str(candidate_id)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                origin = (cx - tw // 2, cy + th // 2)
                cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                mapping.append({'candidate_id': candidate_id, 'shape_badge': shape, 'badge_diameter_px': 36, 'neutral_white_outline': True, 'center_px': [cx, cy], 'mask_pixels': int(mask.sum())})
            return (canvas, mapping)

        @staticmethod
        def _deduplicate(raw_masks, shape):
            valid = []
            for raw in raw_masks:
                mask = np.asarray(raw, dtype=bool)
                if mask.shape != shape or not mask.any():
                    continue
                duplicate = False
                for prior in valid:
                    inter = int((mask & prior).sum())
                    union = int((mask | prior).sum())
                    if union and inter / union >= 0.8:
                        duplicate = True
                        break
                if not duplicate:
                    valid.append(mask)
            return valid

        def _exclude_before_som(self, masks):
            kept, excluded = ([], [])
            for source_index, raw in enumerate(masks, 1):
                mask = np.asarray(raw, dtype=bool)
                area = float(mask.sum())
                reject = False
                overlap_value = 0.0
                for prior in self._row3_exclude_masks:
                    rejected = np.asarray(prior, dtype=bool)
                    if rejected.shape != mask.shape or area <= 0 or (not rejected.any()):
                        continue
                    overlap = float((mask & rejected).sum()) / max(1.0, min(area, float(rejected.sum())))
                    overlap_value = max(overlap_value, overlap)
                    if overlap > 0.5:
                        reject = True
                if reject:
                    excluded.append({'source_candidate_ordinal': source_index, 'overlap_over_min_area': overlap_value})
                else:
                    kept.append(mask)
            return (kept, excluded)

        def _prepare_image_directory(self):
            if self.debug_dir:
                self.debug_dir.mkdir(parents=True, exist_ok=True)

        def select_mask(self, object_name, rgb, segmenter, *, prompt='object', scene_bundle_identity=None):
            target_name = self.memory.target_name_from_instruction(object_name)
            key = memory_module.normalize_key(target_name)
            self._row3_last_object_key = key
            entry = ENTRIES.get(key)
            if entry is None:
                chosen = self._select_mask_baseline(object_name, rgb, segmenter, prompt=prompt)
                evidence = dict(self.last_evidence or {})
                evidence.update({'row3_all_candidate_reground': False, 'row3_entry_hit': False, 'row3_object_key': key, 'row3_parameter_prior_m': Z_PRIORS_M.get(key)})
                self.last_evidence = evidence
                return chosen
            started = time.perf_counter()
            current = self._rgb_array(rgb)
            threshold_before = _threshold_snapshot(segmenter)
            raw_masks = segmenter.segment_candidates(current, prompt='object') or []
            threshold_after = _threshold_snapshot(segmenter)
            if threshold_after != threshold_before:
                raise RuntimeError('SAM3 threshold state changed during generic segmentation')
            masks = self._deduplicate(raw_masks, current.shape[:2])
            masks, excluded = self._exclude_before_som(masks)
            base_evidence = {'schema': 'carryon-row3-all-candidate-reground-selection/v1', 'row3_all_candidate_reground': True, 'row3_entry_hit': True, 'row3_object_key': key, 'sam3_prompt': 'object', 'sam3_prompt_count': 1, 'sam3_threshold_before': threshold_before, 'sam3_threshold_after': threshold_after, 'sam3_threshold_unchanged': True, 'raw_candidate_count': len(raw_masks), 'identity_rejected_consumed_before_som': True, 'identity_rejected_count': len(self._row3_exclude_masks), 'excluded_candidates': excluded, 'candidate_count': len(masks), 'semantic_memory_start_empty': True, 'v7_registry_sha256': memory_module.EXPECTED['v7_registry'], 'row3_parameter_prior_m': Z_PRIORS_M.get(key), 'selection_contract': 'SELECT_AND_MEDIUM_OR_HIGH_ELSE_FAIL_CLOSED', 'no_semantic_location_input': True, 'scene_bundle_identity_present': scene_bundle_identity is not None}
            if not masks:
                self.last_error = 'REGROUND_ALL_CANDIDATE_RECALL_FAILURE'
                base_evidence.update({'selection_status': 'FAIL_CLOSED', 'failure_reason': self.last_error, 'graspgen_called': False})
                self.last_evidence = base_evidence
                self._prepare_image_directory()
                return None
            overlay, mapping = self._neutral_36px_som(current, masks)
            ref_path = self.memory.reference_for(target_name)
            ref_sha = sha256(ref_path)
            if ref_sha != entry['reference_sha256']:
                raise RuntimeError('REGROUND V7 isolated-reference SHA mismatch')
            reference = self._rgb_array(ref_path)
            messages = [{'role': 'system', 'content': PROMPT_BYTES}, {'role': 'user', 'content': [{'type': 'text', 'text': f'Task target name: {target_name!r}. First image is the signed V7 isolated reference; second image is the current all-object candidate SoM.'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + self._b64_jpeg(reference)}}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + self._b64_jpeg(overlay)}}]}]
            response = self._chat(messages)
            raw = getattr(response, 'text', None) or getattr(response, 'content', None) or str(response)
            evidence = dict(base_evidence)
            evidence.update({'mapping': mapping, 'candidate_sources': ['object'] * len(masks), 'reference_rgb': str(ref_path), 'reference_sha256': ref_sha, 'judge_request_id': getattr(response, 'request_id', None), 'judge_raw_response': raw, 'task_target_name_transmitted': target_name, 'semantic_entry_prior_transmitted_to_selector': False})
            try:
                cid, confidence, visual_evidence = _strict_selection(raw, len(masks))
            except Exception as exc:
                self.last_error = f'REGROUND_ALL_CANDIDATE_SELECTION_FAILURE: {exc}'
                evidence.update({'selection_status': 'FAIL_CLOSED', 'failure_reason': self.last_error, 'selected_candidate_id': None, 'graspgen_called': False, 'no_handback_required': True, 'selector_total_ms': round((time.perf_counter() - started) * 1000, 3)})
                self.last_evidence = evidence
                self._prepare_image_directory()
                if self.debug_dir:
                    cv2.imwrite(str(self.debug_dir / 'ROW3_ALL_CANDIDATE_SOM.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                return None
            self._row3_forced_count += 1
            evidence.update({'selection_status': 'SELECTED', 'selected_candidate_id': cid, 'selection_confidence': confidence, 'selection_visual_evidence': visual_evidence, 'forced_reground_count': self._row3_forced_count, 'execution_chain': ['generic_object_sam3', 'identity_rejected', '36px_neutral_white_outline_som', 'V7_isolated_target_name_selection', 'selected_mask_graspgen', 're-grasp'], 'selector_total_ms': round((time.perf_counter() - started) * 1000, 3)})
            self.last_error = None
            self.last_evidence = evidence
            self._prepare_image_directory()
            if self.debug_dir:
                cv2.imwrite(str(self.debug_dir / 'ROW3_ALL_CANDIDATE_SOM.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return masks[cid - 1]

    class AllCandidateBackend(base_backend):

        def _mark_selection_fail_closed(self, reason: str) -> None:
            self._row3_selection_fail_closed = True
            self._row3_selection_fail_reason = reason
            coordinator = getattr(self, '_row3_recovery_coordinator', None)
            if coordinator is not None:
                coordinator.max_attempts = 1

        def perceive_grasps(self, object_hint: str, *, top_k: int=15, source: str='topdown', target_xy=None, target_px=None, grounding=None, sam_text_prompts=None):
            selector = memory_module._selector_for_backend(self)
            key = memory_module.normalize_key(selector.memory.target_name_from_instruction(object_hint) if selector is not None else object_hint)
            self._last_grasp_object_key = key or None
            if key not in ENTRIES:
                return super().perceive_grasps(object_hint, top_k=top_k, source=source, target_xy=target_xy, target_px=target_px, grounding=grounding, sam_text_prompts=sam_text_prompts)
            if not isinstance(selector, AllCandidateSelector):
                self.last_perceive_reason = 'all-candidate selector unavailable'
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
            detector, _ = self._select_detector(source)
            detect_many = getattr(detector, 'detect_many', None)
            if detector is None or not callable(detect_many):
                self.last_perceive_reason = 'selected-mask GraspGen detect_many unavailable'
                self._mark_selection_fail_closed(self.last_perceive_reason)
                return []
            bound_intrinsics = dict(intrinsics)
            bound_intrinsics.pop('segmap', None)
            bound_intrinsics['mask'] = np.asarray(selected, dtype=np.int32)
            try:
                grasps = list(detect_many(rgb=rgb_jpeg, depth_m=depth_m, intrinsics=bound_intrinsics, object_hint=object_hint, top_k=max(1, int(top_k)), target_xy=target_xy, target_px=target_px, grounding=grounding, sam_text_prompts=sam_text_prompts, scene_reference=getattr(self, 'scene_reference', None), exclude_masks=list(getattr(self, 'identity_rejected', None) or [])))
            except Exception as exc:
                self.last_perceive_reason = f'selected-mask GraspGen raised {type(exc).__name__}: {exc}'
                return []
            selection_evidence = dict(selector.last_evidence or {})
            selection_evidence.update({'selected_mask_injected_into_graspgen': True, 'graspgen_called': True, 'graspgen_candidate_count': len(grasps)})
            selector.last_evidence = selection_evidence
            selector._prepare_image_directory()
            self.last_perceive_debug = {'all_candidate_reground': selection_evidence, 'graspgen': dict(getattr(detector, 'last_debug', {}) or {})}
            if not grasps:
                self.last_perceive_reason = getattr(detector, 'last_error', None) or 'selected-mask GraspGen returned no candidates'
            return grasps

        def goto_grasp_topdown_family(self, frozen_target, fallback_grasps, **call_kwargs):
            selector = memory_module._selector_for_backend(self)
            key = memory_module.normalize_key(getattr(selector, '_row3_last_object_key', ''))
            prior_m = Z_PRIORS_M.get(key)
            corrected_z_m = Z_CORRECTED_M.get(key)
            target = frozen_target
            before = [float(value) for value in frozen_target.position]
            if corrected_z_m is not None:
                after = list(before)
                after[2] = float(corrected_z_m)
                target = GraspProposal(position=after, quaternion=[float(value) for value in frozen_target.quaternion], score=frozen_target.score, source=frozen_target.source, predicted_width_m=frozen_target.predicted_width_m)
            from selection_origin_binding import register
            register(self, target, frozen_target)
            result = super().goto_grasp_topdown_family(target, fallback_grasps, **call_kwargs)
            parameter_path = memory_module._store_from_env('CARRYON_RECOVERY_PARAMETER_MEMORY_PATH')
            target_after = [float(value) for value in target.position]
            event = {'schema': 'carryon-fixv2-parameter-memory-application/v2', 'object_key': key or None, 'parameter_memory_sha256': memory_module.sha256(parameter_path) if parameter_path.is_file() else None, 'prior_z_m': prior_m, 'corrected_z_m': corrected_z_m, 'applied': corrected_z_m is not None, 'applied_mode': 'absolute' if corrected_z_m is not None else None, 'target_before_xyz_m': before, 'target_after_xyz_m': target_after, 'delta_effective_m': float(target_after[2]) - float(before[2]), 'v10_mechanical_layer_preserved': True, 's1_retired': True}
            self._row3_last_parameter_event = event
            return result

    def selector_segmenter(selector):
        for detector in list(getattr(selector, '_row3_detector_sources', [])):
            segmenter = getattr(detector, 'segmenter', None)
            if segmenter is not None:
                return segmenter
        raise RuntimeError('all-candidate SAM3 segmenter binding unavailable')

    class FailClosedAwareOrchestrator(base_orchestrator):

        def __init__(self, backend, registry, *, config=None, arbiter=None, log=None):
            wrapped = _FailClosedConfigProxy(config, backend) if isinstance(backend, AllCandidateBackend) and config is not None else config
            super().__init__(backend, registry, config=wrapped, arbiter=arbiter, log=log)
    original_coordinator_init = integration.RecoveryCoordinator.__init__
    original_recover_sync = integration.RecoveryCoordinator.recover_sync

    def coordinator_init(self, *args, **kwargs):
        original_coordinator_init(self, *args, **kwargs)
        if isinstance(self.backend, AllCandidateBackend):
            self.backend._row3_recovery_coordinator = self

    def recover_sync(self, rec):
        if not isinstance(self.backend, AllCandidateBackend):
            return original_recover_sync(self, rec)
        old_max_attempts = self.max_attempts
        self.backend._row3_selection_fail_closed = False
        self.backend._row3_selection_fail_reason = None
        try:
            return original_recover_sync(self, rec)
        finally:
            self.max_attempts = old_max_attempts

    def row3_main(*args, **kwargs):
        vom.VisualMemorySomSelector = AllCandidateSelector
        recovery.GelloRecoveryBackend = AllCandidateBackend
        orchestrator.RecoveryOrchestrator = FailClosedAwareOrchestrator
        integration.RecoveryOrchestrator = FailClosedAwareOrchestrator
        integration.RecoveryCoordinator.__init__ = coordinator_init
        integration.RecoveryCoordinator.recover_sync = recover_sync
        original_selector_for_backend = memory_module._selector_for_backend

        def selector_for_backend_with_sources(backend):
            selector = original_selector_for_backend(backend)
            if isinstance(selector, AllCandidateSelector):
                detectors = list(getattr(backend, 'detector_sources', {}).values())
                fallback = getattr(getattr(backend, 'detector', None), 'fallback', None)
                detectors.extend([getattr(backend, 'detector', None), fallback])
                selector._row3_detector_sources = [detector for detector in detectors if detector is not None]
            return selector
        memory_module._selector_for_backend = selector_for_backend_with_sources
        print('[recovery-grounding] armed: semantic memory and selected-mask recovery', flush=True)
        return true_main(*args, **kwargs)
    main_module.main = row3_main

def main() -> int:
    sys.path.insert(0, str(REPO))
    install_all_candidate_adapter()
    from tools.recovery import recovery_entrypoint as s0
    return int(s0.main() or 0)
if __name__ == '__main__':
    raise SystemExit(main())
