"""Bind selected masks to grasp generation and render candidate labels."""
from __future__ import annotations
from runtime_paths import bound_path, resource
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
REPO = Path(str(bound_path('runtime')))
BASE = REPO / 'logs/recovery'
V1 = Path(__file__).with_name('candidate_selection.py')
BADGE_LAYOUT_POLICY = {'schema': 'carryon-som-black-square-badge-layout/v2', 'badge_style': 'black_square_v2', 'one_digit_size_px': [16, 16], 'two_digit_size_px': [22, 16], 'font': 'cv2.FONT_HERSHEY_SIMPLEX', 'font_scale': 0.5, 'font_thickness_px': 1, 'font_line_type': 'cv2.LINE_AA', 'foreground_rgb': [255, 255, 255], 'background_rgb': [0, 0, 0], 'small_mask_area_multiplier': 3, 'distance_transform_type': 'cv2.DIST_L2', 'distance_transform_mask_size': 5, 'distance_transform_tie_break': 'lexicographic_y_x_smallest', 'directions': [[0, -1, 'N'], [1, -1, 'NE'], [1, 0, 'E'], [1, 1, 'SE'], [0, 1, 'S'], [-1, 1, 'SW'], [-1, 0, 'W'], [-1, -1, 'NW']], 'search_loop': 'radius_major_then_direction_order', 'radius_step_px': 1, 'radius_budget': 'ceil(frame_diagonal_px)+ceil(half_badge_bbox_diagonal_px)', 'diagonal_direction_metric': 'integer_chebyshev_ray', 'fallback': 'minimum_bbox_overlap_then_first_enumeration', 'leader_width_px': 1, 'leader_rgb': [255, 255, 255], 'leader_line_type': 'cv2.LINE_AA', 'leader_participates_in_overlap_gate': False, 'mask_outline_black_px': 6, 'mask_outline_white_px': 3, 'badge_bbox_format': 'xyxy_exclusive'}
BADGE_LAYOUT_POLICY_SHA256 = hashlib.sha256(json.dumps(BADGE_LAYOUT_POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _load_v1():
    if not V1.is_file():
        raise RuntimeError('Candidate selection adapter file is missing')
    spec = importlib.util.spec_from_file_location('carryon_candidate_selection_adapter', V1)
    if spec is None or spec.loader is None:
        raise RuntimeError('Candidate selection adapter import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
selection_module = _load_v1()

def _closure(function) -> dict:
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))

def _badge_size(candidate_id: int) -> tuple[int, int]:
    digits = len(str(candidate_id))
    if digits == 1:
        return tuple(BADGE_LAYOUT_POLICY['one_digit_size_px'])
    if digits == 2:
        return tuple(BADGE_LAYOUT_POLICY['two_digit_size_px'])
    raise RuntimeError('black_square_v2 supports candidate ids 1..99 only')

def _bbox_for_center(cx: int, cy: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(cx) - int(width) // 2
    y0 = int(cy) - int(height) // 2
    return (x0, y0, x0 + int(width), y0 + int(height))

def _bbox_in_frame(bbox, width: int, height: int) -> bool:
    x0, y0, x1, y1 = bbox
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height

def _bbox_overlap(mask, bbox) -> int:
    x0, y0, x1, y1 = bbox
    return int(mask[y0:y1, x0:x1].sum())

def _deepest_point(cv2, np, mask) -> tuple[int, int, float]:
    distance = cv2.distanceTransform(np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, int(BADGE_LAYOUT_POLICY['distance_transform_mask_size']))
    maximum = float(distance.max())
    if maximum <= 0.0:
        raise RuntimeError('distance transform produced no interior point')
    ties = np.argwhere(distance == maximum)
    y, x = (int(ties[0, 0]), int(ties[0, 1]))
    return (x, y, maximum)

def _line_start_on_bbox(cx: int, cy: int, target_x: int, target_y: int, bbox) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    dx, dy = (float(target_x - cx), float(target_y - cy))
    if dx == 0.0 and dy == 0.0:
        return (cx, cy)
    half_w = max(0.5, min(float(cx - x0), float(x1 - 1 - cx)) + 0.5)
    half_h = max(0.5, min(float(cy - y0), float(y1 - 1 - cy)) + 0.5)
    scales = []
    if dx != 0.0:
        scales.append(half_w / abs(dx))
    if dy != 0.0:
        scales.append(half_h / abs(dy))
    scale = min(scales)
    return (int(round(cx + dx * scale)), int(round(cy + dy * scale)))

def _black_square_v2_som(rgb, masks):
    import cv2
    import numpy as np
    current = selection_module.memory_module._selector_for_backend
    del current
    canvas = np.asarray(rgb).copy()
    if canvas.ndim != 3 or canvas.shape[2] < 3:
        raise RuntimeError('SoM rgb must be HxWx3')
    frame_h, frame_w = canvas.shape[:2]
    valid_masks = [np.asarray(raw, dtype=bool) for raw in masks]
    if any((mask.shape != (frame_h, frame_w) or not mask.any() for mask in valid_masks)):
        raise RuntimeError('SoM candidate mask shape/emptiness drift')
    union = np.logical_or.reduce(valid_masks)
    placements = []
    for candidate_id, mask in enumerate(valid_masks, 1):
        ys, xs = np.nonzero(mask)
        centroid_x = int(round(float(xs.mean())))
        centroid_y = int(round(float(ys.mean())))
        deepest_x, deepest_y, deepest_distance = _deepest_point(cv2, np, mask)
        mask_bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        badge_w, badge_h = _badge_size(candidate_id)
        bbox_area = int(badge_w * badge_h)
        threshold = int(BADGE_LAYOUT_POLICY['small_mask_area_multiplier'] * bbox_area)
        mask_pixels = int(mask.sum())
        small = mask_pixels < threshold
        chosen_bbox = _bbox_for_center(deepest_x, deepest_y, badge_w, badge_h)
        chosen_position = (deepest_x, deepest_y)
        chosen_overlap = _bbox_overlap(union, chosen_bbox) if _bbox_in_frame(chosen_bbox, frame_w, frame_h) else None
        fallback = False
        offset = False
        search_radius = 0
        search_direction = None
        searched_positions = 0
        if small:
            offset = True
            best = None
            outer_radius = int(math.ceil(0.5 * math.hypot(badge_w, badge_h)))
            budget = int(math.ceil(math.hypot(frame_w, frame_h))) + outer_radius
            found = None
            for radius in range(1, budget + 1, int(BADGE_LAYOUT_POLICY['radius_step_px'])):
                for dx, dy, direction in BADGE_LAYOUT_POLICY['directions']:
                    cx = deepest_x + int(dx) * radius
                    cy = deepest_y + int(dy) * radius
                    bbox = _bbox_for_center(cx, cy, badge_w, badge_h)
                    if not _bbox_in_frame(bbox, frame_w, frame_h):
                        continue
                    searched_positions += 1
                    overlap = _bbox_overlap(union, bbox)
                    record = (overlap, searched_positions, bbox, (cx, cy), radius, direction)
                    if best is None or record[:2] < best[:2]:
                        best = record
                    if overlap == 0:
                        found = record
                        break
                if found is not None:
                    break
            choice = found if found is not None else best
            if choice is None:
                raise RuntimeError('badge offset search produced no in-frame position')
            chosen_overlap, _, chosen_bbox, chosen_position, search_radius, search_direction = choice
            fallback = found is None
        else:
            x0, y0, x1, y1 = chosen_bbox
            mx0, my0, mx1, my1 = mask_bbox
            if not _bbox_in_frame(chosen_bbox, frame_w, frame_h):
                raise RuntimeError('on-object badge bbox is outside frame')
            if not (mx0 <= x0 < x1 <= mx1 and my0 <= y0 < y1 <= my1):
                raise RuntimeError('on-object badge bbox is outside its mask bbox')
        x0, y0, x1, y1 = chosen_bbox
        own_overlap = _bbox_overlap(mask, chosen_bbox)
        total_overlap = int(chosen_overlap)
        other_overlap = int(total_overlap - own_overlap)
        placements.append({'candidate_id': candidate_id, 'badge_style': BADGE_LAYOUT_POLICY['badge_style'], 'badge_bbox': [int(x0), int(y0), int(x1), int(y1)], 'badge_bbox_format': BADGE_LAYOUT_POLICY['badge_bbox_format'], 'badge_bbox_area': bbox_area, 'badge_position_px': [int(chosen_position[0]), int(chosen_position[1])], 'badge_offset_applied': bool(offset), 'badge_overlap_fallback': bool(fallback), 'placement': 'offset' if offset else 'interior', 'occluded_pixels': total_overlap, 'own_mask_occluded_pixels': own_overlap, 'other_mask_occluded_pixels': other_overlap, 'mask_pixels': mask_pixels, 'small_mask_threshold_px': threshold, 'small_mask_triggered': bool(small), 'mask_bbox': mask_bbox, 'mask_interior_point_px': [deepest_x, deepest_y], 'mask_interior_distance_px': deepest_distance, 'mask_centroid_px': [centroid_x, centroid_y], 'search_radius_px': int(search_radius), 'search_direction': search_direction, 'searched_in_frame_positions': int(searched_positions), 'leader_drawn': bool(offset), 'neutral_white_outline': True})
    for item in placements:
        if not item['badge_offset_applied']:
            continue
        cx, cy = item['badge_position_px']
        tx, ty = item['mask_centroid_px']
        start = _line_start_on_bbox(cx, cy, tx, ty, item['badge_bbox'])
        cv2.line(canvas, start, (tx, ty), tuple(BADGE_LAYOUT_POLICY['leader_rgb']), int(BADGE_LAYOUT_POLICY['leader_width_px']), cv2.LINE_AA)
    for mask in valid_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), int(BADGE_LAYOUT_POLICY['mask_outline_black_px']), cv2.LINE_AA)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), int(BADGE_LAYOUT_POLICY['mask_outline_white_px']), cv2.LINE_AA)
    for item in placements:
        x0, y0, x1, y1 = item['badge_bbox']
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), -1, cv2.LINE_8)
        label = str(item['candidate_id'])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, float(BADGE_LAYOUT_POLICY['font_scale']), int(BADGE_LAYOUT_POLICY['font_thickness_px']))
        origin = (x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 + th) // 2)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, float(BADGE_LAYOUT_POLICY['font_scale']), (255, 255, 255), int(BADGE_LAYOUT_POLICY['font_thickness_px']), cv2.LINE_AA)
    return (canvas, placements)

def install_mask_adapter() -> None:
    if os.environ.get('CARRYON_RECOVERY_PIPELINE') != 'semantic':
        raise RuntimeError('mask-grasp adapter requires CARRYON_RECOVERY_PIPELINE=semantic')
    import numpy as np
    import carryon.main as main_module
    from carryon.recovery.geometry import GraspGenDetector
    selection_module.install_all_candidate_adapter()
    selection_main = main_module.main
    cells = _closure(selection_main)
    selector_class = cells['AllCandidateSelector']
    backend_class = cells['AllCandidateBackend']
    original_select_mask = selector_class.select_mask
    selector_class._black_square_v2_som = staticmethod(_black_square_v2_som)
    selector_class._neutral_36px_som = staticmethod(_black_square_v2_som)

    def select_mask_v2(self, *args, **kwargs):
        selected = original_select_mask(self, *args, **kwargs)
        evidence = dict(getattr(self, 'last_evidence', {}) or {})
        if evidence.get('row3_entry_hit') is True:
            evidence['schema'] = 'carryon-row3-all-candidate-reground-selection/v2'
            evidence['badge_style'] = BADGE_LAYOUT_POLICY['badge_style']
            evidence['badge_layout_policy'] = BADGE_LAYOUT_POLICY
            evidence['badge_layout_policy_sha256'] = BADGE_LAYOUT_POLICY_SHA256
            evidence['shape_badges_removed'] = True
            chain = list(evidence.get('execution_chain') or [])
            evidence['execution_chain'] = ['black_square_v2_som' if step == '36px_neutral_white_outline_som' else step for step in chain]
            self.last_evidence = evidence
            self._prepare_image_directory()
        return selected
    selector_class.select_mask = select_mask_v2

    def selector_segmenter(selector):
        for detector in list(getattr(selector, '_row3_detector_sources', [])):
            segmenter = getattr(detector, 'segmenter', None)
            if segmenter is not None:
                return segmenter
        raise RuntimeError('all-candidate SAM3 segmenter binding unavailable')

    def perceive_grasps_v2(self, object_hint: str, *, top_k: int=15, source: str='topdown', target_xy=None, target_px=None, grounding=None, sam_text_prompts=None):
        selector = selection_module.memory_module._selector_for_backend(self)
        key = selection_module.memory_module.normalize_key(selector.memory.target_name_from_instruction(object_hint) if selector is not None else object_hint)
        if key not in selection_module.ENTRIES:
            return super(backend_class, self).perceive_grasps(object_hint, top_k=top_k, source=source, target_xy=target_xy, target_px=target_px, grounding=grounding, sam_text_prompts=sam_text_prompts)
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
        detector = getattr(self, 'detector', None)
        detect_many = getattr(detector, 'detect_many', None)
        if not isinstance(detector, GraspGenDetector) or not callable(detect_many):
            self.last_perceive_reason = 'production GraspGen primary detect_many unavailable'
            evidence = dict(selector.last_evidence or {})
            evidence.update({'execution_status': 'FAIL_CLOSED', 'failure_reason': self.last_perceive_reason, 'production_graspgen_primary_bound': False, 'selected_mask_injected_into_graspgen': False, 'graspgen_called': False})
            selector.last_evidence = evidence
            selector._prepare_image_directory()
            self._mark_selection_fail_closed(self.last_perceive_reason)
            return []
        bound_intrinsics = dict(intrinsics)
        bound_intrinsics.pop('segmap', None)
        bound_intrinsics['mask'] = np.asarray(selected, dtype=np.int32)
        try:
            grasps = list(detect_many(rgb=rgb_jpeg, depth_m=depth_m, intrinsics=bound_intrinsics, object_hint=object_hint, top_k=max(1, int(top_k)), target_xy=target_xy, target_px=target_px, grounding=grounding, sam_text_prompts=sam_text_prompts, scene_reference=getattr(self, 'scene_reference', None), exclude_masks=list(getattr(self, 'identity_rejected', None) or [])))
        except Exception as exc:
            self.last_perceive_reason = f'production GraspGen primary raised {type(exc).__name__}: {exc}'
            evidence = dict(selector.last_evidence or {})
            evidence.update({'execution_status': 'FAIL_CLOSED', 'failure_reason': self.last_perceive_reason, 'production_graspgen_primary_bound': True, 'production_graspgen_detector_class': type(detector).__name__, 'selected_mask_injected_into_graspgen': True, 'graspgen_called': True})
            selector.last_evidence = evidence
            selector._prepare_image_directory()
            self._mark_selection_fail_closed(self.last_perceive_reason)
            return []
        evidence = dict(selector.last_evidence or {})
        evidence.update({'execution_status': 'GRASPGEN_RETURNED', 'production_graspgen_primary_bound': True, 'production_graspgen_detector_class': type(detector).__name__, 'authored_source_alias_ignored_on_entry_hit': source, 'selected_mask_injected_into_graspgen': True, 'graspgen_called': True, 'graspgen_candidate_count': len(grasps)})
        selector.last_evidence = evidence
        selector._prepare_image_directory()
        self.last_perceive_debug = {'all_candidate_reground': evidence, 'graspgen': dict(getattr(detector, 'last_debug', {}) or {})}
        if not grasps:
            self.last_perceive_reason = getattr(detector, 'last_error', None) or 'selected-mask GraspGen returned no candidates'
        return grasps
    backend_class.perceive_grasps = perceive_grasps_v2


    def row3_main_v2(*args, **kwargs):
        return selection_main(*args, **kwargs)
    main_module.main = row3_main_v2

def main() -> int:
    sys.path.insert(0, str(REPO))
    install_mask_adapter()
    from tools.recovery import recovery_entrypoint as s0
    return int(s0.main() or 0)
if __name__ == '__main__':
    raise SystemExit(main())
