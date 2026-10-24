"""Candidate presentation and final grasp-selection contract."""
from __future__ import annotations
import hashlib
import json
import re
PRESENTATION_POLICY = {'schema': 'carryon-som-bbox-corner-mark/v3', 'presentation': 'bbox_corner_mark_v3', 'candidate_bbox_color_rgb': [255, 255, 255], 'candidate_bbox_thickness_px': 2, 'candidate_bbox_fill': False, 'candidate_bbox_line_type': 'cv2.LINE_8', 'mask_outline_removed': True, 'mark_style': 'solid_black_square_white_numeral', 'one_digit_size_px': [16, 16], 'two_digit_size_px': [22, 16], 'mark_background_rgb': [0, 0, 0], 'mark_foreground_rgb': [255, 255, 255], 'font': 'cv2.FONT_HERSHEY_SIMPLEX', 'font_scale': 0.5, 'font_thickness_px': 1, 'font_line_type': 'cv2.LINE_AA', 'default_placement': 'outside_touching_bbox_upper_left_corner', 'overlap_gate': 'other_candidate_mask_or_2px_bbox_line', 'search': 'own_bbox_perimeter_clockwise_from_upper_left_first_zero_overlap', 'fallback': 'minimum_overlap_then_enumeration_order', 'bbox_format': 'xyxy_exclusive'}
PRESENTATION_POLICY_SHA256 = hashlib.sha256(json.dumps(PRESENTATION_POLICY, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def _mark_size(candidate_id: int) -> tuple[int, int]:
    digits = len(str(candidate_id))
    if digits == 1:
        return tuple(PRESENTATION_POLICY['one_digit_size_px'])
    if digits == 2:
        return tuple(PRESENTATION_POLICY['two_digit_size_px'])
    raise RuntimeError('bbox_corner_mark_v3 supports candidate ids 1..99 only')

def _in_frame(box, width: int, height: int) -> bool:
    x0, y0, x1, y1 = box
    return 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height

def _clockwise_mark_boxes(mask_bbox, mark_w: int, mark_h: int):
    """Yield the upper-left corner then clockwise perimeter placements."""
    mx0, my0, mx1, my1 = mask_bbox
    raw = [(mx0 - mark_w, my0 - mark_h, mx0, my0, 'UPPER_LEFT_CORNER')]
    raw.extend(((x, my0 - mark_h, x + mark_w, my0, 'TOP') for x in range(mx0, mx1)))
    raw.extend(((mx1, y, mx1 + mark_w, y + mark_h, 'RIGHT') for y in range(my0, my1)))
    raw.extend(((x - mark_w + 1, my1, x + 1, my1 + mark_h, 'BOTTOM') for x in range(mx1 - 1, mx0 - 1, -1)))
    raw.extend(((mx0 - mark_w, y - mark_h + 1, mx0, y + 1, 'LEFT') for y in range(my1 - 1, my0 - 1, -1)))
    seen = set()
    for x0, y0, x1, y1, side in raw:
        box = (int(x0), int(y0), int(x1), int(y1))
        if box in seen:
            continue
        seen.add(box)
        yield (box, side)

def _bbox_line_occupancy(np, shape, bbox, thickness: int):
    height, width = shape
    x0, y0, x1, y1 = bbox
    occupancy = np.zeros((height, width), dtype=bool)
    t = int(thickness)
    occupancy[max(0, y0):min(height, y0 + t), max(0, x0):min(width, x1)] = True
    occupancy[max(0, y1 - t):min(height, y1), max(0, x0):min(width, x1)] = True
    occupancy[max(0, y0):min(height, y1), max(0, x0):min(width, x0 + t)] = True
    occupancy[max(0, y0):min(height, y1), max(0, x1 - t):min(width, x1)] = True
    return occupancy

def bbox_corner_mark_v3_som(rgb, masks):
    import cv2
    import numpy as np
    canvas = np.asarray(rgb).copy()
    if canvas.ndim != 3 or canvas.shape[2] < 3:
        raise RuntimeError('SoM rgb must be HxWx3')
    frame_h, frame_w = canvas.shape[:2]
    valid_masks = [np.asarray(raw, dtype=bool) for raw in masks]
    if any((mask.shape != (frame_h, frame_w) or not mask.any() for mask in valid_masks)):
        raise RuntimeError('SoM candidate mask shape/emptiness drift')
    mask_bboxes = []
    bbox_lines = []
    for mask in valid_masks:
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        mask_bboxes.append(bbox)
        bbox_lines.append(_bbox_line_occupancy(np, (frame_h, frame_w), bbox, int(PRESENTATION_POLICY['candidate_bbox_thickness_px'])))
    placements = []
    for index, (mask, mask_bbox) in enumerate(zip(valid_masks, mask_bboxes, strict=True)):
        candidate_id = index + 1
        mark_w, mark_h = _mark_size(candidate_id)
        forbidden = np.zeros((frame_h, frame_w), dtype=bool)
        for other_index, (other_mask, other_line) in enumerate(zip(valid_masks, bbox_lines, strict=True)):
            if other_index == index:
                continue
            forbidden |= other_mask
            forbidden |= other_line
        choices = []
        for enumeration_index, (box, side) in enumerate(_clockwise_mark_boxes(mask_bbox, mark_w, mark_h)):
            if not _in_frame(box, frame_w, frame_h):
                continue
            x0, y0, x1, y1 = box
            overlap = int(forbidden[y0:y1, x0:x1].sum())
            choices.append((overlap, enumeration_index, box, side))
        if not choices:
            raise RuntimeError('bbox_corner_mark_v3 found no in-frame mark placement')
        zero = next((choice for choice in choices if choice[0] == 0), None)
        chosen = zero if zero is not None else min(choices, key=lambda choice: (choice[0], choice[1]))
        overlap, enumeration_index, box, side = chosen
        x0, y0, x1, y1 = box
        own_overlap = int(mask[y0:y1, x0:x1].sum())
        placements.append({'candidate_id': candidate_id, 'presentation': PRESENTATION_POLICY['presentation'], 'mark_style': PRESENTATION_POLICY['mark_style'], 'mark_bbox': [x0, y0, x1, y1], 'mark_bbox_format': PRESENTATION_POLICY['bbox_format'], 'mark_position_px': [x0, y0], 'mark_size_px': [mark_w, mark_h], 'mark_perimeter_side': side, 'mark_search_enumeration_index': int(enumeration_index), 'badge_overlap_fallback': zero is None, 'occluded_pixels': int(overlap), 'own_mask_occluded_pixels': own_overlap, 'mask_pixels': int(mask.sum()), 'mask_bbox': mask_bbox, 'mask_outline_removed': True, 'candidate_bbox_thickness_px': int(PRESENTATION_POLICY['candidate_bbox_thickness_px'])})
    for bbox in mask_bboxes:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), tuple(PRESENTATION_POLICY['candidate_bbox_color_rgb']), int(PRESENTATION_POLICY['candidate_bbox_thickness_px']), cv2.LINE_8)
    for item in placements:
        x0, y0, x1, y1 = item['mark_bbox']
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), -1, cv2.LINE_8)
        label = str(item['candidate_id'])
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, float(PRESENTATION_POLICY['font_scale']), int(PRESENTATION_POLICY['font_thickness_px']))
        origin = (x0 + (x1 - x0 - text_w) // 2, y0 + (y1 - y0 + text_h) // 2)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, float(PRESENTATION_POLICY['font_scale']), (255, 255, 255), int(PRESENTATION_POLICY['font_thickness_px']), cv2.LINE_AA)
    return (canvas, placements)

def strict_selection_d_final(text: str, count: int) -> tuple[int, str, str]:
    """Accept exactly one in-range candidate; confidence is telemetry only."""
    if not isinstance(text, str) or re.fullmatch('\\s*\\{.*\\}\\s*', text, re.DOTALL) is None:
        raise ValueError('response must be exactly one JSON object')
    payload = json.loads(text)
    expected_keys = {'candidate_id', 'confidence', 'evidence'}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError('D-final selection schema keys mismatch')
    cid = payload['candidate_id']
    confidence = payload['confidence']
    evidence = payload['evidence']
    if isinstance(cid, bool) or not isinstance(cid, int) or (not 1 <= cid <= count):
        raise ValueError('candidate_id missing/out_of_range')
    if confidence not in {'LOW', 'MEDIUM', 'HIGH'}:
        raise ValueError('confidence is not LOW|MEDIUM|HIGH')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('evidence must be a non-empty string')
    return (cid, confidence, evidence.strip())
