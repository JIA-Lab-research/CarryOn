"""Compare the target reference with scene candidates and retain each comparison."""
from pathlib import Path
import hashlib
import json
import time
import tempfile
import numpy as np
import cv2
from carryon.recovery.visual_object_memory import VisualMemorySomSelector
from carryon.recovery.semantic_memory import mask_sha256
SYSTEM_PROMPT = 'Match the task target to the existing segmentation candidates in one joint comparison. Image 1 is the clean target reference. Image 2 is the unannotated current scene. Image 3 is an enlarged crop grid of all candidate regions, in original candidate order, with IDs outside the crops. Use Image 2 and the reference to inspect scene context; use Image 3 to inspect candidate appearance and associate regions with IDs. Crop labels are outside the image regions; use the unannotated scene for surrounding context. Candidates may be partially occluded or incomplete; compare visible parts and do not reject solely for incomplete shape. Select the single candidate whose visible identity best matches the reference, only if supported by the images. If no candidate can be supported as the reference object, set matches_reference=false and return the existing candidate ID you considered the closest; that is a rejection, not an instruction to execute it. Judge identity, not graspability, grasp height, or task success. Return exactly one JSON object with exactly keys {"candidate_id":integer,"object_type":non-empty string,"matches_reference":boolean}. No markdown or extra keys.'
USER_PROMPT_TEMPLATE = 'Task target: {target_name!r}. Existing candidate IDs: {candidate_ids}. Images in order: clean reference; unannotated current scene; all-candidate enlarged crop grid. Compare all candidates together and return the best supported identity match, or reject.'

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ThreeImageGridSelector(VisualMemorySomSelector):

    @staticmethod
    def render_candidate_grid(current, masks):
        """Same existing crop operation for every candidate; external ID bands."""
        current = VisualMemorySomSelector._rgb_array(current)
        columns, cell_w, crop_h, label_h = (4, 300, 300, 38)
        rows = (len(masks) + columns - 1) // columns
        canvas = np.full((rows * (crop_h + label_h), columns * cell_w, 3), 127, dtype=np.uint8)
        mapping = []
        for cid, mask in enumerate(masks, 1):
            _, chip, geometry = VisualMemorySomSelector._overlay_inputs(current, mask)
            col, row = ((cid - 1) % columns, (cid - 1) // columns)
            x, y = (col * cell_w, row * (crop_h + label_h))
            canvas[y:y + label_h, x:x + cell_w] = 245
            cv2.putText(canvas, 'ID ' + str(cid), (x + 12, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
            canvas[y + label_h:y + label_h + crop_h, x:x + cell_w] = chip
            assert np.array_equal(canvas[y + label_h:y + label_h + crop_h, x:x + cell_w], chip)
            mapping.append({'candidate_id': cid, 'mask_pixels': int(np.asarray(mask, dtype=bool).sum()), 'center_px': [x + 150, y + label_h + 150], 'crop_bbox_xyxy': [x, y + label_h, x + 300, y + label_h + 300], 'label_bbox_xyxy': [x, y, x + 300, y + label_h], 'source_crop_geometry': geometry, 'crop_pixels_unchanged_by_label': True})
        return (canvas, mapping)

    def _overlay_verify_select(self, object_name, target_name, current, masks, candidate_sources, ref_path):
        """One logical comparison; parsing failure never triggers another model call."""
        if not self.debug_dir:
            raise RuntimeError('three-image selector requires an independent debug directory')
        debug = Path(self.debug_dir)
        debug.mkdir(parents=True, exist_ok=True)
        debug = Path(tempfile.mkdtemp(prefix='comparative_call_', dir=debug))
        evidence_path = debug / 'ALL_CANDIDATE_COMPARATIVE_EVIDENCE.json'
        if evidence_path.exists():
            raise RuntimeError('fresh selector evidence path required')
        current = self._rgb_array(current)
        ref_path = Path(ref_path)
        evidence = {'schema': 'carryon-all-candidate-three-image/v1', 'selection_mode': 'all_candidate_comparative_three_image_crop_grid', 'selection_status': None, 'selected_candidate_id': None, 'candidate_count': len(masks), 'candidates': [], 'reference_path': str(ref_path), 'reference_sha256': _sha(ref_path), 'independent_candidate_verification_calls': 0, 'unique_accept_shortcut': False, 'logical_comparison_calls': 0, 'semantic_retries': 0, 'image_order': ['reference', 'unannotated_scene', 'candidate_crop_grid'], 'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), 'parsed': None}

        def finish(status, selected=None, error=None):
            evidence['selection_status'] = status
            evidence['selected_candidate_id'] = None if selected is None else selected + 1
            if error is not None:
                evidence['error'] = error
            with evidence_path.open('x', encoding='utf-8') as stream:
                json.dump(evidence, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write('\n')
            return (selected, evidence)
        if not masks:
            return finish('EMPTY_CANDIDATES')
        if len(candidate_sources) != len(masks):
            return finish('CANDIDATE_INPUT_INVALID', error='candidate source count differs')
        normalized = []
        for cid, raw in enumerate(masks, 1):
            mask = np.asarray(raw, dtype=bool)
            if mask.shape != current.shape[:2] or not mask.any():
                return finish('CANDIDATE_INPUT_INVALID', error='empty mask or image-shape mismatch')
            normalized.append(mask)
            evidence['candidates'].append({'candidate_id': cid, 'candidate_source': candidate_sources[cid - 1], 'candidate_mask_sha256': mask_sha256(mask), 'mask_pixels': int(mask.sum())})
        ids = list(range(1, len(normalized) + 1))
        som, mapping = self.render_candidate_grid(current, normalized)
        if [item['candidate_id'] for item in mapping] != ids:
            return finish('CANDIDATE_MAPPING_INVALID')
        evidence['candidate_mapping'] = mapping
        for name, array in [('COMPARATIVE_UNANNOTATED_CURRENT.png', current), ('COMPARATIVE_ALL_CANDIDATES_CROP_GRID.png', som)]:
            p = debug / name
            if p.exists():
                raise RuntimeError('fresh image evidence path required')
            ok = cv2.imwrite(str(p), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError('image evidence write failed')
            evidence[name] = {'path': str(p), 'sha256': _sha(p)}
        for cid, mask in enumerate(normalized, 1):
            p = debug / ('comparative_candidate_%02d_mask.npy' % cid)
            with p.open('xb') as stream:
                np.save(stream, mask, allow_pickle=False)
        images = [self._rgb_array(ref_path), current, som]
        urls = ['data:image/jpeg;base64,' + self._b64_jpeg(item) for item in images]
        evidence['wire_image_note'] = 'JPEG byte hashes are frozen with payload; PNG hashes are not wire hashes.'
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': [{'type': 'text', 'text': USER_PROMPT_TEMPLATE.format(target_name=target_name, candidate_ids=ids)}, *[{'type': 'image_url', 'image_url': {'url': url}} for url in urls]]}]
        evidence['logical_comparison_calls'] = 1
        started = time.monotonic()
        try:
            response = self._chat(messages)
        except Exception as exc:
            evidence['latency_ms'] = round((time.monotonic() - started) * 1000, 3)
            return finish('COMPARATIVE_TRANSPORT_ERROR', error=type(exc).__name__)
        raw = getattr(response, 'text', None) or getattr(response, 'content', None) or str(response)
        evidence['latency_ms'] = round((time.monotonic() - started) * 1000, 3)
        evidence['raw_response_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        try:
            parsed = self._overlay_strict(raw, set(ids))
        except Exception as exc:
            return finish('PARSE_FAIL', error=type(exc).__name__)
        evidence['parsed'] = parsed
        if not parsed['matches_reference']:
            return finish('ALL_CANDIDATE_COMPARATIVE_REJECTED')
        return finish('ALL_CANDIDATE_COMPARATIVE_MATCH', parsed['candidate_id'] - 1)
