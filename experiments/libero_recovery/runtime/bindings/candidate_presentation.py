"""Bind the candidate presentation and selection rules for semantic recovery."""
from __future__ import annotations
from runtime_paths import bound_path, resource
import builtins
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
REPO = Path(str(bound_path('runtime')))
V2 = Path(__file__).with_name('mask_grasp_binding.py')
SOM_FILENAME = 'ROW3_ALL_CANDIDATE_SOM.png'
BUILD = Path(__file__).resolve().parent
PROMPT = BUILD / 'candidate_selection_prompt.txt'
CONTRACT = BUILD / 'candidate_contract.py'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _load_mask_adapter():
    if not V2.is_file():
        raise RuntimeError('Missing candidate mask adapter')
    spec = importlib.util.spec_from_file_location('carryon_candidate_mask_adapter', V2)
    if spec is None or spec.loader is None:
        raise RuntimeError('Candidate mask adapter import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
mask_module = _load_mask_adapter()
selection_module = mask_module.selection_module

def _load_contract():
    if not CONTRACT.is_file():
        raise RuntimeError('Missing presentation/selection contract')
    spec = importlib.util.spec_from_file_location('carryon_candidate_presentation_contract', CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError('Presentation contract import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
CONTRACT_MODULE = _load_contract()

def _bind_presentation_prompt() -> None:
    if not PROMPT.is_file():
        raise RuntimeError('Missing candidate selection prompt')
    prompt_bytes = PROMPT.read_text(encoding='utf-8')
    selection_module.ALL_CANDIDATE_PROMPT = PROMPT
    selection_module.PROMPT_BYTES = prompt_bytes
    selection_module._strict_selection = CONTRACT_MODULE.strict_selection_d_final

def _closure(function) -> dict:
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))


def install_candidate_presentation() -> None:
    if os.environ.get('CARRYON_RECOVERY_PIPELINE') != 'semantic':
        raise RuntimeError('candidate-presentation adapter requires CARRYON_RECOVERY_PIPELINE=semantic')
    import carryon.main as main_module
    _bind_presentation_prompt()
    mask_module.install_mask_adapter()
    mask_main = main_module.main
    cells = _closure(mask_main)
    selection_main = cells['selection_main']
    v1_cells = _closure(selection_main)
    selector_class = v1_cells['AllCandidateSelector']
    backend_class = v1_cells['AllCandidateBackend']
    v2_perceive_grasps = backend_class.perceive_grasps

    def perceive_grasps_v7(self, object_hint, *args, **kwargs):
        selector = selection_module.memory_module._selector_for_backend(self)
        key = selection_module.memory_module.normalize_key(selector.memory.target_name_from_instruction(object_hint) if selector is not None else object_hint)
        if key in selection_module.ENTRIES:
            self._last_grasp_object_key = key or None
        return v2_perceive_grasps(self, object_hint, *args, **kwargs)
    backend_class.perceive_grasps = perceive_grasps_v7


    def select_mask_v3(self, object_name, rgb, segmenter, *, prompt='object', scene_bundle_identity=None):
        target_name = self.memory.target_name_from_instruction(object_name)
        key = selection_module.memory_module.normalize_key(target_name)
        self._row3_last_object_key = key
        entry = selection_module.ENTRIES.get(key)
        if entry is None:
            chosen = self._select_mask_baseline(object_name, rgb, segmenter, prompt=prompt)
            evidence = dict(self.last_evidence or {})
            evidence.update({'row3_all_candidate_reground': False, 'row3_entry_hit': False, 'row3_object_key': key, 'row3_parameter_prior_m': selection_module.Z_PRIORS_M.get(key), 'overlay_verify_live': True})
            self.last_evidence = evidence
            return chosen
        started = time.perf_counter()
        current = self._rgb_array(rgb)
        threshold_before = selection_module._threshold_snapshot(segmenter)
        raw_masks = segmenter.segment_candidates(current, prompt='object') or []
        threshold_after = selection_module._threshold_snapshot(segmenter)
        if threshold_after != threshold_before:
            raise RuntimeError('SAM3 threshold state changed during generic segmentation')
        masks = self._deduplicate(raw_masks, current.shape[:2])
        masks, excluded = self._exclude_before_som(masks)
        if not masks:
            self.last_error = 'REGROUND_ALL_CANDIDATE_RECALL_FAILURE'
            self.last_evidence = {'schema': 'carryon-row3-overlay-verify/v1', 'row3_entry_hit': True, 'row3_object_key': key, 'selection_status': 'FAIL_CLOSED', 'failure_reason': self.last_error, 'candidate_count': 0, 'graspgen_called': False}
            self._prepare_image_directory()
            return None
        ref_path = self.memory.reference_for(target_name)
        ref_sha = sha256(ref_path)
        if ref_sha != entry['reference_sha256']:
            raise RuntimeError('REGROUND R7 reference SHA mismatch')
        selected, overlay = self._overlay_verify_select(object_name, target_name, current, masks, ['object'] * len(masks), ref_path)
        evidence = {'schema': 'carryon-row3-overlay-verify/v1', 'row3_all_candidate_reground': True, 'row3_entry_hit': True, 'row3_object_key': key, 'sam3_prompt': 'object', 'sam3_threshold_before': threshold_before, 'sam3_threshold_after': threshold_after, 'sam3_threshold_unchanged': True, 'raw_candidate_count': len(raw_masks), 'excluded_candidates': excluded, 'candidate_count': len(masks), 'reference_rgb': str(ref_path), 'reference_sha256': ref_sha, 'overlay_verify': overlay, 'selection_status': overlay['selection_status'], 'selected_candidate_id': overlay['selected_candidate_id'], 'row3_parameter_prior_m': selection_module.Z_PRIORS_M.get(key), 'semantic_memory_start_empty': True, 'graspgen_called': selected is not None, 'selector_total_ms': round((time.perf_counter() - started) * 1000, 3)}
        self.last_evidence = evidence
        self._prepare_image_directory()
        if selected is None:
            self.last_error = 'REGROUND_OVERLAY_VERIFY_FAIL_CLOSED'
            return None
        self._row3_forced_count += 1
        self.last_error = None
        return masks[selected]
    selector_class.select_mask = select_mask_v3

    def row3_main_v5(*args, **kwargs):
        builtins.print('[recovery-selection] armed: live overlay-verify + key writeback + full traversal + provider-default', flush=True)
        return mask_main(*args, **kwargs)
    main_module.main = row3_main_v5

def main() -> int:
    sys.path.insert(0, str(REPO))
    install_candidate_presentation()
    from tools.recovery import recovery_entrypoint as s0
    return int(s0.main() or 0)
if __name__ == '__main__':
    raise SystemExit(main())
