"""Install after runtime adapters and before factory construction; process-local.
Per-episode hooks collect evidence only. All active reads require admission.
"""
from pathlib import Path
import functools, inspect, json, os
from memory_loading import *

def install(inner, pointer, cpath, csha, record_path):
    import carryon.recovery.jam_z_reflection as pm
    import carryon.recovery.semantic_reflection as sm
    from carryon.recovery import factory
    from types import SimpleNamespace
    pointer = Path(pointer)
    fixed = pointer.read_bytes()

    def checked():
        require(pointer.read_bytes() == fixed, 'collection/consumer snapshot changed in process')
        root, report = load_snapshot(pointer, cpath, csha)
        require(os.environ['CARRYON_RECOVERY_PARAMETER_MEMORY_PATH'] == str(root / 'PARAMETER.jsonl'), 'P path bypass')
        require(os.environ['CARRYON_RECOVERY_SEMANTIC_MEMORY_PATH'] == str(root / 'SEMANTIC.jsonl'), 'S path bypass')
        return root

    def record(v):
        checked()
        p = Path(record_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        v = {**v, 'fixed_active_pointer_sha256': digest(fixed)}
        with p.open('ab') as f:
            f.write((json.dumps(v, sort_keys=True, allow_nan=False) + '\n').encode())
            f.flush()
            os.fsync(f.fileno())
    P = pm.JamZReflector
    S = sm.SemanticMemoryReflector

    class CollectP(P):

        def __init__(self, *a, **kw):
            checked()
            super().__init__(*a, **kw)

        def _reflect_terminal(self, signal, trigger, before):
            payload = signal.payload()
            record({'event': 'OBJECT_BATCH_COLLECTION_ONLY', 'role': 'P', 'trigger': trigger, 'signal': payload, 'evidence': getattr(self, 'attempt_evidence', None), 'images': getattr(self, 'attempt_images', []), 'write_succeeded': False, 'vlm_call_attempted': False})
            return pm.JamZDecision(applied=False, corrected_z_m=payload['current_z_m'], source='object_batch_collection_only', rationale='Deferred to object batch; not model NO_UPDATE')

        def _persist_or_reuse(self, *a, **kw):
            raise RuntimeError('per-episode direct P writer disabled')

    class CollectS(S):

        def __init__(self, *a, **kw):
            checked()
            super().__init__(*a, **kw)

        def reflect(self, *, object_key, failed_steps):
            record({'event': 'OBJECT_BATCH_COLLECTION_ONLY', 'role': 'S', 'object_key': object_key, 'failed_steps': failed_steps, 'write_succeeded': False, 'vlm_call_attempted': False})
            return {'triggered': bool(failed_steps), 'write_succeeded': False, 'vlm_call_attempted': False, 'collection_only': True, 'object_key': object_key}

        def _append_if_absent(self, *a, **kw):
            raise RuntimeError('per-episode direct S writer disabled')

        def _read_store(self):
            checked()
            return super()._read_store()
    pm.JamZReflector = CollectP
    sm.SemanticMemoryReflector = CollectS
    original_factory = factory._build_memory_reflectors_from_env

    def build():
        checked()
        p, s = original_factory()
        require(isinstance(p, CollectP) and isinstance(s, CollectS), 'factory writer bypass')
        return (p, s)
    factory._build_memory_reflectors_from_env = build
    base = inner.selection_module.memory_module
    for name in ('load_parameter_memory', 'load_parameter_priors', 'load_semantic_entries'):
        old = getattr(base, name)

        def guarded(*a, _original=old, **kw):
            checked()
            return _original(*a, **kw)
        setattr(base, name, guarded)
    import carryon.main as mm
    v2 = inspect.getclosurevars(mm.main).nonlocals['mask_main']
    v1 = inspect.getclosurevars(v2).nonlocals['selection_main']
    B = inspect.getclosurevars(v1).nonlocals['AllCandidateBackend']
    for name in ('perceive_grasps', 'goto_grasp_topdown_family'):
        old = getattr(B, name)

        @functools.wraps(old)
        def guarded_backend(self, *a, _original=old, **kw):
            checked()
            return _original(self, *a, **kw)
        setattr(B, name, guarded_backend)
    return {'checked': checked, 'P': CollectP, 'S': CollectS, 'base': base, 'backend_class': B}
