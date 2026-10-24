"""Install recovery geometry, policy execution and evidence collection."""
from runtime_paths import bound_path, resource
from pathlib import Path
import os, sys, importlib.util, functools
ROOT = Path(__file__).resolve().parent

def mod(p, n):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[n] = m
    spec.loader.exec_module(m)
    return m

def configure_inner(inner, selector_arm, row_root):
    from recovery_evidence import configure_inner as common, closure
    from original_pose_execution import install_raw_archive
    import carryon.recovery as recovery
    import carryon.recovery.backend as bm
    from final_failure_backend import install as install_terminal_backend
    install_terminal_backend(bm.GelloRecoveryBackend)
    import carryon.recovery.geometry as gm
    import carryon.main as mm
    assert selector_arm == 'GRID'
    execution_mode = os.environ['CARRYON_GRASP_EXECUTION_MODE']
    assert execution_mode == 'topdown', 'common geometry uses exact topdown execution only'
    from policy_binding import install as install_policy_entry, load_checked
    install_policy_entry(row_root)
    om = mod(ROOT / 'orchestrator_handback_standoff.py', 'carryon.recovery.orchestrator')
    recovery.orchestrator = om
    candidate = mod(ROOT / 'geometry_panda_axes.py', 'carryon.recovery.geometry_panda_axes_candidate')
    gm.GraspGenDetector._grasp_world_to_proposal = candidate.GraspGenDetector._grasp_world_to_proposal
    install_raw_archive(gm.GraspGenDetector, row_root)
    bound = mod(ROOT / 'backend_selection_origin.py', 'carryon.recovery.backend_selection_candidate')
    bm.GelloRecoveryBackend._goalset_with_retry = bound.GelloRecoveryBackend._goalset_with_retry
    bm.GelloRecoveryBackend.goto_grasp_topdown_family = bound.GelloRecoveryBackend.goto_grasp_topdown_family
    common(inner, 'GRID', row_root)
    before = inner.install_candidate_presentation

    @functools.wraps(before)
    def install():
        before()
        v2 = closure(mm.main)['mask_main']
        v1 = closure(v2)['selection_main']
        B = closure(v1)['AllCandidateBackend']
        from pose_worker_runtime import install_family as install_common_p_family
        install_common_p_family(B, bm.GelloRecoveryBackend.goto_grasp_topdown_family, archive_root=row_root)
        import policy_binding as _common_guard
        from common_direction_retry import install as install_common_retry
        install_common_retry(_common_guard)
        policy_path = os.environ['CARRYON_POLICY_JSON']
        load_checked(policy_path)
    inner.install_candidate_presentation = install
