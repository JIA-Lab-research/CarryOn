"""Install recovery and its parameter-evidence inputs."""
import os

def configure_inner(inner, selector_arm, row_root):
    assert os.environ['CARRYON_GRASP_EXECUTION_MODE'] == 'topdown'
    from parameter_evidence import install
    from parameter_attempt_evidence import install_attribution_capture, install_motion_capture
    from recovery_method import configure_inner as accepted
    import carryon.recovery.backend as bm
    install()
    install_attribution_capture()
    accepted(inner, selector_arm, row_root)
    install_motion_capture(bm.GelloRecoveryBackend)
