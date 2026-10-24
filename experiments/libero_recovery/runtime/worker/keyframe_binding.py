"""Explicit online composition of the accepted single-grasp/common-generator path."""
from runtime_paths import bound_path, resource
import functools
import inspect
import os
from pathlib import Path

def configure_inner(inner, selector_arm, row_root):
    import policy_binding as guard
    from parameter_visual_binding import configure_inner as visual
    from common_graspgen_binding import configure_inner as common
    root = Path(row_root)
    policy = Path(__file__).resolve().parent / 'recovery_policy.json'
    os.environ['CARRYON_POLICY_JSON'] = str(policy)
    visual(inner, selector_arm, root)
    common(inner)
    previous = inner.install_candidate_presentation

    @functools.wraps(previous)
    def install():
        value = previous()
        import carryon.main as main_module
        outer = inspect.getclosurevars(main_module.main).nonlocals['mask_main']
        earlier = inspect.getclosurevars(outer).nonlocals['selection_main']
        cells = inspect.getclosurevars(earlier).nonlocals
        backend_class = cells['AllCandidateBackend']
        original_init = backend_class.__init__

        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            assert self.perceive_grasps.__name__ == 'perceive_grasps_common_v1'
            assert getattr(self._stream_joint_path, '__critical_frame_v1__', False)
            assert os.environ['CARRYON_POLICY_JSON'] == str(policy)
        backend_class.__init__ = init
        from memory_binding import install as fixed_install
        fixed_install(inner, root)
        from execution_binding import install as final_install
        final_install(inner, root)
        return value
    inner.install_candidate_presentation = install
