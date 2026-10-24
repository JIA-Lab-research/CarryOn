"""Evidence collection for recovery candidate selection and attribution."""
import functools

def closure(fn):
    return dict(zip(fn.__code__.co_freevars, (x.cell_contents for x in fn.__closure__ or ())))

def configure_inner(inner, arm, row_root):
    assert arm in ('OLD', 'GRID')
    import carryon.recovery.visual_object_memory as vom
    import carryon.main as main_module
    from failure_evidence_attribution import bind_failure_evidence_hooks, FailureEvidenceAttributionBackend
    accepted_selector = vom.VisualMemorySomSelector
    if arm == 'GRID':
        from reference_selector import ThreeImageGridSelector
        assert ThreeImageGridSelector.__bases__ == (accepted_selector,)
        vom.VisualMemorySomSelector = ThreeImageGridSelector
    original_install = inner.install_candidate_presentation

    @functools.wraps(original_install)
    def install():
        original_install()
        v2 = closure(main_module.main)['mask_main']
        v1 = closure(v2)['selection_main']
        cells = closure(v1)
        B = cells['AllCandidateBackend']
        S = cells['AllCandidateSelector']
        before = {n: getattr(B, n) for n in ('perceive_grasps', 'goto_grasp_topdown_family')}
        bind_failure_evidence_hooks(B)
        assert all((getattr(B, n) is fn for n, fn in before.items()))
        assert B.reflect_parameter_memory_at_recovery_end is FailureEvidenceAttributionBackend.reflect_parameter_memory_at_recovery_end
        if arm == 'GRID':
            assert S._overlay_verify_select is ThreeImageGridSelector._overlay_verify_select
        else:
            assert S._overlay_verify_select is accepted_selector._overlay_verify_select
        original_init = B.__init__

        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            assert type(self).reflect_parameter_memory_at_recovery_end is FailureEvidenceAttributionBackend.reflect_parameter_memory_at_recovery_end
        B.__init__ = init
    inner.install_candidate_presentation = install
