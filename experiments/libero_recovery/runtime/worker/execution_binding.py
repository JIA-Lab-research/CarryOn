"""Install recovery execution and failure-evidence collection."""
from runtime_paths import bound_path, resource
import os, sys, json, inspect, functools, importlib.util
from pathlib import Path
ROOT = Path(str(bound_path('bindings')))
MANIFEST = Path(resource('row_manifest'))


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


def install(inner, root):
    root = Path(root)
    manifest = json.loads(MANIFEST.read_text())
    from memory_binding import resolve_manifest_row
    row = resolve_manifest_row(manifest, root.name)
    assert row['judge_control_enabled'] is False and row['judge_model_calls'] == 0 and (row['protocol_recovery_request'] is True)
    import gripper_width as N
    cal = json.loads((ROOT / 'REAL_WIDTH_CONTRACT.json').read_bytes())
    assert N.calibration_valid(cal, cpu=False)
    assert os.environ.get('CARRYON_MAX_ATTEMPTS') == '1'
    import policy_binding as guard
    import failure_evidence_attribution as attribution
    import parameter_attempt_evidence as evidence
    B = inner._memory_consumer_install['backend_class']
    N.bind_bridge_recorder_paths(cal)
    N.bind_measurement_paths(B)
    N.bind_source_exclusion(attribution)
    N.bind_evidence_source(evidence)
    phase = load(ROOT / 'preplace_width.py', 'preplace_sampling')
    phase.bind_holding_entry(guard, cal, cpu=False)
    from attempt_retention import install as retain
    retain(guard, B, root)
    import carryon.jobs.manager as JM
    import carryon.transports.http as HT
    cm = load(ROOT / 'completion_manager.py', 'carryon.jobs.collection_manager_candidate')
    ct = load(ROOT / 'completion_http.py', 'carryon.transports.collection_http_candidate')
    for original, new in [(JM.JobManager, cm.JobManager), (HT.HTTPTransport, ct.HTTPTransport)]:
        for n, v in new.__dict__.items():
            if inspect.isfunction(v) or isinstance(v, (staticmethod, classmethod, property)):
                setattr(original, n, v)
    oldinit = B.__init__

    @functools.wraps(oldinit)
    def init(self, *a, **kw):
        oldinit(self, *a, **kw)
        inner._memory_consumer_install['checked']()
    B.__init__ = init

    from benchmark_start_recovery import install as install_start
    install_start(JM.JobManager)
    return dict(backend_class=B, calibration=cal)
