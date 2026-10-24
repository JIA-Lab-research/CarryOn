from runtime_paths import bound_path, resource
import hashlib, json, os, sys, functools
from pathlib import Path


def record(root, name, x):
    p = Path(root) / name
    p.write_text(json.dumps(x, sort_keys=True, indent=2) + '\n')

def install_prompt(inner, row_root):
    import reference_selector as selector
    prompt = Path(__file__).with_name('selector_prompt.txt')
    selector.SYSTEM_PROMPT = prompt.read_text()
    assert selector.ThreeImageGridSelector._overlay_verify_select.__globals__['SYSTEM_PROMPT'] == prompt.read_text()
    previous = inner.install_candidate_presentation

    @functools.wraps(previous)
    def install():
        value = previous()
        import carryon.main as mm

        def closure(fn):
            return dict(zip(fn.__code__.co_freevars, (x.cell_contents for x in fn.__closure__ or ())))
        v2 = closure(mm.main)['mask_main']
        v1 = closure(v2)['selection_main']
        cells = closure(v1)
        S = cells['AllCandidateSelector']
        assert S._overlay_verify_select is selector.ThreeImageGridSelector._overlay_verify_select
        assert S._overlay_verify_select.__globals__['SYSTEM_PROMPT'] == prompt.read_text()
        record(row_root, 'SELECTOR_NEW_WORKER_ENTRY.json', dict(status='PASS', pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19], selector_file=selector.__file__, class_name=S.__qualname__, actual_function=S._overlay_verify_select.__qualname__, only_system_global_changed=True))
        return value
    inner.install_candidate_presentation = install
