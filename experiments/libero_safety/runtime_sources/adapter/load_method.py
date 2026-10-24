"""Load the isolated Safety modules in their required order."""
import os, sys, importlib, importlib.util
from pathlib import Path
MODULE_FILES={
 'carryon.recovery.geometry':'carryon/recovery/geometry.py',
 'carryon.planner.instruction_roles':'carryon/planner/instruction_roles.py',
 **{n:'safety_eval/'+n+'.py' for n in ['contact_graph','phase_latch','relational_grounding','relational_scene','relational_shield','rollout_predictor','degraded_record']},
 'shield_adapter':'safety_eval/shield_adapter.py'}

def load_method(root):
    root = Path(root).resolve()
    os.environ['SAFE_ROLLOUT_PREDICTOR_DIR'] = str(root / 'safety_eval')
    sys.path.insert(0, str(root / 'safety_eval'))
    loaded = {}
    for name, relative in MODULE_FILES.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        parent = name.rpartition('.')[0]
        if parent:
            importlib.import_module(parent)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if parent:
            setattr(sys.modules[parent], name.rpartition('.')[2], module)
        loaded[name] = str(path)
    return loaded
