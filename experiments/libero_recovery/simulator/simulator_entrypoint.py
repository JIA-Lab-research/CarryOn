"""Start the isolated simulator and install its execution timing hooks."""
import importlib.util
import json
import os
from pathlib import Path
import sys
ENTRY = Path(__file__).resolve().parent / 'core'


def main():
    output = Path(os.environ['CARRYON_SIM_STARTUP_LOG'])
    sys.path.insert(0, str(ENTRY))
    spec = importlib.util.spec_from_file_location('isolated_handoff_actual_shim', ENTRY / 'sim_gello_shim.py')
    shim = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = shim; spec.loader.exec_module(shim)
    import sim_world
    if shim.SimWorld is not sim_world.SimWorld:
        raise RuntimeError('Conflicting simulator imports')
    from streak_timing import install_sim
    install_sim(shim, sim_world, output.parent)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'status': 'started', 'pid': os.getpid(), 'module': shim.main.__module__}, stream)
    return shim.main()


if __name__ == '__main__':
    raise SystemExit(main())
