"""Install the execution methods while retaining the policy registry and class."""
import functools
import importlib.util
import sys
from pathlib import Path
STATE = None
METHODS = ('_run_step', '_run_pose_step_retry')


def aliases():
    import carryon.recovery as recovery
    import carryon.recovery.policy as policy
    import carryon.recovery.reflection as reflection
    if not recovery.SequentialPolicy is policy.SequentialPolicy is reflection.SequentialPolicy:
        raise RuntimeError('Conflicting policy class imports')
    return recovery, policy, reflection, policy.SequentialPolicy


def inspect_instance(instance):
    _, _, _, cls = aliases()
    valid = STATE is not None and type(instance) is cls
    if valid:
        valid = all(getattr(instance, name).__func__ is getattr(STATE['candidate'].SequentialPolicy, name)
                    for name in METHODS + ('_run_retry',))
    return {'status': 'PASS' if valid else 'INVALID', 'class': type(instance).__name__}


def original_load(path):
    recovery, _, reflection, _ = aliases()
    registry = recovery.default_registry()
    loader = STATE['loader'] if STATE else reflection.load_authored_policy
    name, version, ok, problems = loader(registry, str(path))
    return (registry.get(name, version) if ok else None,
            {'name': name, 'version': version, 'ok': ok, 'problems': problems})


def load_checked(path):
    instance, result = original_load(path)
    if not result['ok'] or inspect_instance(instance)['status'] != 'PASS':
        raise ValueError('Invalid recovery policy: ' + str(result['problems']))
    return instance


def install(row_root):
    global STATE
    if STATE is not None:
        raise RuntimeError('Install policy once per worker')
    recovery, policy, reflection, cls = aliases()
    path = Path(__file__).with_name('policy_selection_origin.py')
    spec = importlib.util.spec_from_file_location('carryon.recovery.policy_selection_origin', path)
    candidate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = candidate; spec.loader.exec_module(candidate)
    STATE = {'class': cls, 'canonical': policy, 'candidate': candidate,
             'loader': reflection.load_authored_policy, 'root': Path(row_root)}
    for name in METHODS:
        setattr(cls, name, getattr(candidate.SequentialPolicy, name))

    @functools.wraps(STATE['loader'])
    def load_policy(registry, path):
        loaded = STATE['loader'](registry, path)
        name, version, ok, problems = loaded
        if not ok:
            raise ValueError('Invalid recovery policy: ' + str(problems))
        instance = registry.get(name, version)
        if inspect_instance(instance)['status'] != 'PASS':
            raise RuntimeError('Recovery policy methods are not installed')
        return loaded
    recovery.load_authored_policy = load_policy
    return policy
