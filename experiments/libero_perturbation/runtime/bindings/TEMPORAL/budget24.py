"""Instrument-only cap. No model/scene semantics."""
import os, json, time, threading, functools, hashlib
from pathlib import Path
CAPS = {'planner': 1, 'judge': 19, 'reground_selector': 0, 'jam_z': 0}
TOTAL = 20

class BudgetStop(RuntimeError):
    pass
_LOCK = threading.RLock()
_GUARD = None

class Guard:

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self.root / 'COUNT.json'
        self.stopfile = self.root / 'STOP.json'
        if self.state.exists():
            raise BudgetStop('refuse reuse of budget root')
        self.count = 0
        self.roles = {k: 0 for k in CAPS}

    def atomic(self, p, obj):
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, p)

    def stop(self, reason, **details):
        with _LOCK:
            if not self.stopfile.exists():
                self.atomic(self.stopfile, {'reason': reason, 'total': self.count, 'roles': self.roles, 'wall_ts': time.time(), **details})

    def check(self):
        if self.stopfile.exists():
            raise BudgetStop('instrument stop latched')

    def reserve(self, role, body):
        with _LOCK:
            self.check()
            if role not in CAPS:
                self.stop('UNKNOWN_ROLE', role=role)
                raise BudgetStop('unknown role')
            if self.count >= TOTAL or self.roles[role] >= CAPS[role]:
                self.stop('INSTRUMENT_BUDGET_TRUNCATED', blocked_role=role)
                raise BudgetStop('HTTP cap reached')
            self.count += 1
            self.roles[role] += 1
            self.atomic(self.state, {'total': self.count, 'roles': self.roles, 'last_role': role, 'request_sha256': hashlib.sha256(body).hexdigest(), 'wall_ts': time.time(), 'reserved_before_send': True})
            return self.count

    def completed(self):
        with _LOCK:
            if self.count >= TOTAL:
                self.stop('INSTRUMENT_BUDGET_TRUNCATED')

def guard():
    global _GUARD
    if _GUARD is None:
        with _LOCK:
            if _GUARD is None:
                _GUARD = Guard(os.environ['ANCHOR_BUDGET_ROOT'])
    return _GUARD

def check_stop_path(root):
    p = Path(root) / 'STOP.json'
    if p.exists():
        raise BudgetStop('INSTRUMENT_STOP: ' + p.read_text())

def install_parse_guards():
    import carryon.planner.planner as pp
    import carryon.judge.client as jc
    import carryon.recovery.visual_object_memory as vm
    import carryon.recovery.jam_z_reflection as jz
    installed = []

    def wrap(fn, role, judge=False):
        if getattr(fn, '_instrument_budget_guard', False):
            return fn

        @functools.wraps(fn)
        def checked(*a, **kw):
            guard().check()
            try:
                result = fn(*a, **kw)
            except Exception as exc:
                guard().stop('PARSE_FAILURE', role=role, error_type=type(exc).__name__)
                raise
            if judge:
                try:
                    value = jc._extract_decision(json.loads(a[0]))
                    valid = isinstance(value, dict) and 'status' in value and (not jc._validate_decision_schema(value))
                except Exception:
                    valid = False
                if not valid:
                    guard().stop('PARSE_FAILURE', role=role, error=str(getattr(result, 'error', None)))
            return result
        checked._instrument_budget_guard = True
        return checked
    for name in ('parse_and_validate_plan', 'parse_and_validate_reaction_plan'):
        setattr(pp, name, wrap(getattr(pp, name), 'planner'))
        installed.append('planner.' + name)
    jc._parse_decision_text = wrap(jc._parse_decision_text, 'judge', True)
    installed.append('judge._parse_decision_text')
    for module, name, role in ((vm, '_strict_candidate_id', 'reground_selector'), (jz, '_parse', 'jam_z')):
        found = False
        for cls in vars(module).values():
            if not isinstance(cls, type) or cls.__module__ != module.__name__ or name not in cls.__dict__:
                continue
            original = cls.__dict__[name]
            if isinstance(original, staticmethod):
                setattr(cls, name, staticmethod(wrap(original.__func__, role)))
            else:
                setattr(cls, name, wrap(original, role))
            installed.append(cls.__name__ + '.' + name)
            found = True
        if not found:
            raise RuntimeError('missing real parser ' + name)
    return installed
