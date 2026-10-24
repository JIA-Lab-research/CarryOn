"""Isolated content-bound structural admission; not an efficacy validator.

No GeometryValidator, replay efficacy gate, network or robot operations.
The trusted contract hash and active snapshot are explicit launch inputs.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy, hashlib, json, math, os, tempfile, fcntl
VERSION = 'carryon-object-batch-structural/v1'
TEMPLATE = 'REGROUND_V7_ALL_CANDIDATE_V1'
REQUIRED = ('contract', 'object_key', 'evidence_binding', 'policy_structure', 'candidate_fields_bounds', 'store_content')

def encoded(v):
    return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def digest(b):
    return hashlib.sha256(b).hexdigest()

def sha(p):
    return digest(Path(p).read_bytes())

def read(p):
    return json.loads(Path(p).read_bytes())

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    b = v if isinstance(v, bytes) else encoded(v)
    with p.open('xb') as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    return digest(b)

def require(test, message):
    if not test:
        raise ValueError(message)

def finite(x):
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x)

def rows(b):
    out = {}
    for line in b.splitlines():
        v = json.loads(line)
        k = v['object_key']
        require(k not in out, 'duplicate object key')
        out[k] = v
    return out

def jsonl(v):
    return b''.join(((json.dumps(v[k], sort_keys=True, allow_nan=False) + '\n').encode() for k in sorted(v)))

def check_contract(path, expected):
    require(sha(path) == expected, 'contract SHA mismatch')
    c = read(path)
    require(c['schema'] == VERSION and c['coordinate'] == {'frame': 'planner world', 'reference': 'contact_TCP', 'unit': 'm'}, 'coordinate contract')
    if c.get('common_geometry') is not None:
        from geometry_contract import check_common_contract
        check_common_contract(c, require, sha)
    else:
        require(c['template'] == TEMPLATE and c['physical_grasps'] == 1 and (c['planning_candidates'] == 15), 'template/budget contract')
    require(c['geometry_validator_required'] is False, 'unauthorized geometry gate')
    mr = c.get('molmo_failure_retention')
    if mr is not None:
        require(mr['failure_path_only'] is True and mr['success_path_unchanged'] is True and (mr['extra_model_robot_camera_calls'] == 0), 'molmo retention scope')
    ro = c.get('resume_instruction_original')
    if ro is not None:
        require(ro['simworld_task_language_unchanged'] is True and ro['vla_lang_source'] == 'SimWorld._task_description', 'resume original scope')
    for item in c['source_bindings']:
        require(Path(item['path']).is_file(), 'Missing runtime file: ' + item['path'])
    return c

def check_policy(c):
    p = read(c['policy_path'])
    steps = p['steps']
    ops = [s['op'] for s in steps]
    from carryon.recovery.reflection import load_authored_policy
    from carryon.recovery import default_registry
    registry = default_registry()
    name, version, ok, problems = load_authored_policy(registry, c['policy_path'])
    require(ok and registry.get(name, version) is not None, 'existing policy structural loader: ' + str(problems))
    if c.get('common_geometry') is not None:
        from geometry_contract import check_common_policy
        check_common_policy(p, c, require)
        return
    require(ops.count('close_gripper') == 1 and ops.count('perceive_grasps') == 1, 'physical attempt structure')
    grasp = [s for s in steps if s['op'] == 'goto_grasp_topdown_family']
    require(len(grasp) == 1, 'topdown method drift')
    params = grasp[0]['params']
    require(params['verify'] is True and params['fresh_retry_max_attempts'] == 1 and (params['yaw_degs'] == []), 'verification/retry contract')
    require(next((s for s in steps if s['op'] == 'perceive_grasps'))['params']['top_k'] == 15, 'candidate budget')
    require(ops.index('close_gripper') < ops.index('lift') < ops.index('assert_holding'), 'holding structure')
    require(p['handback_mode'] == 'pre_place', 'handback mode')
    require(c['failure_handback'] == ['fail_safe_rewind', 'failure_handback_goto_grasp', 'failure_handback_open_gripper', 'failure_handback_ack'], 'failure structure')

def validate_stores(p, s, c):
    from parameter_rationale_preference import LengthPreferenceReflector
    from types import SimpleNamespace
    validator = SimpleNamespace(min_z_m=c['min_z_m'], max_z_m=c['max_z_m'], max_abs_delta_m=c['max_abs_delta_m'])
    pr = rows(p)
    sr = rows(s)
    for key, v in pr.items():
        require(key in c['registry_keys'], 'P object registry mismatch')
        LengthPreferenceReflector._parse_persistent_row(validator, v, line_number=1)
    from carryon.recovery.semantic_reflection import STORE_FIELDS, STORE_SCHEMA
    for key, v in sr.items():
        require(set(v) == STORE_FIELDS and v['schema'] == STORE_SCHEMA, 'S schema mismatch')
        require(key in c['registry_keys'] and v['template'] == TEMPLATE, 'S template/key')
        require(isinstance(v['rationale'], str), 'S rationale type')
        require(v['registry_sha256'] == c['registry_sha256'], 'S registry SHA')
        require(str(Path(v['reference_path']).resolve()) == c['references'][key]['path'] and sha(v['reference_path']) == c['references'][key]['sha256'] == v['reference_sha256'], 'S reference mismatch')
        require(v['value_source'] == 'vlm_runtime' and v['trigger'] == 'postrow_recovery_failure', 'S legacy row contract')
    return (pr, sr)

def load_snapshot(pointer, contract_path, contract_sha):
    c = check_contract(contract_path, contract_sha)
    active = read(pointer)
    require(active['contract_sha256'] == contract_sha, 'Memory contract relationship differs')
    root = Path(active['snapshot'])
    report = read(root / active['report_file'])
    require(report['kind'] == 'prepared_memory_load' and report['status'] == 'MEMORY_LOADED',
            'Expected a prepared memory version')
    require(report['initial_store_sha256'] == c['initial_store_sha256'], 'Memory version differs')
    require(report['contract_sha256'] == contract_sha, 'Memory contract differs')
    validate_stores((root / 'PARAMETER.jsonl').read_bytes(), (root / 'SEMANTIC.jsonl').read_bytes(), c)
    check_policy(c)
    return root, report
