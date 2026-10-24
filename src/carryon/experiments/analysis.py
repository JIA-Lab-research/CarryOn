"""Result accounting shared by the public experiment analysis commands."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def binary(value, name):
    if type(value) not in (int, bool) or value not in (0, 1):
        raise ValueError(f"{name} must be Boolean or 0/1")
    return bool(value)


def safety(rows, expected):
    """Use recorded execution-window metrics, without recomputing trajectories."""
    if not rows or not expected:
        raise ValueError("Empty Safety results or case inventory")
    inventory = list(expected)
    expected = {(policy, row['id']): row for policy, row in inventory}
    if len(expected) != len(inventory):
        raise ValueError('Duplicate Safety inventory case')
    seen, groups, errors = set(), defaultdict(list), []
    for record in rows:
        row = record['row']; key = (record['policy'], row['id'])
        if key in seen:
            raise ValueError(f"Duplicate Safety episode: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected Safety episode: {key}")
        seen.add(key)
        for field in ('suite', 'level', 'task_index', 'seed', 'init_index'):
            if row[field] != expected[key][field]:
                raise ValueError(f"Safety case identity differs: {key}, {field}")
        valid = (record.get('status') == 'completed'
                 and record.get('trajectory_complete') is True
                 and not record.get('model_errors') and not record.get('service_errors'))
        if not valid:
            errors.append({'policy': key[0], 'episode_id': key[1],
                           'status': record.get('status'),
                           'model_errors': record.get('model_errors', []),
                           'service_errors': record.get('service_errors', [])})
            continue
        metrics = record['metrics']
        success = binary(metrics['task_success'], 'task_success')
        violation = binary(metrics['execution_fixed_violation'], 'execution_fixed_violation')
        paper_success = binary(metrics['paper_success'], 'paper_success')
        if paper_success != (success and not violation):
            raise ValueError(f"Inconsistent recorded Safety metrics: {key}")
        for group in ((key[0], row['suite'], row['level']), (key[0], 'overall', 'all')):
            groups[group].append((success, violation, paper_success))
    missing = sorted(set(expected) - seen)
    complete = not missing and not errors
    table = []
    expected_groups = defaultdict(int)
    for (policy, _), row in expected.items():
        expected_groups[(policy, row['suite'], row['level'])] += 1
        expected_groups[(policy, 'overall', 'all')] += 1
    for key, count in sorted(expected_groups.items(), key=lambda x: tuple(map(str, x[0]))):
        values = groups[key]; n = len(values)
        table.append(dict(policy=key[0], suite=key[1], level=key[2], expected=count,
                          completed=n, task_successes=sum(v[0] for v in values),
                          violations=sum(v[1] for v in values),
                          violation_free_successes=sum(v[2] for v in values),
                          violation_rate=sum(v[1] for v in values)/n if n == count else None,
                          violation_free_success_rate=sum(v[2] for v in values)/n if n == count else None))
    return dict(status='completed' if complete else 'incomplete', table=table,
                expected_episodes=len(expected), recorded_episodes=len(seen),
                missing=[dict(policy=p, episode_id=i) for p, i in missing], service_errors=errors)


def usage_totals(records, original_count):
    """Account for every attempted call; unavailable usage never means zero."""
    result = {}
    for transport in ('local', 'remote', 'unknown'):
        calls = [r for r in records if r['transport'] == transport]
        known, missing, actual, missing_cost, upper = [], [], Decimal(0), [], Decimal(0)
        unknown_upper = []
        for record in calls:
            usage = record.get('usage') or {}
            if not isinstance(usage, dict):
                usage = {}
            tokens = usage.get('total_tokens')
            if tokens is None and all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')):
                tokens = usage['prompt_tokens'] + usage['completion_tokens']
            if type(tokens) is not int or tokens < 0:
                missing.append(record['path'])
            else:
                known.append(tokens)
            cost = usage.get('cost') if transport == 'remote' else None
            if transport == 'remote':
                try:
                    cost = Decimal(str(cost))
                    if not cost.is_finite() or cost < 0:
                        raise InvalidOperation
                except (InvalidOperation, ValueError):
                    missing_cost.append(record['path']); cost = None
                if cost is not None:
                    actual += cost; upper += cost
                elif record.get('reserved_usd') is not None:
                    reservation = Decimal(str(record['reserved_usd']))
                    if not reservation.is_finite() or reservation < 0:
                        raise ValueError('Invalid request reservation')
                    upper += reservation
                else:
                    unknown_upper.append(record['path'])
        total = None if missing else sum(known)
        result[transport] = dict(attempts=len(calls), known_tokens=sum(known), total_tokens=total,
            tokens_per_original_episode=total/original_count if total is not None and original_count else None,
            missing_usage=missing, service_errors=sum(r.get('status') != 'completed' for r in calls))
        if transport == 'remote':
            result[transport].update(actual_cost_usd=str(actual) if not missing_cost else None,
                known_cost_usd=str(actual), missing_cost=missing_cost,
                cost_upper_bound_usd=str(upper) if not unknown_upper else None,
                missing_reservation=unknown_upper)
    return result


def recovery_usage(root, original_count):
    root = Path(root)
    records, execution_ids, issues = [], set(), []
    for attempt in sorted(root.rglob('ATTEMPT.json')):
        data = read(attempt)
        if not ('global_call' in data or 'item' in data):
            continue
        final = attempt.with_name('RESULT.json')
        record = {**data, **(read(final) if final.exists() else {'status': 'interrupted'})}
        if not record.get('usage'):
            raw = attempt.with_name('response.body.bin')
            if raw.is_file():
                try:
                    value = json.loads(raw.read_bytes())
                    if isinstance(value, dict) and isinstance(value.get('usage'), dict):
                        record['usage'] = value['usage']
                except (ValueError, UnicodeError):
                    pass
        relative = attempt.relative_to(root)
        parts = relative.parts
        phase = 'reflection' if 'item' in data else 'execution'
        round_name = next((p for p in parts if p in ('R0', 'R1', 'R2')), None)
        if phase == 'reflection':
            generation = next((int(p.rsplit('_', 1)[1]) for p in parts
                               if p in ('reflection_1', 'reflection_2')), None)
            round_name = 'R'+str(generation) if generation is not None else None
        else:
            call = data['global_call']
            if type(call) is not int or call < 1 or call in execution_ids:
                raise ValueError('Duplicate or invalid execution call number')
            execution_ids.add(call)
        if round_name is None:
            raise ValueError('Cannot associate request with a recovery round: '+str(relative))
        transport = record.get('transport')
        if transport not in ('local', 'remote'):
            host = record.get('endpoint_host')
            transport = 'local' if host in ('localhost', '127.0.0.1', '::1') else 'remote' if host else 'unknown'
        records.append({**record, 'transport': transport, 'round': round_name,
                        'phase': phase, 'path': relative.as_posix()})
    state_path = root / 'model_budget' / 'STATE.json'
    if state_path.is_file():
        state = read(state_path)
        wanted = set(range(1, state['attempts'] + 1))
        if execution_ids != wanted:
            issues.append({'phase': 'execution', 'missing_calls': sorted(wanted-execution_ids),
                           'unexpected_calls': sorted(execution_ids-wanted)})
    else:
        issues.append({'phase': 'execution', 'reason': 'Request inventory unavailable'})
    for cost_path in root.rglob('COST.json'):
        cost = read(cost_path)
        attempts = list(cost_path.parent.glob('*/ATTEMPT.json'))
        if cost.get('attempts') != len(attempts):
            issues.append({'phase': 'reflection', 'path': str(cost_path.relative_to(root)),
                           'expected': cost.get('attempts'), 'recorded': len(attempts)})
    for folder in root.rglob('reflection_*'):
        if folder.is_dir() and folder.name in ('reflection_1', 'reflection_2'):
            if not list(folder.rglob('COST.json')):
                issues.append({'phase': 'reflection', 'path': str(folder.relative_to(root)),
                               'reason': 'Request inventory unavailable'})
    def summarize(items):
        totals = usage_totals(items, original_count)
        for values in totals.values():
            values['complete'] = not issues and not values['missing_usage']
            if issues:
                values['total_tokens'] = values['tokens_per_original_episode'] = None
                if 'actual_cost_usd' in values:
                    values['actual_cost_usd'] = values['cost_upper_bound_usd'] = None
        return totals
    totals = summarize(records)
    return dict(status='incomplete' if issues or any(x['missing_usage'] for x in totals.values()) else 'completed',
        attempts_recorded=len(records), inventory_errors=issues,
        by_round={name: {
            'execution': summarize([r for r in records if r['round'] == name and r['phase'] == 'execution']),
            'reflection': summarize([r for r in records if r['round'] == name and r['phase'] == 'reflection']),
            'cumulative': summarize([r for r in records if r['round'] <= name]),
        } for name in ('R0', 'R1', 'R2')}, total=totals)


def recovery(root, inventory=None):
    root = Path(root)
    legacy = None
    if root.is_file():
        legacy = read(root)
        root = root.parent
    elif not (root / 'PLAN.json').exists():
        legacy = read(root / 'RESULT.json')
    if legacy is not None:
        if inventory is None or legacy.get('preset') != 'smoke' or legacy.get('memory_mode') != 'paper':
            raise ValueError('A run without PLAN.json requires a fixed paper-memory smoke RESULT and its case inventory')
        if legacy.get('suite') != inventory['suite']:
            raise ValueError('Recovery suite mismatch')
        if 'fresh_smoke' not in inventory or not all(isinstance(r, dict) for r in inventory['smoke']):
            raise ValueError('Legacy fixed-case results require their original case inventory')
        plan = dict(suite=inventory['suite'], preset='smoke', memory='paper',
                    fixed_round_cases=inventory['smoke'], round_zero_cases=inventory['fresh_smoke'])
        flat = legacy['cases']
        if legacy.get('count') != len(flat):
            raise ValueError('Recovery result count mismatch')
        round_data = {}
        for generation in range(3):
            name = 'R'+str(generation)
            cases = [r for r in flat if r.get('row_id') == r.get('logical_key', '')+'_'+name]
            round_data[name] = dict(round=name, cases=cases,
                expected_cases=[r['logical_key'] for r in plan['fixed_round_cases'] if r['generation']==generation],
                status='closed' if all(r.get('status')=='completed' for r in cases) else 'incomplete')
        if sum(len(v['cases']) for v in round_data.values()) != len(flat):
            raise ValueError('Invalid fixed-case round identity')
    else:
        plan = read(root / 'PLAN.json')
        round_data = None
    fixed = plan.get('fixed_round_cases')
    directory = root if fixed is not None else root / 'rounds'
    originals = plan['round_zero_cases']
    if not originals or len(originals) != len(set(originals)):
        raise ValueError('Invalid initial Recovery inventory')
    if plan['preset'] not in ('smoke', 'paper') or plan['memory'] not in ('paper', 'fresh'):
        raise ValueError('Unknown Recovery preset or memory mode')
    if (fixed is not None) != (plan['preset']=='smoke' and plan['memory']=='paper'):
        raise ValueError('Fixed-case and retry-chain protocols cannot be mixed')
    if fixed is not None:
        keys = [(r['generation'], r['logical_key']) for r in fixed]
        if len(keys)!=len(set(keys)) or any(type(g) is not int or g not in (0,1,2) for g,k in keys):
            raise ValueError('Invalid fixed-round inventory')
    if inventory is not None:
        if inventory['suite'] != plan['suite']:
            raise ValueError('Recovery suite mismatch')
        expected_initial = ([r['logical_key'] for r in inventory['rows']] if plan['preset']=='paper'
                            else inventory.get('fresh_smoke', inventory['smoke']))
        if originals != expected_initial or (fixed is not None and fixed != inventory['smoke']):
            raise ValueError('Recovery plan differs from the selected preset inventory')
    pending, completed, rows, errors = set(originals), {}, [], []
    for generation in range(3):
        name = 'R' + str(generation)
        path = directory / name / 'ROUND.json'
        expected = ({r['logical_key'] for r in fixed if r['generation'] == generation}
                    if fixed is not None else pending)
        if round_data is None and not path.is_file():
            errors.append({'round': name, 'reason': 'Missing round result'})
            break
        result = round_data[name] if round_data is not None else read(path)
        cases = result['cases']
        keys = [r['logical_key'] for r in cases]
        declared = result['expected_cases']
        if (result['round'] != name or len(keys) != len(set(keys)) or set(keys) != expected
                or len(declared) != len(set(declared)) or set(declared) != expected):
            raise ValueError('Recovery cases differ from the retry protocol: ' + name)
        failed = [r for r in cases if r.get('status') != 'completed']
        for row in cases:
            if row.get('status') == 'completed' and type(row.get('success')) is not bool:
                raise ValueError('Completed Recovery case requires Boolean success')
            if row.get('row_id', row['logical_key']+'_'+name) != row['logical_key']+'_'+name:
                raise ValueError('Recovery row/round conflict')
        entry = dict(round=name, expected=len(expected), completed=len(cases)-len(failed),
                     successes=sum(r.get('success') is True for r in cases if r['status'] == 'completed'))
        if failed or result['status'] != 'closed':
            errors.extend({'round': name, 'case': r['logical_key'], 'status': r.get('status')} for r in failed)
            if not failed:
                errors.append({'round': name, 'reason': 'Round not closed'})
            entry['status'] = 'incomplete'; rows.append(entry)
            if fixed is None:
                break
            continue
        entry['status'] = 'completed'
        if fixed is None:
            completed.update({r['logical_key']: r['success'] for r in cases})
            pending = {r['logical_key'] for r in cases if not r['success']}
            entry.update(cumulative_successes=sum(completed.values()),
                         cumulative_success_rate=sum(completed.values())/len(originals))
        rows.append(entry)
    final_path = directory / 'RESULT.json'
    final = legacy if legacy is not None else read(final_path) if final_path.is_file() else None
    if final is not None and final.get('status') != 'completed' and not errors:
        errors.append({'reason': 'Run did not complete'})
    return dict(status='incomplete' if errors else 'completed', suite=plan['suite'],
                preset=plan['preset'], memory_mode=plan['memory'],
                protocol='fixed_round_cases' if fixed is not None else 'failure_retry_chain',
                original_case_count=None if fixed is not None else len(originals), rounds=rows,
                errors=errors, usage=recovery_usage(root, None if fixed is not None else len(originals)))
