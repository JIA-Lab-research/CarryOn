"""Summarize the recorded Safety execution-window metrics."""
from pathlib import Path
import argparse
import csv
import json
from carryon.experiments.analysis import read, safety


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=('paper', 'smoke'))
    parser.add_argument('--policies', nargs='+', choices=('pi05', 'pi05_sft', 'openvla_oft', 'vlact'))
    args = parser.parse_args()
    rows, presets, plans = [], set(), []
    for path in args.inputs:
        if path.suffix == '.jsonl':
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        else:
            document = read(path)
            if 'results' not in document:
                raise ValueError('Expected a runner RESULT.json containing results')
            rows.extend(document['results'])
            if document.get('preset'):
                presets.add(document['preset'])
        plan_path = path.parent / 'PLAN.json'
        if plan_path.is_file():
            plan = read(plan_path); plans.append(plan)
            presets.add(plan['preset'])
    if args.preset:
        presets.add(args.preset)
    if len(presets) != 1:
        parser.error('Provide one consistent preset via --preset, PLAN.json or RESULT.json')
    preset = presets.pop()
    inventory = read(Path(__file__).parent / 'cases.json')
    cases = inventory['rows'] if preset == 'paper' else [r for r in inventory['rows'] if r['id'] in inventory['smoke_ids']]
    policies = args.policies or (sorted({p for plan in plans for p in plan['policies']}) if plans else ['pi05', 'pi05_sft', 'openvla_oft', 'vlact'])
    for plan in plans:
        if plan['rows'] != cases:
            # Compare scientific identity; older archives contain additional provenance fields.
            fields = ('id', 'suite', 'level', 'task_index', 'seed', 'init_index')
            if [{k:r[k] for k in fields} for r in plan['rows']] != [{k:r[k] for k in fields} for r in cases]:
                raise ValueError('Run plan does not match the selected preset')
    report = safety(rows, [(policy, row) for policy in policies for row in cases])
    report['preset'] = preset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['table'][0]))
        writer.writeheader(); writer.writerows(report['table'])
    print(json.dumps(report, indent=2))
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
