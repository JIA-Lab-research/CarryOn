"""Summarize Recovery outcomes and execution/reflection API usage."""
import argparse
import json
from pathlib import Path
from carryon.experiments.analysis import recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inventory', type=Path, default=Path(__file__).with_name('cases.json'))
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
    report = recovery(args.run, inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
