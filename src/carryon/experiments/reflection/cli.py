"""Prepare, execute and publish immutable recovery reflection rounds."""
from pathlib import Path
import argparse
import json

from ..files import read_json
from . import collect, memory, prepare, transport


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    zero = sub.add_parser('empty', help='Create byte-empty M0')
    zero.add_argument('--suite', choices=memory.SUITES, required=True)
    zero.add_argument('--output', required=True)
    for command in ('collect', 'build', 'send', 'publish'):
        op = sub.add_parser(command)
        op.add_argument('--experiment', required=True,
                        help='Closed round, collected evidence, or prepared requests directory')
        op.add_argument('--output', required=True, help='New directory; existing outputs are never overwritten')
        if command != 'send':
            op.add_argument('--parent', required=True, help='Prepared M0 or M1 directory')
            op.add_argument('--generation', type=int, choices=(1, 2), required=True)
        if command == 'collect':
            op.add_argument('--suite', choices=memory.SUITES, required=True)
        if command in ('build', 'send'):
            op.add_argument('--config', required=True, help='Explicit reflection model/transport JSON configuration')
        if command == 'publish':
            op.add_argument('--responses', required=True, help='Directory produced by the send command')
    args = parser.parse_args(argv)
    try:
        if args.operation == 'empty':
            result = memory.empty(args.output, args.suite)
        else:
            if args.operation != 'send':
                parent = memory.load(args.parent)
                if parent['manifest']['generation'] != args.generation - 1:
                    raise ValueError('The parent does not precede this reflection generation')
            if args.operation == 'collect':
                result = collect.collect(args.experiment, args.parent, args.output,
                                         suite=args.suite, generation=args.generation)
            elif args.operation == 'build':
                result = prepare.prepare(args.experiment, args.parent, read_json(args.config), args.output)
            elif args.operation == 'send':
                result = transport.send(args.experiment, read_json(args.config), args.output)
            else:
                result = memory.publish(args.parent, read_json(Path(args.experiment) / 'REQUESTS.json'),
                                        read_json(Path(args.responses) / 'RESPONSES.json'), args.output)
        if isinstance(result, dict) and 'snapshot_sha256' in result:
            print(json.dumps({'output': str(Path(args.output).resolve()), 'snapshot_sha256': result['snapshot_sha256']}))
        else:
            print(json.dumps({'output': str(Path(args.output).resolve()), 'records': len(result)}))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, type(exc).__name__ + ': ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
