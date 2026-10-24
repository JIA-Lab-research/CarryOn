"""One independent CarryOn Safety episode with an isolated simulator and Shield."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import numpy as np

from ..files import read_json, sha256, write_json
from ..configuration import validate_public_config
from .gates import load_module, NativeEight, Pi05Gate, pi05_query
from .materialize import runtime_files
from .process import RuntimeClient
from .health import classify

AUTH = b'carryon-safety-local-protocol4-v1'


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def configure_modules(config):
    root = Path(config['runtime']).resolve()
    runtime_files(root)
    sys.path.insert(0, str(root / 'support'))
    return root


def supervisor(config_path, socket_path):
    from multiprocessing.connection import Listener
    config = read_json(config_path)
    root = configure_modules(config)
    sys.path.insert(0, str(Path(config['benchmark']).resolve()))
    from phase1_blind_supervisor import SupervisorConfig, run_supervisor
    from phase1_protocol4_channel import protocol4_accept
    driver_module = load_module(root / 'adapter/simulation_driver.py', 'carryon_benchmark_driver')
    cls = driver_module.SafetySimulationDriver
    if config['policy']['kind'] in ('pi05', 'pi05_sft'):
        from .pi05_driver import make_driver_class
        cls = make_driver_class(driver_module)
    driver_config = driver_module.RawDriverConfig(**config['driver'])
    listener = Listener(socket_path, family='AF_UNIX', authkey=AUTH)
    try:
        channel = protocol4_accept(listener, 'carryon_safety_supervisor')
        run_supervisor(lambda: cls(driver_config), channel, channel, SupervisorConfig(**config['supervisor']))
    finally:
        listener.close()


def isolated_checkpoint(source, destination):
    """OFT's loader may rewrite model config; keep every shared cache file intact."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir()
    for item in source.iterdir():
        target = destination / item.name
        if item.is_file() and item.suffix in ('.safetensors', '.pt', '.bin'):
            target.symlink_to(item.resolve())
        elif item.is_file():
            shutil.copyfile(item, target)
        elif item.is_dir():
            shutil.copytree(item, target)
    return destination


def get_policy(config, root, output):
    spec, row = config['policy'], config['row']
    kind = spec['kind']
    if kind == 'openvla_oft':
        os.environ['HF_MODULES_CACHE'] = str(output / 'hf_modules')
        module = load_module(root / 'adapter/openvla_policy.py', 'carryon_oft_policy')
        checkpoint = isolated_checkpoint(spec['checkpoint'], output / 'deployed_checkpoint')
        policy = module.OpenVLAPolicy(checkpoint, spec['upstream'])
        identity = policy.identity(checkpoint)
        return lambda obs, step, sequence: policy.query(obs, row['task_index']), identity, None
    if kind == 'vlact':
        sys.path.insert(0, str(Path(spec['upstream']).resolve()))
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        client = WebsocketClientPolicy('127.0.0.1', int(spec['port']))
        metadata = client.get_server_metadata()
        if metadata.get('checkpoint_revision') != spec['checkpoint_revision'] or metadata.get('chunk') != 8:
            raise RuntimeError('VLAct service identity differs from the configured checkpoint')
        def query(obs, step, sequence):
            request = {k: obs[k] for k in ('policy_agentview_raw', 'policy_wrist_raw', 'task_description')}
            response = client.predict_action(dict(observation=request, episode_key=row['id'], control_step=step))
            if response.get('ok') is not True:
                raise RuntimeError('VLAct service did not return a successful response')
            value = response['data']
            return np.asarray(value['actions'], dtype=np.float64), value['receipt']
        return query, metadata, client
    if kind in ('pi05', 'pi05_sft'):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy(host='127.0.0.1', port=int(spec['port']))
        metadata = client.get_server_metadata()
        for key in ('protocol', 'model_id', 'model_revision'):
            if metadata.get(key) != spec['metadata'][key]:
                raise RuntimeError('pi0.5 service identity mismatch: ' + key)
        return pi05_query(client, row, root), metadata, client
    raise ValueError('Unknown policy: ' + kind)


def run(config, output):
    validate_public_config(config)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    root = configure_modules(config)
    sys.path.insert(0,str(Path(config['benchmark']).resolve()))
    os.environ['LIBERO_CONFIG_PATH']=str(Path(config['benchmark_config']).resolve())
    policy_kind = config['policy']['kind']
    pi05 = policy_kind in ('pi05', 'pi05_sft')
    row = config['row']
    if config.get('control_steps', 900) != 900:
        raise ValueError('Smoke changes sample count, never the per-episode execution budget')
    output.mkdir(parents=True)
    private = output / 'evaluation_private'
    private.mkdir(mode=0o700)
    driver_module = load_module(root / 'adapter/simulation_driver.py', 'carryon_driver_configuration')
    scene = Path(config['benchmark']) / 'libero/libero/assets/scenes' / driver_module.SCENES[row['task_index']]
    driver = dict(safety_repo_root=str(Path(config['benchmark']).resolve()), reuse_tag=row['id'],
        ordinal=row['ordinal'], suite_name=row['suite'], level=row['level'], task_id=row['task_index'],
        seed=row['seed'], init_index=row['init_index'], problem_class=driver_module.PROBLEMS[row['task_index']],
        scene_xml_path=str(scene))
    driver_module.RawDriverConfig(**driver).validate()
    document = dict(config, driver=driver, supervisor=dict(settle_steps=10, max_control_steps=900,
        coordinator_namespace=str(private), coordinator_record_name='EPISODE_ENDPOINTS.json',
        public_pointer_path=str(output / 'COORDINATOR_ARTIFACT_POINTER.json')))
    config_file = output / 'EPISODE_CONFIG.json'
    write_json(config_file, document)
    with tempfile.TemporaryDirectory(prefix='carryon-safety-') as short:
        sock = str(Path(short) / 'sim.sock')
        from phase1_blind_supervisor import run_guarded_runner
        from phase1_protocol4_channel import protocol4_connect
        release_src = Path(__file__).resolve().parents[3]
        env = dict(os.environ)
        env.update({k: str(v) for k, v in config['sim_environment'].items()})
        env.update(PYTHONPATH=os.pathsep.join((str(release_src), str(root / 'support'), str(Path(config['benchmark']).resolve()))),
            LIBERO_CONFIG_PATH=str(Path(config['benchmark_config']).resolve()), PYTHONDONTWRITEBYTECODE='1')
        process = runtime = client = gate = None
        try:
            query, identity, client = get_policy(config, root, output)
            write_json(output / 'POLICY_IDENTITY.json', json_value(identity))
            with (output / 'simulator.log').open('xb') as log:
                process = subprocess.Popen([config['sim_python'], '-B', '-m',
                    'carryon.experiments.safety.episode', '--role', 'supervisor', '--config', str(config_file), '--socket', sock],
                    cwd=output, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 180
            while not Path(sock).exists():
                if process.poll() is not None:
                    raise RuntimeError('Simulator startup failed; see simulator.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Simulator startup timed out')
                time.sleep(.1)
            channel = protocol4_connect(sock, AUTH, 'carryon_safety_runner')
            method_env = {k: str(v) for k, v in config['method_environment'].items()}
            method_env['SAFE_POLICY_FAMILY'] = 'pi05' if pi05 else 'native8'
            runtime = RuntimeClient(root, output / 'perception', environment=method_env)
            gate = Pi05Gate(runtime, query, root, sft=policy_kind == 'pi05_sft') if pi05 else NativeEight(runtime, query, root)
            run_guarded_runner(channel, channel, gate.build, str(output / 'RUNNER_ATTESTATION.json'), True,
                               observation_consumer=gate.observe if pi05 else None)
            rc = process.wait(timeout=120)
            if rc:
                raise RuntimeError('Simulator exited with code ' + str(rc))
            write_json(output / 'RUNTIME_IDENTITY.json', json_value(gate.initial_identity))
            write_json(output / 'INTERFACE_RECEIPT.json', json_value({'policy': policy_kind, 'row': row,
                'control_actions': gate.step, 'halted': gate.halted, 'queries': gate.queries,
                'dispatches': gate.dispatches, 'integration': runtime.integration_records(),
                'final_method_identity': runtime.identity(), 'endpoint_read_during_execution': False}))
            endpoint = read_json(private / 'EPISODE_ENDPOINTS.json')
            if pi05:
                from score_execution_window import score_arm
                metrics = score_arm(dict(endpoint['full_horizon_record'], halted=gate.halted))
            else:
                metrics = {'task_success': endpoint['success'], 'paper_success': endpoint['paper_success_fixed'],
                    'execution_fixed_violation': endpoint['fixed_violation'], 'halt': int(gate.halted)}
            health = classify(runtime.identity())
            result = {**health, 'trajectory_complete': True, 'row': row, 'policy': policy_kind, 'metrics': metrics,
                'endpoint_sha256': sha256(private / 'EPISODE_ENDPOINTS.json'), 'model_answers_replayed': config['method_environment'].get('SAFE_ROLES_MODE','live') != 'live',
                'evaluation_mode': 'live' if config['method_environment'].get('SAFE_ROLES_MODE','live') == 'live' else 'controlled_role_diagnostic'}
            write_json(output / 'RESULT.json', result)
            return result
        except BaseException as exc:
            if gate is not None:
                write_json(output / 'PARTIAL_INTERFACE.json', json_value({'control_actions': gate.step,
                    'queries': gate.queries, 'dispatches': gate.dispatches, 'initial_identity': gate.initial_identity}))
            write_json(output / 'FAILURE.json', {'status': 'service_error', 'error': type(exc).__name__,
                'message': str(exc), 'traceback': traceback.format_exc(), 'row': row, 'policy': policy_kind})
            raise
        finally:
            if runtime is not None:
                runtime.close()
            if client is not None and callable(getattr(client, 'close', None)):
                client.close()
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('episode', 'supervisor'), default='episode')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--socket')
    args = parser.parse_args(argv)
    if args.role == 'supervisor':
        supervisor(args.config, args.socket)
    else:
        if not args.output:
            parser.error('--output is required')
        run(read_json(args.config), args.output)


if __name__ == '__main__':
    main()
