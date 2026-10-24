"""Persistent cuRobo planner worker for the LIBERO subprocess boundary."""
from __future__ import annotations
import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any
STOP = False

def _stop(_signum, _frame) -> None:
    global STOP
    STOP = True

def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def _safe(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _dispatch(planner, method: str, args: dict[str, Any]):
    if method == 'plan_to_q':
        return planner.plan_to_q(args['q_start'], args['q_goal'])
    if method == 'plan_to_pose':
        return planner.plan_to_pose(args['q_start'], args['position'], args['quaternion_wxyz'])
    if method == 'plan_transport_segment':
        return planner.plan_transport_segment(args['q_start'], args['position'], args['quaternion_wxyz'], phase=args['phase'])
    if method == 'ee_poses':
        return planner.ee_poses(args['qs'])
    if method == 'plan_grasp_goalset':
        from carryon.recovery.schema import GraspProposal
        grasps = [GraspProposal(**row) for row in args['grasps']]
        return planner.plan_grasp_goalset(args['q_start'], grasps, tcp_offset_m=float(args['tcp_offset_m']), pre_grasp_dis=float(args['pre_grasp_dis']), plan_grasp_to_lift=bool(args['plan_grasp_to_lift']))
    if method == 'ee_pose':
        return planner.ee_pose(args['q'])
    if method == 'set_scene_pointcloud':
        return planner.set_scene_pointcloud(args['points_world'], **args.get('kwargs', {}))
    if method == 'clear_scene':
        planner.clear_scene()
        return True
    raise ValueError(f'unsupported planner method: {method}')

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', required=True)
    parser.add_argument('--ipc-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    ipc = args.ipc_dir.resolve()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    from carryon.recovery.geometry import CuroboConfig, CuroboMotionPlanner
    planner = CuroboMotionPlanner(CuroboConfig(**config))
    import torch
    import curobo
    runtime = {'torch_version': str(torch.__version__), 'curobo_module': str(Path(curobo.__file__).resolve()), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    _atomic_json(ipc / 'READY.json', {'schema': 'carryon-curobo-planner-ready/v1', 'ready': True, 'pid': os.getpid(), 'ready_wall_ts': time.time(), **runtime})
    handled: set[str] = set()
    while not STOP:
        for request_path in sorted(ipc.glob('REQUEST_*.json')):
            request_id = request_path.stem.removeprefix('REQUEST_')
            if request_id in handled:
                continue
            handled.add(request_id)
            response_path = ipc / f'RESPONSE_{request_id}.json'
            started = time.time()
            try:
                request = json.loads(request_path.read_text(encoding='utf-8'))
                if request.get('request_id') != request_id:
                    raise ValueError('request id/path mismatch')
                result = _dispatch(planner, str(request['method']), dict(request.get('args', {})))
                payload = {'schema': 'carryon-curobo-planner-response/v1', 'request_id': request_id, 'ok': True, 'result': _safe(result), 'planner_error': _safe(getattr(planner, 'last_error', None)), 'started_wall_ts': started, 'completed_wall_ts': time.time()}
            except Exception as exc:
                payload = {'schema': 'carryon-curobo-planner-response/v1', 'request_id': request_id, 'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(), 'started_wall_ts': started, 'completed_wall_ts': time.time()}
            _atomic_json(response_path, payload)
        time.sleep(0.01)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
