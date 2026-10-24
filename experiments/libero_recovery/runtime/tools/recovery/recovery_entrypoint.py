"""Run a LIBERO episode with the configured recovery runtime."""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]

class _RuntimeFlags:
    """Atomic structured evidence from the actual formal treatment process."""
    _NAMES = ('destination_transport_started', 'vla_started', 'judge_started')

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._flags = {name: False for name in self._NAMES}
        self._transitions: list[dict] = []
        if self.path.exists():
            raise RuntimeError(f'refuse existing formal runtime flags: {self.path}')
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        payload = {'schema': 'destination-suffix-formal-runtime-flags/v1', **self._flags, 'transitions': self._transitions}
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary, self.path)

    def flip(self, name: str, **evidence) -> None:
        if name not in self._flags:
            raise KeyError(name)
        with self._lock:
            if self._flags[name]:
                return
            self._flags[name] = True
            self._transitions.append({'flag': name, 'wall_ts': time.time(), **evidence})
            self._write()

def _load_json_env(name: str) -> dict:
    path = Path(os.environ[name]).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise RuntimeError(f'{name} must name a JSON object')
    return value

def main() -> int:
    assembly = _load_json_env('CARRYON_FORMAL_ASSEMBLY_CONFIG')
    ipc_dir = Path(os.environ['CARRYON_FORMAL_CUROBO_IPC_DIR']).resolve()
    runtime_flags = _RuntimeFlags(Path(os.environ['CARRYON_FORMAL_RUNTIME_FLAGS']).resolve())
    import carryon.config as config_module
    import carryon.recovery as recovery
    from carryon.judge.supervisor import VLMJudgeSupervisor
    from carryon.recovery.arbiter import GelloArbiter
    from carryon.recovery.curobo_subprocess import CuroboSubprocessClient, CuroboSubprocessConfig
    original_from_file = config_module.Config.from_file.__func__

    def formal_from_file(cls, path):
        cfg = original_from_file(cls, path)
        cfg.recovery.sam3_cuda_visible_devices = str(assembly['resources']['sam3_gpu'])
        cfg.recovery.molmo_cuda_visible_devices = str(assembly['resources']['molmo_gpu'])
        cfg.recovery.verify_identity = True
        cfg.recovery.use_strategy_selector = False
        return cfg
    config_module.Config.from_file = classmethod(formal_from_file)
    original_handback = GelloArbiter.handback

    def formal_handback(self):
        ok = bool(original_handback(self))
        if ok:
            runtime_flags.flip('vla_started', resumed_job_id=self.resumed_job_id)
        return ok
    GelloArbiter.handback = formal_handback
    original_start_job = VLMJudgeSupervisor.start_job

    def formal_start_job(self, *args, **kwargs):
        runtime_flags.flip('judge_started', provider=getattr(self.config, 'provider', None), post_handback=bool(kwargs.get('post_handback', False)))
        return original_start_job(self, *args, **kwargs)
    VLMJudgeSupervisor.start_job = formal_start_job
    planner_clients = []

    def build_subprocess_planner(curobo_cfg):
        client = CuroboSubprocessClient(CuroboSubprocessConfig(python_exe=assembly['resources']['curobo_python_exe'], runner_path=str((REPO / assembly['resources']['curobo_runner_path']).resolve()), repo_root=str(REPO / 'runtime'), ipc_dir=str(ipc_dir), cuda_visible_devices=str(assembly['resources']['curobo_gpu']), startup_timeout_s=float(assembly['resources']['curobo_startup_timeout_s']), request_timeout_s=float(assembly['resources']['curobo_request_timeout_s']), planner_config={'robot_config': curobo_cfg.robot_config, 'world_config': curobo_cfg.world_config, 'table_top_z': curobo_cfg.table_top_z, 'collision_activation_distance_m': curobo_cfg.collision_activation_distance_m, 'seed': curobo_cfg.seed, 'robot_base_xyz_m': list(curobo_cfg.robot_base_xyz_m)}))
        if not client._ensure_server():
            raise RuntimeError(client.last_error or 'cuRobo subprocess prewarm failed')
        planner_clients.append(client)
        print(f'[formal-assembly] cuRobo subprocess READY pid={client.pid}', flush=True)
        return client
    recovery.build_motion_planner = build_subprocess_planner
    base_backend = recovery.GelloRecoveryBackend

    class FormalDestinationBackend(base_backend):
        """Expose one generic top-down approach through the configured planner."""

        def goto_place(self, *args, **kwargs):
            runtime_flags.flip('destination_transport_started', standoff_only=kwargs.get('standoff_only'), closed_gripper=True)
            return super().goto_place(*args, **kwargs)

        def goto_grasp_topdown_family(self, frozen_target, fallback_grasps, **kwargs):
            kwargs['yaw_degs'] = ()
            kwargs['fallback_max_tilt_deg'] = 0.0
            return super().goto_grasp_topdown_family(frozen_target, [], **kwargs)
    recovery.GelloRecoveryBackend = FormalDestinationBackend
    print('[recovery-runtime] ready: generic_topdown_single_approach, z_reflection=off, retry=off, bucket_specialization=off', flush=True)
    from carryon.main import main as carryon_main
    try:
        return int(carryon_main() or 0)
    finally:
        for client in planner_clients:
            client.close()
if __name__ == '__main__':
    raise SystemExit(main())
