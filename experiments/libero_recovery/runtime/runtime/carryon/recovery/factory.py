"""Assemble recovery recording, perception, motion and policy execution."""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
if TYPE_CHECKING:
    from ..config import Config
    from ..jobs.manager import JobManager

def _build_memory_reflectors_from_env():

    enabled = os.environ.get('CARRYON_RECOVERY_ENABLE', '').strip() == '1'
    if not enabled:
        return (None, None)
    if os.environ.get('CARRYON_ROW6_JAM_Z_ENABLE', '').strip() == '1' or os.environ.get('CARRYON_BV2_GRASP_NO_HOLD_ENABLE', '').strip() == '1':
        raise ValueError('legacy action/B-v2 reflector modes are retired')
    api_key = 'unused' if os.environ.get('CARRYON_COLLECTION_ONLY') == '1' else os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise ValueError('Memory reflectors require OPENROUTER_API_KEY')

    def absolute_env(name: str, *, fresh: bool) -> Path:
        text = os.environ.get(name, '').strip()
        if not text:
            raise ValueError(f'{name} is required')
        path = Path(text)
        if not path.is_absolute():
            raise ValueError(f'{name} must be absolute')
        if fresh and path.exists():
            raise FileExistsError(f'{name} must be fresh')
        return path
    parameter_ledger = absolute_env('CARRYON_RECOVERY_PARAMETER_LEDGER_PATH', fresh=True)
    parameter_memory = absolute_env('CARRYON_RECOVERY_PARAMETER_MEMORY_PATH', fresh=False)
    semantic_ledger = absolute_env('CARRYON_RECOVERY_SEMANTIC_LEDGER_PATH', fresh=True)
    semantic_memory = absolute_env('CARRYON_RECOVERY_SEMANTIC_MEMORY_PATH', fresh=False)
    compat_manifest = absolute_env('CARRYON_RECOVERY_V7_COMPAT_MANIFEST', fresh=False)
    registry_sha = os.environ.get('CARRYON_REFERENCE_REGISTRY_ID', '').strip()
    if len(registry_sha) != 64:
        raise ValueError('Reference registry binding absent')
    from .jam_z_reflection import JamZReflector, OpenRouterJamZClient
    from .semantic_reflection import OpenRouterSemanticClient, SemanticMemoryReflector
    parameter = JamZReflector(OpenRouterJamZClient(api_key=api_key), ledger_path=parameter_ledger, persistent_memory_path=parameter_memory, write_trigger='postrow_failure_evidence')
    semantic = SemanticMemoryReflector(OpenRouterSemanticClient(api_key=api_key), memory_path=semantic_memory, ledger_path=semantic_ledger, compat_manifest_path=compat_manifest, registry_sha256=registry_sha)
    print(f'[recovery] parameter+semantic memory reflectors ACTIVE parameter_memory={parameter_memory} semantic_memory={semantic_memory}', flush=True)
    return (parameter, semantic)

def validate_live_grasp_config(primary: str, fallback_topdown: bool) -> str:
    """Return the canonical live detector or reject method-mixing configs."""
    mode = str(primary or 'graspgen').strip().lower()
    if mode != 'graspgen':
        raise ValueError("live recovery supports grasp_primary='graspgen' only; Contact-GraspNet is offline-only/unsupported")
    if bool(fallback_topdown):
        raise ValueError('live GraspGen recovery forbids grasp_fallback_topdown: detector failures must fail closed, not silently change grasp source')
    return mode

def dump_recovery_episode(record, job_key: str, output_dir: Optional[str]=None) -> None:
    """Persist the completed episode and its recording-quality fields."""
    from .recorder import data_tier_quality
    record.recording_quality = data_tier_quality(record)
    safe_key = re.sub('[^A-Za-z0-9_.-]+', '_', job_key).strip('_') or 'episode'
    out_dir = Path(output_dir or os.environ.get('CARRYON_RECOVERY_EPISODE_DIR', 'logs/recovery_episodes'))
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f'{safe_key}.json.tmp'
    dst = out_dir / f'{safe_key}.json'
    tmp.write_text(json.dumps(record.to_dict(), indent=2), encoding='utf-8')
    tmp.replace(dst)

def _camera_intrinsics_matrix(intr: dict) -> list[list[float]]:
    return [[float(intr['fx']), 0.0, float(intr['ppx'])], [0.0, float(intr['fy']), float(intr['ppy'])], [0.0, 0.0, 1.0]]

def _load_recovery_camera_specs(config: 'Config', gello, CameraSpec):
    """Load camera calibration for recovery records.

    Read extrinsics, name and role from recovery.camera_spec_path. Prefer live
    camera intrinsics, size and depth scale when available."""
    spec_path = str(config.recovery.camera_spec_path or '').strip()
    if not spec_path:
        return []
    p = Path(spec_path)
    try:
        raw = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        print(f'[recovery] camera calibration unavailable: {p} ({type(e).__name__}: {e})')
        return []
    items = raw.get('cameras') if isinstance(raw, dict) and isinstance(raw.get('cameras'), list) else [raw]
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name = str(config.recovery.camera_name or '').strip() or str(item.get('name', '')).strip() or 'overhead_rgb'
        role = str(item.get('role', 'external_fixed'))
        if role not in ('eye_in_hand', 'eye_to_hand', 'external_fixed'):
            role = 'external_fixed'
        extrinsics = item.get('extrinsics')
        if not (isinstance(extrinsics, list) and len(extrinsics) == 4 and all((isinstance(r, list) and len(r) == 4 for r in extrinsics))):
            print(f'[recovery] skip camera spec #{i}: invalid 4x4 extrinsics')
            continue
        intrinsics = item.get('intrinsics')
        width = item.get('width')
        height = item.get('height')
        depth_scale = float(item.get('depth_scale', 0.001))
        try:
            intr_live = gello.fetch_intrinsics(name)
            intrinsics = _camera_intrinsics_matrix(intr_live)
            width = int(intr_live.get('width', width))
            height = int(intr_live.get('height', height))
            depth_scale = float(intr_live.get('depth_scale', depth_scale))
        except Exception:
            pass
        if not (isinstance(intrinsics, list) and len(intrinsics) == 3 and all((isinstance(r, list) and len(r) == 3 for r in intrinsics))):
            print(f"[recovery] skip camera spec '{name}': invalid 3x3 intrinsics")
            continue
        if width is None or height is None:
            print(f"[recovery] skip camera spec '{name}': missing width/height")
            continue
        out.append(CameraSpec(name=name, role=role, width=int(width), height=int(height), intrinsics=[[float(x) for x in row] for row in intrinsics], extrinsics=[[float(x) for x in row] for row in extrinsics], depth_scale=float(depth_scale)))
    if not out:
        print('[recovery] no valid camera calibration loaded; episodes may fail calibration_present')
    return out

@dataclass
class RecoveryWiring:
    """What the host needs back from :func:`build_recovery`.

    ``recorder_registry`` is also consumed by the VLM-judge tick feed (so the
    judge timeline lands in the episode) and by shutdown teardown; ``bridges``
    are the zmq clients the host must close on exit (LINGER=0) so the shared zmq
    context can terminate. The actuating ``recovery_hook`` is armed directly on
    the passed-in ``JobManager``.
    """
    recorder_registry: Any | None = None
    bridges: list = field(default_factory=list)
    gello: Any | None = None

def build_recovery(config: 'Config', jm: 'JobManager') -> RecoveryWiring:
    """Assemble the recovery hook when recovery is enabled.

    A judge failure triggers takeover, recovery and handback. If assembly fails,
    the hook remains unset and the normal judge-driven release path is retained."""
    wiring = RecoveryWiring()
    try:
        from . import CameraSpec, EpisodeRecorderRegistry, GelloControlClient, GelloRecoveryBackend, RecoveryCoordinator, ZmqBridgeClient, default_registry
        _gello = GelloControlClient(config.recovery.gello_base_url, camera_url=config.recovery.gello_camera_url or None)
        _bridge = ZmqBridgeClient(config.recovery.bridge_host, config.recovery.bridge_rep_port, config.recovery.bridge_pull_port, config.recovery.bridge_pub_port)
        _rec_bridge = ZmqBridgeClient(config.recovery.bridge_host, config.recovery.bridge_rep_port, config.recovery.bridge_pull_port, config.recovery.bridge_pub_port)
        wiring.bridges = [_bridge, _rec_bridge]
        _default_cameras = _load_recovery_camera_specs(config, _gello, CameraSpec)
        _extrinsics = _default_cameras[0].extrinsics if _default_cameras else None
        if _default_cameras:
            names = ', '.join((c.name for c in _default_cameras))
            print(f'[recovery] camera calibration loaded: {names}')
        _planner = None
        _detector = None
        _detector_sources = {}
        if config.recovery.use_curobo:
            from . import CuroboConfig, build_motion_planner
            _planner = build_motion_planner(CuroboConfig(robot_config=config.recovery.curobo_robot_config, world_config=config.recovery.curobo_world_config or None, table_top_z=config.recovery.table_top_z, collision_activation_distance_m=config.recovery.collision_activation_m, seed=config.recovery.curobo_seed, robot_base_xyz_m=(config.recovery.robot_base_x_m, config.recovery.robot_base_y_m, config.recovery.robot_base_z_m)))
        if config.recovery.use_graspnet:
            from . import Sam3Config
            _sam3_kwargs = {}
            if config.recovery.sam3_model_path:
                _sam3_kwargs['model_path'] = config.recovery.sam3_model_path
            if config.recovery.sam3_python_exe:
                _sam3_kwargs['python_exe'] = config.recovery.sam3_python_exe
            if config.recovery.sam3_device:
                _sam3_kwargs['device'] = config.recovery.sam3_device
            if config.recovery.sam3_cuda_visible_devices:
                _sam3_kwargs['cuda_visible_devices'] = config.recovery.sam3_cuda_visible_devices
            if config.recovery.sam3_threshold >= 0:
                _sam3_kwargs['threshold'] = config.recovery.sam3_threshold
            if getattr(config.recovery, 'sam3_retry_threshold', -1.0) >= 0:
                _sam3_kwargs['retry_threshold'] = config.recovery.sam3_retry_threshold
            if config.recovery.sam3_mask_threshold >= 0:
                _sam3_kwargs['mask_threshold'] = config.recovery.sam3_mask_threshold
            if getattr(config.recovery, 'sam3_backend', ''):
                _sam3_kwargs['backend'] = config.recovery.sam3_backend
            _sam3_cfg = Sam3Config(**_sam3_kwargs) if _sam3_kwargs else None
            _grasp_primary = validate_live_grasp_config(getattr(config.recovery, 'grasp_primary', 'graspgen'), getattr(config.recovery, 'grasp_fallback_topdown', False))
            if _grasp_primary == 'graspgen':
                from . import GraspGenConfig, GraspGenDetector, MaskDepthGraspConfig, MaskDepthGraspDetector, build_sam3_segmenter
                _seg = build_sam3_segmenter(_sam3_cfg) if _sam3_cfg is not None else None
                _pointer = None
                if bool(getattr(config.recovery, 'enable_molmo', False)):
                    from . import MolmoConfig, build_molmo_pointer
                    _molmo_kwargs = {}
                    if config.recovery.molmo_model_path:
                        _molmo_kwargs['model_path'] = config.recovery.molmo_model_path
                    if config.recovery.molmo_python_exe:
                        _molmo_kwargs['python_exe'] = config.recovery.molmo_python_exe
                    if config.recovery.molmo_pythonpath_prepend:
                        _molmo_kwargs['pythonpath_prepend'] = config.recovery.molmo_pythonpath_prepend
                    if config.recovery.molmo_cuda_visible_devices:
                        _molmo_kwargs['cuda_visible_devices'] = config.recovery.molmo_cuda_visible_devices
                    _pointer = build_molmo_pointer(MolmoConfig(verbose=True, **_molmo_kwargs))
                from .catalog import load_object_catalog
                _catalog_path = '' if os.environ.get('CARRYON_DISABLE_OBJECT_CATALOG') == '1' else getattr(config.recovery, 'object_catalog_path', '') or ''
                _catalog = load_object_catalog(_catalog_path)
                _positionfix_memory = None
                _positionfix_path = os.environ.get('CARRYON_POSITIONFIX_MEMORY')
                if _positionfix_path:
                    from .positionfix_memory import load_positionfix_memory
                    _positionfix_memory = load_positionfix_memory(_positionfix_path)
                    print(f'[recovery] loaded FROZEN positionfix memory {_positionfix_memory.generation} ({len(_positionfix_memory.rules)} rule(s), hash {_positionfix_memory.content_hash[:12]})', flush=True)
                _topdown = MaskDepthGraspDetector(MaskDepthGraspConfig(physical_open_axis_angle_deg=config.recovery.open_axis_angle_deg, offset_world_m=(config.recovery.offset_world_x_m, config.recovery.offset_world_y_m, config.recovery.offset_world_z_m), local_table_clearance_m=config.recovery.local_table_clearance_m, center_top_band_m=config.recovery.grasp_center_top_band_m, grasp_below_top_enabled=config.recovery.grasp_below_top_enabled, grasp_below_top_frac=config.recovery.grasp_below_top_frac, grasp_below_top_max_m=config.recovery.grasp_below_top_max_m, world_from_camera=_extrinsics, spatial_disambig=config.recovery.grasp_spatial_disambig, disambig_prompt=config.recovery.grasp_disambig_prompt, disambig_max_dist_m=config.recovery.grasp_disambig_max_dist_m, molmo_anchor_gate_frac=config.recovery.grasp_molmo_anchor_gate_frac, verbose=True), segmenter=_seg, pointer=_pointer, positionfix_memory=_positionfix_memory)
                _topdown.object_catalog = _catalog
                _gg = GraspGenDetector(GraspGenConfig(host=str(config.recovery.graspgen_host), port=int(config.recovery.graspgen_port), request_timeout_ms=10000, num_grasps=64, min_grasps=1, max_tries=1, remove_outliers=False, world_from_camera=_extrinsics, disambig_prompt=config.recovery.grasp_disambig_prompt, disambig_max_dist_m=config.recovery.grasp_disambig_max_dist_m, molmo_anchor_gate_frac=config.recovery.grasp_molmo_anchor_gate_frac, table_z_world=config.recovery.table_top_z, pc_table_clearance_m=config.recovery.grasp_pc_table_clearance_m, verbose=True), segmenter=_seg, pointer=_pointer)
                _gg.object_catalog = _catalog
                from .backend import EXPLICIT_TOPDOWN_SOURCE_ALIASES
                _detector_sources = {alias: _topdown for alias in EXPLICIT_TOPDOWN_SOURCE_ALIASES}
                _detector = _gg
        _parameter_reflector, _semantic_reflector = _build_memory_reflectors_from_env()
        from . import BackendConfig
        _backend = GelloRecoveryBackend(_bridge, _gello, planner=_planner, detector=_detector, detector_sources=_detector_sources, jam_z_reflector=_parameter_reflector, semantic_reflector=_semantic_reflector, config=BackendConfig(grasp_camera=config.recovery.grasp_camera, tcp_offset_m=config.recovery.tcp_offset_m, stream_hz=config.recovery.exec_hz, max_exec_step_rad=config.recovery.exec_max_step_rad, rewind_escape_nudge_rad=float(getattr(config.recovery, 'rewind_escape_nudge_rad', 0.0) or 0.0), gripper_close_width=config.recovery.close_width_m, **{'verify_joint_tol_rad': float(config.recovery.grasp_verify_joint_tol_rad)} if float(config.recovery.grasp_verify_joint_tol_rad) > 0.0 else {}))
        import os as _os_env
        _goalset_env = {k: v for k, v in _os_env.environ.items() if k.startswith('CARRYON_GOALSET_')}
        if _goalset_env:
            print(f'[recovery] goalset env overrides ACTIVE: {_goalset_env}', flush=True)
        if bool(getattr(config.recovery, 'verify_identity', True)):
            try:
                from ..planner import build_reflection_vlm_client
                _backend.identity_vlm = build_reflection_vlm_client(config)
                _visual_manifest = _os_env.environ.get('CARRYON_VISUAL_OBJECT_MEMORY_MANIFEST', '').strip()
                if _visual_manifest:
                    from .visual_object_memory import VisualMemorySomSelector, VisualObjectMemory
                    _visual_debug = _os_env.environ.get('CARRYON_VISUAL_OBJECT_MEMORY_DEBUG_DIR', '').strip()
                    _memory_recovery = _os_env.environ.get('CARRYON_VISUAL_OBJECT_MEMORY_POINT_RECOVERY', '').strip().lower() in {'1', 'true', 'yes', 'on'}
                    _molmo_targeted = _os_env.environ.get('CARRYON_VISUAL_OBJECT_MEMORY_MOLMO_POINTER', '').strip().lower() in {'1', 'true', 'yes', 'on'}
                    _semantic_path = _os_env.environ.get('CARRYON_SEMANTIC_MEMORY_SNAPSHOT', '').strip()
                    _semantic_snapshot = None
                    _semantic_environment = 'sim'
                    _semantic_seed = None
                    _semantic_episode_key = ''
                    if _semantic_path:
                        from .semantic_memory import SemanticMemorySnapshot
                        _semantic_environment = _os_env.environ.get('CARRYON_SEMANTIC_ENVIRONMENT', '').strip().lower()
                        if _semantic_environment not in {'sim', 'real'}:
                            raise RuntimeError('semantic environment must be sim or real')
                        _seed_raw = _os_env.environ.get('CARRYON_SEMANTIC_FORMAL_SEED', '').strip()
                        _semantic_episode_key = _os_env.environ.get('CARRYON_SEMANTIC_EPISODE_KEY', '').strip()
                        if not _seed_raw or not _semantic_episode_key:
                            raise RuntimeError('semantic formal seed and episode key required')
                        _semantic_seed = int(_seed_raw)
                        _semantic_snapshot = SemanticMemorySnapshot(_semantic_path)
                    if _memory_recovery and _molmo_targeted:
                        raise RuntimeError('memory-conditioned Judge point and Molmo pointer cannot both be active')
                    if _molmo_targeted and _pointer is None:
                        raise RuntimeError('Molmo target-directed proposal requested but pointer unavailable')
                    if _molmo_targeted:
                        _sam3 = getattr(_topdown, 'segmenter', None)
                        for _name, _service in (('molmo', _pointer), ('point_sam3', _sam3)):
                            _ensure = getattr(_service, '_ensure_server', None)
                            if not callable(_ensure):
                                raise RuntimeError(f'{_name} persistent prewarm unavailable')
                            import time as _time
                            _started = _time.perf_counter()
                            if not _ensure():
                                raise RuntimeError(f"{_name} persistent prewarm failed: {getattr(_service, 'last_error', '')}")
                            print(f'[recovery] prewarm {_name} startup_model_load_ms={(_time.perf_counter() - _started) * 1000:.3f}', flush=True)
                    _topdown.visual_object_selector = VisualMemorySomSelector(VisualObjectMemory(_visual_manifest), _backend.identity_vlm, debug_dir=_visual_debug, memory_conditioned_recovery=_memory_recovery, target_pointer=_pointer, molmo_target_directed=_molmo_targeted, semantic_memory=_semantic_snapshot, semantic_environment=_semantic_environment, semantic_formal_seed=_semantic_seed, semantic_episode_key=_semantic_episode_key)
                    _backend.semantic_scene_bundle_required = bool(_semantic_snapshot is not None)
                    print(f"[recovery] registry visual memory + SoM selector ACTIVE manifest={_visual_manifest} point_recovery={_memory_recovery} molmo_targeted={_molmo_targeted} semantic_snapshot={_semantic_path or 'off'}", flush=True)
            except Exception as _e:
                if _os_env.environ.get('CARRYON_VISUAL_OBJECT_MEMORY_MOLMO_POINTER', '').strip().lower() in {'1', 'true', 'yes', 'on'} or _os_env.environ.get('CARRYON_SEMANTIC_MEMORY_SNAPSHOT', '').strip():
                    raise
                print(f'[recovery.factory] identity VLM unavailable: {_e}')
        from . import GelloFrameSource
        _capture_names = [c.name for c in _default_cameras]
        for _extra in config.recovery.record_cameras or []:
            if _extra and _extra not in _capture_names:
                _capture_names.append(_extra)
        _frame_source = GelloFrameSource(_gello, _capture_names) if _capture_names else None
        if config.recovery.record_cameras:
            print(f"[recovery] frame recording cameras: {', '.join(_capture_names)}")
        recorder_registry = EpisodeRecorderRegistry(state_source=_rec_bridge, default_cameras=_default_cameras, frame_source=_frame_source)
        recorder_registry.start()
        wiring.recorder_registry = recorder_registry
        wiring.gello = _gello

        def _subtask_instruction(rec) -> str:
            pr = jm.plans.get(getattr(rec, 'plan_id', ''))
            if pr is not None:
                for s in pr.plan.subtasks:
                    if s.id == getattr(rec, 'subtask_id', None):
                        return s.instruction
            return ''

        def _object_hint(rec) -> str:
            jk = getattr(rec, 'job_key', '')
            judge_obj = recorder_registry.get_recovery_object(jk) if jk else ''
            return judge_obj or config.recovery.object_hint or _subtask_instruction(rec)

        def _target_px(rec):
            jk = getattr(rec, 'job_key', '')
            return recorder_registry.get_recovery_point(jk) if jk else None

        def _record_provider(rec):
            return recorder_registry.build(getattr(rec, 'job_key', ''))

        def _log(msg: str, plan_id, subtask_id) -> None:
            prefix = f"[recovery {plan_id or '?'}:{subtask_id or '?'}]"
            print(f'{prefix} {msg}', flush=True)
            jm._sys_event(msg, plan_id, subtask_id)

        def _on_recovery_complete(rec, result) -> None:
            job_key = getattr(rec, 'job_key', '')
            if recorder_registry.set_recovery(job_key, result.invocation):
                record = recorder_registry.build(job_key)
                if record is not None:
                    dump_recovery_episode(record, job_key, output_dir=os.environ.get('CARRYON_RECOVERY_EPISODE_DIR', 'logs/recovery_episodes'))
        from . import RecoveryConfig, load_authored_policy
        _recovery_registry = default_registry()
        _authored = os.environ.get('CARRYON_POLICY_JSON') or config.recovery.policy_json
        _pinned_policy_version: Optional[str] = None
        for _pth in [_authored] if isinstance(_authored, str) else _authored or []:
            name, version, ok, problems = load_authored_policy(_recovery_registry, _pth)
            if ok:
                _pinned_policy_version = version
                print(f'[recovery] loaded authored policy {name}@{version} from {_pth}', flush=True)
            else:
                _fallthrough = 'the GATED SEED-STRATEGY LIBRARY (use_strategy_selector=true)' if bool(getattr(config.recovery, 'use_strategy_selector', False)) else 'the built-in LATEST recover_hold_lost policy'
                print(f'[recovery] *** REJECTED authored policy pin from {_pth}: {problems}. NO version is pinned -> recovery will run {_fallthrough} instead of the intended pin. If this pin was meant to define the condition, FIX IT (e.g. bump the version so it does not collide with a built-in) before trusting the sweep. ***', flush=True)
        _reflector = None
        import os as _os
        _refl_cfg = bool(getattr(config.recovery, 'use_reflector', False))
        _refl_env = _os.environ.get('CARRYON_REFLECTOR') == '1'
        if _refl_cfg or _refl_env:
            print('[recovery] reflector ACTIVE via ' + ('config.recovery.use_reflector' if _refl_cfg else 'CARRYON_REFLECTOR=1 env'), flush=True)
            try:
                from ..planner import build_reflection_vlm_client
                from .reflector_live import OnlineStrategyReflector
                _reflector = OnlineStrategyReflector(build_reflection_vlm_client(config), log=lambda m: print(f'[reflector] {m}', flush=True))
                print('[recovery] online reflector armed (C2)', flush=True)
            except Exception as _e:
                print(f'[recovery] reflector unavailable ({type(_e).__name__}: {_e}); running without reflection (C1)', flush=True)
        _strategy_specs: tuple = ()
        _strategy_canonical = None
        _lib_path = os.environ.get('CARRYON_STRATEGY_JSON') or getattr(config.recovery, 'strategy_json', None)
        _sel_env = os.environ.get('CARRYON_USE_STRATEGY_SELECTOR')
        _use_sel = _sel_env == '1' if _sel_env is not None else bool(getattr(config.recovery, 'use_strategy_selector', False))
        if _lib_path and _use_sel:
            try:
                from .library import StrategyLibrary, merge_for_selector
                _lib = StrategyLibrary.load(_lib_path, require_frozen=True)
                _, _strategy_canonical = merge_for_selector(_lib)
                _strategy_specs = tuple(_lib.strategies)
                print(f'[recovery] loaded FROZEN strategy library {_lib.version} (round {_lib.reflection_round}, {len(_lib.strategies)} learned spec(s), hash {_lib.content_hash[:12]})', flush=True)
            except Exception as _e:
                raise RuntimeError(f'failed to load frozen strategy library {_lib_path!r}: {type(_e).__name__}: {_e}') from _e
        elif _lib_path and (not _use_sel):
            print('[recovery] strategy_json set but use_strategy_selector=false; frozen library IGNORED (selector off)', flush=True)
        _coordinator = RecoveryCoordinator(backend=_backend, gello=_gello, registry=_recovery_registry, record_provider=_record_provider, object_hint_provider=_object_hint, target_px_provider=_target_px, resume_instruction_provider=_subtask_instruction, config=RecoveryConfig(verify_identity=os.environ.get('CARRYON_VERIFY_IDENTITY', '1') == '1' and bool(getattr(config.recovery, 'verify_identity', True)), library_oneshot=_os.environ.get('CARRYON_LIBRARY_ONESHOT') == '1' or bool(getattr(config.recovery, 'library_oneshot', False)), reflect_turns=int(_os.environ.get('CARRYON_REFLECT_TURNS') or getattr(config.recovery, 'reflect_turns', 1)), gripper_close_width=config.recovery.close_width_m, gripper_hold_check=getattr(config.recovery, 'hold_check', 'command_relative'), grasp_z_offset=float(os.environ.get('CARRYON_GRASP_Z_OFFSET', config.recovery.grasp_z_offset_m)), grasp_x_offset=float(os.environ.get('CARRYON_GRASP_X_OFFSET', config.recovery.grasp_x_offset_m)), grasp_y_offset=float(os.environ.get('CARRYON_GRASP_Y_OFFSET', config.recovery.grasp_y_offset_m)), grasp_yaw_offset_deg=float(os.environ.get('CARRYON_GRASP_YAW_OFFSET_DEG', config.recovery.grasp_yaw_offset_deg)), init_q=list(config.recovery.init_q)), policy_name=config.recovery.policy_name, policy_version=_pinned_policy_version, enabled=True, max_attempts=int(_os.environ.get('CARRYON_MAX_ATTEMPTS') or config.recovery.max_attempts), use_strategy_selector=_use_sel, reflector=_reflector, intervention_rule=os.environ.get('CARRYON_INTERVENTION_RULE') == '1', strict_reflected_library=os.environ.get('CARRYON_STRICT_REFLECTED_LIBRARY') == '1', strategy_library_specs=_strategy_specs, strategy_canonical_by_bucket=_strategy_canonical, strategy_handback_mode=os.environ.get('CARRYON_STRATEGY_HANDBACK_MODE') or None, log=_log, on_complete=_on_recovery_complete)
        if _planner is not None and _detector is not None:
            jm.recovery_hook = _coordinator.on_judge_failure
            print('[recovery] actuating hook armed (planner + detector present)')
        else:
            missing = []
            if _planner is None:
                missing.append('motion planner (recovery.use_curobo)')
            if _detector is None:
                missing.append('grasp detector (recovery.use_graspnet)')
            print('[recovery] recorder ON, actuating hook DISARMED -- missing ' + ' + '.join(missing) + ' (no arm takeover on judge failure)')
    except Exception as e:
        wiring.recorder_registry = None
        print(f'[recovery] disabled (wiring failed): {type(e).__name__}: {e}')
    return wiring
