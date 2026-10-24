"""Configuration loaded from a YAML file.

The YAML file contains all runtime configuration. Sensitive fields (API keys,
auth tokens) may either be written inline (when the YAML file itself is kept
out of version control via .gitignore) or sourced from environment variables
using shell-style ``${VAR}`` / ``${VAR:-default}`` placeholders -- see
:func:`expand_env_vars`. The latter is strongly recommended for any YAML
file that is committed to the repo (e.g. ``config.*.example.yaml``).

See ``config.example.yaml`` for the template.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field, fields
from typing import Any
CONFIG_ENV_VAR = 'CARRYON_CONFIG'
DEFAULT_CONFIG_PATH = './config.yaml'
_ENV_VAR_RE = re.compile('\\$\\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\\}')

def expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside a YAML-parsed
    structure (dict / list / scalar).

    Semantics (mirroring POSIX shell / docker-compose):

    - ``${VAR}``           -> raises :class:`KeyError` if ``VAR`` is not set in
      the process environment. This makes missing secrets fail fast at config
      load time instead of surfacing as an obscure HTTP 401 later.
    - ``${VAR:-default}``  -> substitutes ``default`` (possibly empty) when
      ``VAR`` is unset or empty.

    Only *string* values are inspected; dict keys and non-string scalars pass
    through untouched. Use this to keep secrets out of committed YAML, e.g.::

        # config.example.yaml
        planner_vlm:
          api_key: "${OPENROUTER_API_KEY}"

    Then export ``OPENROUTER_API_KEY`` in your shell (or .env, etc.) before
    starting CarryOn.
    """
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return _expand_env_in_str(value)
    return value

def _expand_env_in_str(s: str) -> str:
    if '${' not in s:
        return s
    missing: list[str] = []

    def repl(m: 're.Match[str]') -> str:
        name = m.group(1)
        default = m.group(2)
        val = os.environ.get(name)
        if val is not None and val != '':
            return val
        if default is not None:
            return default
        missing.append(name)
        return ''
    out = _ENV_VAR_RE.sub(repl, s)
    if missing:
        unique = sorted(set(missing))
        raise KeyError('missing required environment variable(s) referenced in config: ' + ', '.join(unique) + '. Export them before starting CarryOn, or use ${VAR:-default} to provide a fallback.')
    return out

@dataclass
class RecoverySettings:
    """User-facing ``recovery:`` YAML section (the **optional** physical
    recovery loop). Grouped into its own dataclass because it is ~1/3 of all
    config fields yet only the recovery subsystem consumes it.

    Distinct from :class:`carryon.recovery.policy.RecoveryConfig`, which is the
    *internal runtime* policy/orchestrator config; these app-level settings are
    translated into that (and BackendConfig / CuroboConfig / ...) during wiring.
    """
    enabled: bool = False
    gello_base_url: str = 'http://127.0.0.1:18080'
    gello_camera_url: str = ''
    policy_name: str = 'recover_hold_lost'
    object_hint: str = ''
    max_attempts: int = 1
    policy_json: Any = None
    bridge_host: str = '127.0.0.1'
    bridge_rep_port: int = 6003
    bridge_pull_port: int = 6004
    bridge_pub_port: int = 6005
    camera_spec_path: str = 'tools/calibration/camera_spec.json'
    camera_name: str = ''
    record_cameras: list[str] = field(default_factory=list)
    use_curobo: bool = False
    curobo_robot_config: str = 'franka.yml'
    curobo_world_config: str = ''
    curobo_seed: int = 123
    use_graspnet: bool = False
    graspnet_ckpt: str = ''
    sam3_model_path: str = ''
    sam3_python_exe: str = ''
    sam3_device: str = 'auto'
    sam3_cuda_visible_devices: str = ''
    sam3_backend: str = 'native'
    enable_molmo: bool = False
    verify_identity: bool = True
    library_oneshot: bool = False
    reflect_turns: int = 1
    object_catalog_path: str = ''
    molmo_model_path: str = ''
    molmo_python_exe: str = ''
    molmo_pythonpath_prepend: str = ''
    molmo_cuda_visible_devices: str = ''
    sam3_threshold: float = -1.0
    sam3_retry_threshold: float = -1.0
    sam3_mask_threshold: float = -1.0
    graspnet_repo_dir: str = ''
    graspnet_python_exe: str = ''
    grasp_fallback_topdown: bool = False
    grasp_primary: str = 'graspgen'
    graspgen_host: str = '127.0.0.1'
    graspgen_port: int = 5557
    grasp_camera: str = 'overhead_rgb'
    tcp_offset_m: float = 0.21
    close_width_m: float = 0.012
    hold_check: str = 'command_relative'
    use_strategy_selector: bool = False
    strategy_json: Any = None
    use_reflector: bool = False
    rewind_escape_nudge_rad: float = 0.0
    grasp_z_offset_m: float = 0.0
    grasp_x_offset_m: float = 0.0
    grasp_y_offset_m: float = 0.0
    grasp_yaw_offset_deg: float = 0.0
    exec_hz: float = 40.0
    exec_max_step_rad: float = 0.01
    grasp_verify_joint_tol_rad: float = 0.0
    offset_world_x_m: float = 0.0
    offset_world_y_m: float = -0.01
    offset_world_z_m: float = -0.032
    local_table_clearance_m: float = 0.0
    grasp_center_top_band_m: float = 0.0
    grasp_below_top_enabled: bool = False
    grasp_below_top_frac: float = 0.32
    grasp_below_top_max_m: float = 0.05
    grasp_spatial_disambig: bool = True
    grasp_disambig_prompt: str = ''
    grasp_disambig_max_dist_m: float = 0.2
    grasp_molmo_anchor_gate_frac: float = 0.0
    grasp_pc_table_clearance_m: float = 0.02
    open_axis_angle_deg: float = 135.0
    min_primary_score: float = 0.1
    min_primary_z_above_table_m: float = 0.005
    table_top_z: float = -0.02
    collision_activation_m: float = 0.01
    robot_base_x_m: float = 0.0
    robot_base_y_m: float = 0.0
    robot_base_z_m: float = 0.0
    init_q: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, -1.566, 0.0, 1.579, 0.0])

    @classmethod
    def from_dict(cls, section: dict[str, Any] | None) -> 'RecoverySettings':
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (section or {}).items() if k in allowed and v is not None}
        return cls(**kwargs)

@dataclass
class Config:
    planner_vlm_provider: str = 'openrouter'
    planner_vlm_model: str = 'openai/gpt-5.2'
    planner_vlm_temperature: float = 0.2
    planner_vlm_api_key: str | None = None
    planner_vlm_base_url: str = 'https://openrouter.ai/api/v1'
    planner_vlm_timeout_s: float = 60.0
    planner_vlm_cameras: list[dict[str, Any]] | None = None
    planner_vlm_mock_mode: str = 'echo'
    planner_vlm_mock_responses: list | None = None
    planner_vlm_http_referer: str | None = None
    planner_vlm_x_title: str | None = None
    scene_describer_mock_locations: dict[str, Any] = field(default_factory=dict)
    reflection_vlm_provider: str | None = None
    reflection_vlm_model: str | None = None
    reflection_vlm_temperature: float | None = None
    reflection_vlm_api_key: str | None = None
    reflection_vlm_base_url: str | None = None
    reflection_vlm_timeout_s: float | None = None
    memory_max_tokens: int = 32000
    executor_max_retries: int = 2
    executor_react_on_failure: bool = True
    executor_max_concurrency: int = 8
    executor_retry_backoff_base_s: float = 0.5
    executor_retry_backoff_cap_s: float = 5.0
    judge_release_timeout_s: float = 10.0
    endpoints_config: str = './endpoints.yaml'
    locations_config: str = './locations.yaml'
    default_transport: str = 'http'
    http_default_timeout: float = 10.0
    vla_auth_token: str | None = None
    ros_force_enable: bool = False
    ros_node_name: str = 'carryon_ros_transport'
    ros_domain_id: int | None = None
    ros_qos_depth: int = 10
    ros_spin_period_s: float = 0.05
    ros_cancel_topic_suffix: str = '/cancel'
    mock_auto_callback_after_s: float = 0.0
    result_channel: str = 'webhook'
    callback_host: str = '0.0.0.0'
    callback_port: int = 9000
    callback_public_url: str = 'http://127.0.0.1:9000'
    callback_auth_token: str | None = None
    callback_ros_enabled: bool = False
    callback_ros_topic: str = '/carryon/callback'
    callback_ros_node_name: str = 'carryon_ros_callback'
    callback_ros_qos_depth: int = 50
    job_default_deadline_s: float = 300.0
    job_tick_interval_s: float = 5.0
    shutdown_grace_s: float = 0.0
    planner_log_path: str = './logs/planner.log'
    planner_log_stdout: bool = True
    transport_log_path: str = './logs/transport.log'
    transport_log_stdout: bool = True
    vlm_judge_enabled: bool = False
    vlm_judge_settings: dict[str, Any] = field(default_factory=dict)
    vlm_judge_ignore_executor_terminal: bool = True
    vlm_judge_applies_to_tags: list[str] = field(default_factory=list)
    recovery: RecoverySettings = field(default_factory=RecoverySettings)
    memory_data_dir: str = '.carryon/memory'
    scene_memory_enabled: bool = True
    scene_memory_inject_top_k: int = 20
    user_memory_enabled: bool = True
    user_memory_summarize_on_finish: bool = True
    user_memory_min_messages_for_summary: int = 3
    user_id: str = 'default'

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Config':
        """Build Config from a nested dict parsed from YAML.

        Supports both a flat shape and a grouped shape::

            # flat
            planner_vlm_model: openai/gpt-5.2
            planner_vlm_api_key: sk-or-...

            # grouped (preferred for readability)
            planner_vlm:
              provider: openrouter
              model: openai/gpt-5.2
              api_key: ${OPENROUTER_API_KEY}
            executor:
              max_retries: 2
              react_on_failure: true

        Unknown keys are ignored (logged by caller if desired) so old/new
        fields don't crash each other.
        """
        flat = _flatten(data or {})
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in flat.items() if k in allowed and v is not None}
        cfg = cls(**kwargs)
        vlm_judge_raw = (data or {}).get('vlm_judge')
        if isinstance(vlm_judge_raw, dict):
            cfg.vlm_judge_settings = {k: v for k, v in vlm_judge_raw.items() if k in ('vlm', 'judge', 'camera', 'cameras', 'debug', 'mock')}
        recovery_raw = (data or {}).get('recovery')
        if isinstance(recovery_raw, dict):
            cfg.recovery = RecoverySettings.from_dict(recovery_raw)
        _validate(cfg)
        return cfg

    @classmethod
    def from_file(cls, path: str | None=None) -> 'Config':
        """Load Config from a YAML file.

        Lookup precedence for ``path``:
            1. the ``path`` argument if given
            2. the ``CARRYON_CONFIG`` env var
            3. ``./config.yaml``

        Raises FileNotFoundError if none of the candidate files exists.
        """
        chosen = path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
        if not os.path.exists(chosen):
            raise FileNotFoundError(f'config file not found: {chosen}. Create one from config.example.yaml or set {CONFIG_ENV_VAR}.')
        import yaml
        with open(chosen, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f'config file {chosen}: top-level must be a mapping')
        data = expand_env_vars(data)
        return cls.from_dict(data)
_GROUP_ALIASES: dict[tuple[str, str], str] = {('planner_vlm', 'provider'): 'planner_vlm_provider', ('planner_vlm', 'model'): 'planner_vlm_model', ('planner_vlm', 'temperature'): 'planner_vlm_temperature', ('planner_vlm', 'api_key'): 'planner_vlm_api_key', ('planner_vlm', 'base_url'): 'planner_vlm_base_url', ('planner_vlm', 'timeout_s'): 'planner_vlm_timeout_s', ('planner_vlm', 'mock_mode'): 'planner_vlm_mock_mode', ('planner_vlm', 'mock_responses'): 'planner_vlm_mock_responses', ('planner_vlm', 'cameras'): 'planner_vlm_cameras', ('planner_vlm', 'http_referer'): 'planner_vlm_http_referer', ('planner_vlm', 'x_title'): 'planner_vlm_x_title', ('scene_describer', 'mock_locations'): 'scene_describer_mock_locations', ('reflection_vlm', 'provider'): 'reflection_vlm_provider', ('reflection_vlm', 'model'): 'reflection_vlm_model', ('reflection_vlm', 'temperature'): 'reflection_vlm_temperature', ('reflection_vlm', 'api_key'): 'reflection_vlm_api_key', ('reflection_vlm', 'base_url'): 'reflection_vlm_base_url', ('reflection_vlm', 'timeout_s'): 'reflection_vlm_timeout_s', ('memory', 'max_tokens'): 'memory_max_tokens', ('executor', 'max_retries'): 'executor_max_retries', ('executor', 'react_on_failure'): 'executor_react_on_failure', ('executor', 'max_concurrency'): 'executor_max_concurrency', ('executor', 'retry_backoff_base_s'): 'executor_retry_backoff_base_s', ('executor', 'retry_backoff_cap_s'): 'executor_retry_backoff_cap_s', ('executor', 'judge_release_timeout_s'): 'judge_release_timeout_s', ('vlm_judge', 'release_timeout_s'): 'judge_release_timeout_s', ('endpoints', 'config'): 'endpoints_config', ('endpoints', 'default_transport'): 'default_transport', ('locations', 'config'): 'locations_config', ('http', 'default_timeout'): 'http_default_timeout', ('http', 'vla_auth_token'): 'vla_auth_token', ('ros', 'force_enable'): 'ros_force_enable', ('ros', 'node_name'): 'ros_node_name', ('ros', 'domain_id'): 'ros_domain_id', ('ros', 'qos_depth'): 'ros_qos_depth', ('ros', 'spin_period_s'): 'ros_spin_period_s', ('ros', 'cancel_topic_suffix'): 'ros_cancel_topic_suffix', ('transport', 'mock_auto_callback_after_s'): 'mock_auto_callback_after_s', ('mock', 'auto_callback_after_s'): 'mock_auto_callback_after_s', ('callback', 'channel'): 'result_channel', ('callback', 'host'): 'callback_host', ('callback', 'port'): 'callback_port', ('callback', 'public_url'): 'callback_public_url', ('callback', 'auth_token'): 'callback_auth_token', ('callback', 'ros_enabled'): 'callback_ros_enabled', ('callback', 'ros_topic'): 'callback_ros_topic', ('callback', 'ros_node_name'): 'callback_ros_node_name', ('callback', 'ros_qos_depth'): 'callback_ros_qos_depth', ('jobs', 'default_deadline_s'): 'job_default_deadline_s', ('jobs', 'tick_interval_s'): 'job_tick_interval_s', ('shutdown', 'grace_s'): 'shutdown_grace_s', ('logging', 'planner_log_path'): 'planner_log_path', ('logging', 'planner_log_stdout'): 'planner_log_stdout', ('logging', 'transport_log_path'): 'transport_log_path', ('logging', 'transport_log_stdout'): 'transport_log_stdout', ('vlm_judge', 'enabled'): 'vlm_judge_enabled', ('vlm_judge', 'ignore_executor_terminal'): 'vlm_judge_ignore_executor_terminal', ('vlm_judge', 'applies_to_tags'): 'vlm_judge_applies_to_tags', ('memory', 'data_dir'): 'memory_data_dir', ('memory', 'scene_enabled'): 'scene_memory_enabled', ('memory', 'scene_inject_top_k'): 'scene_memory_inject_top_k', ('memory', 'user_enabled'): 'user_memory_enabled', ('memory', 'user_summarize_on_finish'): 'user_memory_summarize_on_finish', ('memory', 'user_min_messages_for_summary'): 'user_memory_min_messages_for_summary', ('memory', 'user_id'): 'user_id'}

def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    """Produce a flat {field_name: value} dict from either grouped or flat YAML."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                mapped = _GROUP_ALIASES.get((k, sk))
                out[mapped or f'{k}_{sk}'] = sv
        else:
            out[k] = v
    return out

def _validate(cfg: Config) -> None:
    if cfg.planner_vlm_provider not in ('openrouter', 'mock'):
        raise ValueError(f'unsupported planner_vlm_provider: {cfg.planner_vlm_provider!r}')
    if cfg.planner_vlm_mock_mode not in ('echo', 'scripted'):
        raise ValueError(f'unsupported planner_vlm_mock_mode: {cfg.planner_vlm_mock_mode!r}')
    if cfg.result_channel not in ('webhook', 'polling', 'mq'):
        raise ValueError(f'unsupported result_channel: {cfg.result_channel!r}')
    if cfg.callback_port < 1 or cfg.callback_port > 65535:
        raise ValueError(f'callback_port out of range: {cfg.callback_port}')
    if cfg.executor_max_retries < 0:
        raise ValueError('executor_max_retries must be >= 0')
    if cfg.executor_max_concurrency < 1:
        raise ValueError('executor_max_concurrency must be >= 1')
    if cfg.mock_auto_callback_after_s < 0:
        raise ValueError('mock_auto_callback_after_s must be >= 0')
    if cfg.shutdown_grace_s < 0:
        raise ValueError('shutdown_grace_s must be >= 0')
    if cfg.planner_vlm_timeout_s <= 0:
        raise ValueError('planner_vlm_timeout_s must be > 0')
    if cfg.planner_vlm_cameras is not None and (not isinstance(cfg.planner_vlm_cameras, list)):
        raise ValueError('planner_vlm_cameras must be a list')
    if not isinstance(cfg.vlm_judge_applies_to_tags, list) or not all((isinstance(t, str) for t in cfg.vlm_judge_applies_to_tags)):
        raise ValueError('vlm_judge.applies_to_tags must be a list of strings')
    if not isinstance(cfg.recovery.record_cameras, list) or not all((isinstance(c, str) for c in cfg.recovery.record_cameras)):
        raise ValueError('recovery.record_cameras must be a list of camera-name strings')
