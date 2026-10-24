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


# The single environment variable we honor for picking the config file path:
# lets the user override the config location without passing a CLI flag.
# All other fields live in the YAML, optionally with ``${VAR}`` placeholders
# resolved by :func:`expand_env_vars`.
CONFIG_ENV_VAR = "CARRYON_CONFIG"
DEFAULT_CONFIG_PATH = "./config.yaml"


# ``${VAR}`` or ``${VAR:-default}``. Default may be empty (``${VAR:-}``) and
# may itself contain any character except a closing brace.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


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
    if "${" not in s:
        return s
    missing: list[str] = []

    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        default = m.group(2)
        val = os.environ.get(name)
        if val is not None and val != "":
            return val
        if default is not None:
            return default
        missing.append(name)
        return ""

    out = _ENV_VAR_RE.sub(repl, s)
    if missing:
        unique = sorted(set(missing))
        raise KeyError(
            "missing required environment variable(s) referenced in config: "
            + ", ".join(unique)
            + ". Export them before starting CarryOn, or use "
            "${VAR:-default} to provide a fallback."
        )
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

    # When enabled, a vlm_judge `failed` terminal triggers physical recovery
    # (takeover -> regrasp/rewind -> handback) before the default release.
    # Default OFF: requires the gello control plane + a motion planner backend.
    enabled: bool = False
    gello_base_url: str = "http://127.0.0.1:18080"
    gello_camera_url: str = ""
    policy_name: str = "recover_hold_lost"
    object_hint: str = ""
    # Max recovery invocations per subtask before the closed loop gives up and
    # lets the plan finalize as failed. The recovery policy itself is single-shot;
    # this caps how many times a *re-failed* (post-handback) job may re-trigger
    # recovery. 1 = one attempt then fail (matches single-shot testing).
    max_attempts: int = 1
    # LLM-authored declarative recovery policies to load at startup (one path or a
    # list). Each is structurally gated then registered; since the coordinator runs
    # the *latest* version, the highest-versioned authored policy becomes the live
    # default. None loads only the built-in policies.
    policy_json: Any = None
    # ZMQ bridge ports for state and commands.
    bridge_host: str = "127.0.0.1"
    bridge_rep_port: int = 6003
    bridge_pull_port: int = 6004
    bridge_pub_port: int = 6005
    # Attach camera calibration to recovery records for world-coordinate reconstruction.
    camera_spec_path: str = "tools/calibration/camera_spec.json"
    # Override camera name when the spec file omits/renames it.
    camera_name: str = ""
    # Extra camera names to RECORD frames from during recovery, on top of the
    # calibrated camera(s) above. Recorded for reflection evidence only (e.g. an
    # eye-in-hand "wrist_rgb" view) -- NOT added to the episode's calibrated
    # CameraSpec list, so they need no hand-eye extrinsics. Empty => record only
    # the calibrated cam(s).
    record_cameras: list[str] = field(default_factory=list)
    # Optional geometry backends; unavailable motion or perception fails closed.
    use_curobo: bool = False              # build CuroboMotionPlanner
    curobo_robot_config: str = "franka.yml"
    curobo_world_config: str = ""
    # RNG seed for cuRobo IK/trajopt seed sampling; fixed for reproducible plans
    # (identical perception -> identical reachability/trajectory run-to-run).
    curobo_seed: int = 123
    use_graspnet: bool = False            # build ContactGraspNetDetector
    graspnet_ckpt: str = ""
    # Paths to the configured perception environments, repositories and weights.
    sam3_model_path: str = ""
    sam3_python_exe: str = ""
    # SAM3 inference device: "auto" (cuda if available), "cuda", or "cpu".
    sam3_device: str = "auto"
    # Physical GPU id(s) for the SAM3 *subprocess only* (CUDA_VISIBLE_DEVICES).
    # cuRobo stays on the main process GPU; set this to a free card (e.g. "1").
    sam3_cuda_visible_devices: str = ""
    # SAM3 backend: "native" (sam3 package, promptable point -> full object,
    # cap-x aligned) or "transformers" (HF Sam3Model, point via small box).
    sam3_backend: str = "native"
    # Molmo2 open-vocabulary pointing (live grounding point source). When enabled,
    # SAM3 text grounding failures fall back to Molmo surface points before the
    # sim's low centre-projection anchor drives a table-height point-prompt.
    enable_molmo: bool = False
    # Pre-grasp VLM identity check of the grounded mask (verify_object_identity
    # op). Enabled by default.
    verify_identity: bool = True
    # YAML mapping object-hint -> open-vocab segmentation phrases (generated by
    # the object catalog). Empty = no catalog.
    object_catalog_path: str = ""
    molmo_model_path: str = ""
    molmo_python_exe: str = ""
    molmo_pythonpath_prepend: str = ""
    molmo_cuda_visible_devices: str = ""
    # SAM3 text-grounding detection / mask thresholds. <0 => keep geometry.py
    # Sam3Config defaults (0.5/0.5). Lower the detection threshold (e.g. 0.25)
    # to recover weak groundings on awkward names ("alphabet soup"), at the cost
    # of more spurious instances (the spatial disambiguator then prunes them).
    sam3_threshold: float = -1.0
    # Adaptive-cascade retry: relaxed text-grounding threshold used ONLY when the
    # strict ``sam3_threshold`` pass returns no usable mask. < 0 disables the retry
    # (single-pass). Recovers weak-but-correct detections (small LIBERO objects
    # score ~0.30 and dip under the primary gate frame-to-frame) without lowering
    # everyday precision. The SAM3 server loads at min(threshold, retry_threshold).
    sam3_retry_threshold: float = -1.0
    sam3_mask_threshold: float = -1.0
    graspnet_repo_dir: str = ""
    graspnet_python_exe: str = ""
    # Hybrid grasp source (cap-x-style): CGN primary + deterministic mask/depth
    # top-down fallback for objects CGN can't place contacts on. When False,
    # perceive_grasp is CGN-only.
    grasp_fallback_topdown: bool = True
    # Select the primary grasp detector: Contact-GraspNet subprocess or GraspGen server.
    grasp_primary: str = "cgn"
    graspgen_host: str = "127.0.0.1"
    graspgen_port: int = 5557
    # Camera used for calibrated RGB-D recovery perception.
    grasp_camera: str = "overhead_rgb"
    # TCP offset from the hand frame to the fingertip contact centre.
    tcp_offset_m: float = 0.21
    close_width_m: float = 0.012          # gripper close target (~object size)
    # assert_holding lower-bound mode: "command_relative" (real rig: a miss
    # stalls at the close command) or "empty_relative" (binarising rig, e.g.
    # the LIBERO sim shim: a miss fully closes to ~empty width, and the
    # command-relative bound wrongly rejects held THIN objects).
    hold_check: str = "command_relative"
    # Live strategy-library routing: on each recovery, select a gated seed
    # strategy by failure bucket (diagnosis x perception-inferred geometry, no
    # object-name priors) instead of always running the pinned default policy.
    # Fails open to the default policy on any inference/selection error.
    use_strategy_selector: bool = False
    # Online strategy reflection. When a job re-triggers recovery after a
    # FAILED attempt, ask the reflection VLM (reflection_vlm.* config, falls
    # back to planner_vlm.*) for a bounded StrategySpec proposal before
    # routing. Fails open to the deterministic library on any error. Also
    # enabled by env CARRYON_REFLECTOR=1 for eval sweeps.
    use_reflector: bool = False
    # Escape a collision-locked freeze pose with a short unplanned nudge toward
    # the rewind target before goto_init/rewind planning (see BackendConfig).
    rewind_escape_nudge_rad: float = 0.0
    # Default ADDITIONAL world-+Z nudge on the recovery regrasp's perceived fingertip
    # (per-object override via goto_grasp.z_offset). 0 = trust perception; negative
    # descends deeper (tall objects). Distinct from offset_world_z_m, which lives in
    # the perception/detector layer; this one is the policy-tunable descent knob.
    grasp_z_offset_m: float = 0.0
    # Default ADDITIONAL world-frame (x, y) nudge on the recovery regrasp's perceived
    # fingertip (per-object override via goto_grasp.x_offset / y_offset). 0 = trust
    # perception. Lateral bias correction only; does not touch grasp orientation.
    grasp_x_offset_m: float = 0.0
    grasp_y_offset_m: float = 0.0
    # Default ADDITIONAL world-Z yaw nudge on the recovery regrasp orientation
    # (per-object override via goto_grasp.yaw_offset_deg). 0 = trust perception.
    grasp_yaw_offset_deg: float = 0.0
    exec_hz: float = 40.0                 # motion stream rate
    exec_max_step_rad: float = 0.01       # re-interp cap (smooth tracking)
    # Joint-tolerance gate for the precision grasp move (goto_grasp / goto_pose).
    # 0 = keep the backend default (3 deg). The sim JOINT_POSITION PD carries a
    # steady-state residual on a mid-arm/wrist joint (~0.055 rad observed on the
    # milk regrasp approach) that the tight 3 deg gate marginally fails even after
    # the full settle window -- a 3.15 deg approach residual moves the fingertip
    # only ~mm at the ~0.1 m grasp lever, so a slightly looser gate (~4 deg) lets
    # the grasp complete without hurting accuracy. Real-robot PD settles tighter,
    # so this is a sim-side knob.
    grasp_verify_joint_tol_rad: float = 0.0
    # mask/depth top-down fallback grasp tuning (world-frame nudges + yaw mount cal)
    offset_world_x_m: float = 0.0
    offset_world_y_m: float = -0.01
    offset_world_z_m: float = -0.032      # LOWERS fingertip onto the object
    local_table_clearance_m: float = 0.0  # off (backend is TCP-aware)
    # Top-band centroid correction for angled-overhead cameras (e.g. LIBERO
    # agentview): compute the grasp XY from only the object's top-face points so a
    # tall object's visible front face does not bias the grasp off-footprint (which
    # tips the object on close). 0 = legacy (all mask points). ~0.03 suits
    # libero_object cartons. See MaskDepthGraspConfig.center_top_band_m.
    grasp_center_top_band_m: float = 0.0
    # Property-driven grasp descent: descend a fraction of the object's perceived
    # height below its top so a top-down pinch lands on the wide body (not the
    # tapered top). Generalises the per-object descent knob across object heights;
    # bounded and floored by the table. False = legacy median-z. See
    # MaskDepthGraspConfig.grasp_below_top_*.
    grasp_below_top_enabled: bool = False
    grasp_below_top_frac: float = 0.32
    grasp_below_top_max_m: float = 0.05
    # Spatial disambiguation: pick the instance nearest the VLA's pre-failure EE
    # XY instead of SAM3's argmax mask. ``grasp_disambig_prompt`` is the
    # enumeration prompt -- a GENERIC category noun (e.g. "can") makes SAM3 list
    # every candidate so the prior can choose; empty falls back to the object
    # hint (which often collapses to a single, possibly wrong, instance).
    grasp_spatial_disambig: bool = True
    grasp_disambig_prompt: str = ""
    grasp_disambig_max_dist_m: float = 0.20
    # Consistency gate for the Molmo point source: reject a Molmo point farther
    # than this fraction of the image diagonal from the Judge anchor (target_px)
    # and fall back to the anchor point-prompt. Routes final authority to the
    # reflector/Judge anchor; guards against Molmo pointing at a look-alike
    # neighbour. <= 0 disables the gate; the default is disabled.
    grasp_molmo_anchor_gate_frac: float = 0.0
    # World-Z clearance above table_top_z below which GraspGen object-cloud points
    # are dropped (removes tabletop leaked by an over-segmented mask so grasps are
    # not pulled to table height). Applied only when table_top_z is meaningful.
    grasp_pc_table_clearance_m: float = 0.02
    open_axis_angle_deg: float = 135.0    # physical gripper yaw mounting cal
    # hybrid routing gates (cap-x-style): trust CGN only if score + height clear.
    min_primary_score: float = 0.1
    min_primary_z_above_table_m: float = 0.005
    # cuRobo tabletop preset (table top ~z, collision buffer) for on-table grasping.
    table_top_z: float = -0.02
    collision_activation_m: float = 0.01
    # Robot base pose in the WORLD frame perception/records use. cuRobo plans with
    # the base at the origin, so when the world frame is NOT the robot base the
    # Cartesian goals must be shifted before planning. Real rig: calibrated world
    # == base, so the default identity is a no-op. LIBERO sim: the Panda base sits
    # at robosuite world x=-0.6, so set robot_base_x_m=-0.6 or every recovery
    # grasp is planned 0.6 m off (lands wrong / reads as unreachable).
    robot_base_x_m: float = 0.0
    robot_base_y_m: float = 0.0
    robot_base_z_m: float = 0.0
    # Arm "initialization" joint config (rad, 7-DoF) the regrasp policy retreats to
    # before re-perceiving/grasping. Default is the franka ready pose; override per rig.
    init_q: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, -1.566, 0.0, 1.579, 0.0]
    )

    @classmethod
    def from_dict(cls, section: dict[str, Any] | None) -> "RecoverySettings":
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (section or {}).items() if k in allowed and v is not None}
        return cls(**kwargs)


@dataclass
class Config:
    # ----- Top-level planner VLM -----
    planner_vlm_provider: str = "openrouter"  # openrouter | mock
    planner_vlm_model: str = "openai/gpt-5.2"
    planner_vlm_temperature: float = 0.2
    planner_vlm_api_key: str | None = None
    planner_vlm_base_url: str = "https://openrouter.ai/api/v1"
    planner_vlm_timeout_s: float = 60.0
    planner_vlm_cameras: list[dict[str, Any]] | None = None
    # Cameras the  skill may look through. Deliberately SEPARATE
    # from planner_vlm_cameras: whatever is listed there is attached to every
    # planning round as ambient context, so listing the station cameras there
    # would show the planner every place at once and there would be no question
    # left to ask. None (the default) means check_endpoint falls back to the
    # planner's list, which is right for a single-workspace robot.
    check_endpoint_cameras: list[dict[str, Any]] | None = None
    planner_vlm_mock_mode: str = "echo"  # echo | scripted
    planner_vlm_mock_responses: list | None = None
    planner_vlm_http_referer: str | None = None
    planner_vlm_x_title: str | None = None
    # Attachment budget for the camera snapshots the planner adds to a planning
    # turn. Frames are downscaled to this long edge before base64/upload, and at
    # most this many views are attached (configured order; later ones dropped).
    # 0 disables the respective cap. Downscaling needs Pillow; without it the
    # edge cap is skipped with a one-off warning rather than failing the turn.
    planner_vlm_max_image_edge_px: int = 768
    planner_vlm_max_image_attachments: int = 4
    # Persist the exact image bytes sent to each planner reasoning call. The
    # resulting paths are also recorded in planner_response.usage.snapshot_paths.
    # Empty disables persistence.
    planner_vlm_snapshot_output_dir: str = "./logs/planner_snapshots"

    # Reflection VLM for memory updates between execution rounds.
    # Kept separate from planner_vlm so reflection can use a stronger/cheaper model
    # than the hot-path planner. Any field left None falls back to its planner_vlm
    # counterpart (so a single planner_vlm config still works unchanged).
    reflection_vlm_provider: str | None = None
    reflection_vlm_model: str | None = None
    reflection_vlm_temperature: float | None = None
    reflection_vlm_api_key: str | None = None
    reflection_vlm_base_url: str | None = None
    reflection_vlm_timeout_s: float | None = None

    # ----- memory / executor -----
    memory_max_tokens: int = 32000
    # Failure handling is a staged ladder: local Retry up to executor_max_retries
    # (backoff), then — if react_on_failure — hand to the Reactor (planner.react).
    executor_max_retries: int = 2
    executor_react_on_failure: bool = True
    executor_max_concurrency: int = 8
    executor_retry_backoff_base_s: float = 0.5
    executor_retry_backoff_cap_s: float = 5.0
    # When the VLM Judge declares a subtask done while the executor's policy
    # is still running, JobManager fires a cancel and waits up to this many
    # seconds for the executor's terminal callback (the "released" ack)
    # before forcibly advancing the DAG. Tune up if your VLA's brake +
    # return-to-home routine takes longer than the default.
    judge_release_timeout_s: float = 10.0

    # ----- endpoints / transport -----
    endpoints_config: str = "./endpoints.yaml"
    default_transport: str = "http"
    http_default_timeout: float = 10.0
    vla_auth_token: str | None = None
    # ----- ROS 2 transport (only constructed when actually needed) -----
    # If true, force-build ROSTransport even when no endpoint declares
    # transport=ros. Useful when you want to /register a ROS endpoint at
    # runtime without restarting CarryOn.
    ros_force_enable: bool = False
    ros_node_name: str = "carryon_ros_transport"
    ros_domain_id: int | None = None
    ros_qos_depth: int = 10
    ros_spin_period_s: float = 0.05
    ros_cancel_topic_suffix: str = "/cancel"
    # When using MockTransport for end-to-end manual tests: after accepting a
    # submit, auto-call the callback URL (success) after this many seconds.
    # 0 disables auto-callback. Ignored by HTTPTransport.
    mock_auto_callback_after_s: float = 0.0

    # ----- callback / result channel -----
    result_channel: str = "webhook"
    callback_host: str = "0.0.0.0"
    callback_port: int = 9000
    callback_public_url: str = "http://127.0.0.1:9000"
    callback_auth_token: str | None = None
    # When true, ROS-transport endpoints receive their callbacks over a
    # dedicated ROS topic instead of HTTP. HTTP-transport endpoints still
    # use the webhook channel as before. Auto-enabled (in main.py) when
    # any endpoint declares transport=ros.
    callback_ros_enabled: bool = False
    callback_ros_topic: str = "/carryon/callback"
    callback_ros_node_name: str = "carryon_ros_callback"
    callback_ros_qos_depth: int = 50

    # ----- jobs -----
    job_default_deadline_s: float = 300.0
    job_tick_interval_s: float = 5.0

    # Wait up to this duration for in-flight plans before cancelling on shutdown.
    # Zero cancels immediately.
    shutdown_grace_s: float = 0.0

    # ----- logging (Planner VLM I/O tracing) -----
    # Master switch for REPL / stdout chatter. When false, the terminal only
    # shows the planner's user_reply, the initial/reactor subtask listing,
    # plan-done summaries, and errors; detailed planner/transport one-liners
    # and per-subtask dispatch chatter stay in the log files.
    verbose: bool = False
    # Path to write a readable pretty-JSON log of every planner VLM call.
    # call. Each record is ``indent=2`` JSON, records are separated by a
    # ``---`` line. Empty = disable. Directory is created on demand.
    planner_log_path: str = "./logs/planner.log"
    # Also print a one-line summary per call to stdout (only when verbose).
    planner_log_stdout: bool = True
    # ----- logging (Transport I/O tracing) -----
    # Path to write a readable pretty-JSON log of every outbound
    # submit/cancel call to a RobotEndpoint AND every inbound VLA callback.
    # Empty = disable.
    transport_log_path: str = "./logs/transport.log"
    # Also print a one-line summary per call to stdout (only when verbose).
    transport_log_stdout: bool = True

    # ----- VLM judge -----
    # Optional sub-agent that observes camera inputs and decides when a VLA job
    # has reached a terminal state. When enabled, it reports success/failed/
    # cancelled through the same JobManager callback path used by VLA callbacks.
    vlm_judge_enabled: bool = False
    # The judge's own settings (vlm / judge / camera / debug / mock sub-sections)
    # are nested inline under the ``vlm_judge:`` block in config.yaml and captured
    # here verbatim, then parsed by ``VLMJudgeConfig.from_dict`` in main.py.
    vlm_judge_settings: dict[str, Any] = field(default_factory=dict)
    # When true (default), HTTP/ROS executor callbacks that carry terminal
    # status are downgraded to ``progress`` so only VLM judge closes the job.
    vlm_judge_ignore_executor_terminal: bool = True
    # Tag-based whitelist: VLM Judge (and the corresponding executor-terminal
    # downgrade) only applies to endpoints whose ``tags`` intersect this list.
    # An empty list (default) preserves the legacy behaviour where Judge
    # supervises every endpoint. Useful when only the arm needs visual
    # supervision while a mobile base (e.g. a quadruped) should keep its
    # native terminal callback semantics.
    vlm_judge_applies_to_tags: list[str] = field(default_factory=list)

    # Optional failure-recovery loop.
    recovery: RecoverySettings = field(default_factory=RecoverySettings)

    # ----- Three-tier memory (short-term / scene / user) -----
    memory_data_dir: str = ".carryon/memory"
    scene_memory_enabled: bool = True
    scene_memory_inject_top_k: int = 20
    user_memory_enabled: bool = True
    user_memory_summarize_on_finish: bool = True
    user_memory_min_messages_for_summary: int = 3
    user_id: str = "default"

    # ---------- factories ----------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
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
        if "demo" in (data or {}):
            raise ValueError("Unsupported configuration section: demo")
        flat = _flatten(data or {})
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in flat.items() if k in allowed and v is not None}
        cfg = cls(**kwargs)
        # The VLM judge's own settings are nested inline under ``vlm_judge:``.
        # Capture the judge sub-sections verbatim (they are passed to
        # VLMJudgeConfig.from_dict by main.py) instead of letting _flatten
        # silently drop them as unknown ``vlm_judge_*`` keys.
        vlm_judge_raw = (data or {}).get("vlm_judge")
        if isinstance(vlm_judge_raw, dict):
            cfg.vlm_judge_settings = {
                k: v
                for k, v in vlm_judge_raw.items()
                if k in ("vlm", "judge", "camera", "cameras", "debug", "mock")
            }
        # The optional recovery loop is grouped into its own dataclass; parse the
        # nested ``recovery:`` section directly (flat ``recovery_*`` keys are no
        # longer supported -- the example configs all use the grouped shape).
        recovery_raw = (data or {}).get("recovery")
        if isinstance(recovery_raw, dict):
            cfg.recovery = RecoverySettings.from_dict(recovery_raw)
        _validate(cfg)
        return cfg

    @classmethod
    def from_file(cls, path: str | None = None) -> "Config":
        """Load Config from a YAML file.

        Lookup precedence for ``path``:
            1. the ``path`` argument if given
            2. the ``CARRYON_CONFIG`` env var
            3. ``./config.yaml``

        Raises FileNotFoundError if none of the candidate files exists.
        """
        chosen = path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
        if not os.path.exists(chosen):
            raise FileNotFoundError(
                f"config file not found: {chosen}. "
                f"Create one from config.example.yaml or set {CONFIG_ENV_VAR}."
            )
        import yaml  # Import lazily so constructing Config does not require PyYAML.

        with open(chosen, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config file {chosen}: top-level must be a mapping")
        data = expand_env_vars(data)
        return cls.from_dict(data)


# ---------- helpers ----------

# Maps grouped YAML keys -> flat dataclass field names.
# Anything on the left not matched here is taken as-is (so flat YAML works too).
_GROUP_ALIASES: dict[tuple[str, str], str] = {
    ("planner_vlm", "provider"): "planner_vlm_provider",
    ("planner_vlm", "model"): "planner_vlm_model",
    ("planner_vlm", "temperature"): "planner_vlm_temperature",
    ("planner_vlm", "api_key"): "planner_vlm_api_key",
    ("planner_vlm", "base_url"): "planner_vlm_base_url",
    ("planner_vlm", "timeout_s"): "planner_vlm_timeout_s",
    ("planner_vlm", "mock_mode"): "planner_vlm_mock_mode",
    ("planner_vlm", "mock_responses"): "planner_vlm_mock_responses",
    ("planner_vlm", "cameras"): "planner_vlm_cameras",
    ("check_endpoint", "cameras"): "check_endpoint_cameras",
    ("planner_vlm", "http_referer"): "planner_vlm_http_referer",
    ("planner_vlm", "x_title"): "planner_vlm_x_title",
    ("planner_vlm", "max_image_edge_px"): "planner_vlm_max_image_edge_px",
    ("planner_vlm", "max_image_attachments"): "planner_vlm_max_image_attachments",
    ("planner_vlm", "snapshot_output_dir"): "planner_vlm_snapshot_output_dir",
    ("reflection_vlm", "provider"): "reflection_vlm_provider",
    ("reflection_vlm", "model"): "reflection_vlm_model",
    ("reflection_vlm", "temperature"): "reflection_vlm_temperature",
    ("reflection_vlm", "api_key"): "reflection_vlm_api_key",
    ("reflection_vlm", "base_url"): "reflection_vlm_base_url",
    ("reflection_vlm", "timeout_s"): "reflection_vlm_timeout_s",

    ("memory", "max_tokens"): "memory_max_tokens",

    ("executor", "max_retries"): "executor_max_retries",
    ("executor", "react_on_failure"): "executor_react_on_failure",
    ("executor", "max_concurrency"): "executor_max_concurrency",
    ("executor", "retry_backoff_base_s"): "executor_retry_backoff_base_s",
    ("executor", "retry_backoff_cap_s"): "executor_retry_backoff_cap_s",
    ("executor", "judge_release_timeout_s"): "judge_release_timeout_s",
    ("vlm_judge", "release_timeout_s"): "judge_release_timeout_s",

    ("endpoints", "config"): "endpoints_config",
    ("endpoints", "default_transport"): "default_transport",

    ("http", "default_timeout"): "http_default_timeout",
    ("http", "vla_auth_token"): "vla_auth_token",

    ("ros", "force_enable"): "ros_force_enable",
    ("ros", "node_name"): "ros_node_name",
    ("ros", "domain_id"): "ros_domain_id",
    ("ros", "qos_depth"): "ros_qos_depth",
    ("ros", "spin_period_s"): "ros_spin_period_s",
    ("ros", "cancel_topic_suffix"): "ros_cancel_topic_suffix",

    ("transport", "mock_auto_callback_after_s"): "mock_auto_callback_after_s",
    ("mock", "auto_callback_after_s"): "mock_auto_callback_after_s",

    ("callback", "channel"): "result_channel",
    ("callback", "host"): "callback_host",
    ("callback", "port"): "callback_port",
    ("callback", "public_url"): "callback_public_url",
    ("callback", "auth_token"): "callback_auth_token",
    ("callback", "ros_enabled"): "callback_ros_enabled",
    ("callback", "ros_topic"): "callback_ros_topic",
    ("callback", "ros_node_name"): "callback_ros_node_name",
    ("callback", "ros_qos_depth"): "callback_ros_qos_depth",

    ("jobs", "default_deadline_s"): "job_default_deadline_s",
    ("jobs", "tick_interval_s"): "job_tick_interval_s",

    ("shutdown", "grace_s"): "shutdown_grace_s",

    ("logging", "verbose"): "verbose",
    ("logging", "planner_log_path"): "planner_log_path",
    ("logging", "planner_log_stdout"): "planner_log_stdout",
    ("logging", "transport_log_path"): "transport_log_path",
    ("logging", "transport_log_stdout"): "transport_log_stdout",

    ("vlm_judge", "enabled"): "vlm_judge_enabled",
    ("vlm_judge", "ignore_executor_terminal"): "vlm_judge_ignore_executor_terminal",
    ("vlm_judge", "applies_to_tags"): "vlm_judge_applies_to_tags",

    # NOTE: the ``recovery:`` section is parsed into a nested RecoverySettings
    # dataclass in Config.from_dict, not via these flat aliases.
    ("memory", "data_dir"): "memory_data_dir",
    ("memory", "scene_enabled"): "scene_memory_enabled",
    ("memory", "scene_inject_top_k"): "scene_memory_inject_top_k",
    ("memory", "user_enabled"): "user_memory_enabled",
    ("memory", "user_summarize_on_finish"): "user_memory_summarize_on_finish",
    ("memory", "user_min_messages_for_summary"): "user_memory_min_messages_for_summary",
    ("memory", "user_id"): "user_id",
}


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    """Produce a flat {field_name: value} dict from either grouped or flat YAML."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            # grouped section: map each sub-key
            for sk, sv in v.items():
                mapped = _GROUP_ALIASES.get((k, sk))
                # fall back to "<group>_<sub>" if not in alias table
                out[mapped or f"{k}_{sk}"] = sv
        else:
            out[k] = v
    return out


def _validate(cfg: Config) -> None:
    if cfg.planner_vlm_provider not in ("openrouter", "mock"):
        raise ValueError(f"unsupported planner_vlm_provider: {cfg.planner_vlm_provider!r}")
    if cfg.planner_vlm_mock_mode not in ("echo", "scripted"):
        raise ValueError(f"unsupported planner_vlm_mock_mode: {cfg.planner_vlm_mock_mode!r}")
    if cfg.result_channel not in ("webhook", "polling", "mq"):
        raise ValueError(f"unsupported result_channel: {cfg.result_channel!r}")
    if cfg.callback_port < 1 or cfg.callback_port > 65535:
        raise ValueError(f"callback_port out of range: {cfg.callback_port}")
    if cfg.executor_max_retries < 0:
        raise ValueError("executor_max_retries must be >= 0")
    if cfg.executor_max_concurrency < 1:
        raise ValueError("executor_max_concurrency must be >= 1")
    if cfg.mock_auto_callback_after_s < 0:
        raise ValueError("mock_auto_callback_after_s must be >= 0")
    if cfg.shutdown_grace_s < 0:
        raise ValueError("shutdown_grace_s must be >= 0")
    if cfg.planner_vlm_timeout_s <= 0:
        raise ValueError("planner_vlm_timeout_s must be > 0")
    if cfg.planner_vlm_cameras is not None and not isinstance(cfg.planner_vlm_cameras, list):
        raise ValueError("planner_vlm_cameras must be a list")
    if cfg.check_endpoint_cameras is not None and not isinstance(cfg.check_endpoint_cameras, list):
        raise ValueError("check_endpoint_cameras must be a list")
    if not isinstance(cfg.vlm_judge_applies_to_tags, list) or not all(
        isinstance(t, str) for t in cfg.vlm_judge_applies_to_tags
    ):
        raise ValueError("vlm_judge.applies_to_tags must be a list of strings")
    if not isinstance(cfg.recovery.record_cameras, list) or not all(
        isinstance(c, str) for c in cfg.recovery.record_cameras
    ):
        raise ValueError("recovery.record_cameras must be a list of camera-name strings")
