"""Bind experiment parameters to local resources and one case."""
from __future__ import annotations
import copy
import re
from pathlib import Path
from ..files import read_json, write_json, sha256
from ..configuration import validate_public_config
from .materialize import materialize

TEMPLATES = {"libero_perturbation":"libero_perturbation/runtime", "libero_recovery":"libero_recovery/runtime"}
PORTS = ("control", "camera", "rep", "pull", "pub", "callback", "graspgen")


def bind(value, values):
    if isinstance(value, dict): return {k:bind(v,values) for k,v in value.items()}
    if isinstance(value, list): return [bind(v,values) for v in value]
    if not isinstance(value, str): return value
    full=re.fullmatch(r"@([a-z0-9_]+)@",value)
    if full: return values[full[1]]
    return re.sub(r"@([a-z0-9_]+)@",lambda m:str(values[m[1]]),value)


def validate_case(suite, case):
    if suite not in TEMPLATES: raise ValueError("Unknown Recovery suite")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",case["logical_key"]): raise ValueError("Invalid case key")
    if not case.get("instruction", "").strip(): raise ValueError("Task instruction is required")
    if type(case["task_id"]) is not int or not 0<=case["task_id"]<10: raise ValueError("Invalid object task")
    if type(case["init_state_index"]) is not int or not 0<=case["init_state_index"]<(10 if suite=="libero_recovery" else 50):
        raise ValueError("Invalid initial state index")
    if suite=="libero_recovery" and not case.get("scene_case"):
        raise ValueError("Recorded Recovery requires a bound recorded scene")


def prepare(release, spec, memory, output, generation):
    release,output=Path(release).resolve(),Path(output).resolve()
    validate_public_config(spec)
    suite=spec["suite"];case=copy.deepcopy(spec["case"]);validate_case(suite,case)
    if generation not in (0,1,2): raise ValueError("Invalid memory round")
    ports=spec["ports"]
    if set(ports)!=set(PORTS) or any(type(p) is not int or not 1024<=p<=65535 for p in ports.values()) or len(set(ports.values()))!=len(PORTS):
        raise ValueError("Supply seven distinct unprivileged simulator ports")
    required=("sam3_python","sam3_model","molmo_python","molmo_model","curobo_python")
    resources=copy.deepcopy(spec["resources"])
    for key in required:
        if not Path(resources[key]).is_absolute() or not Path(resources[key]).exists():
            raise ValueError("Missing absolute resource: "+key)
    row_id=case["logical_key"]+"_R"+str(generation)
    root=output/row_id;root.mkdir(parents=True,exist_ok=False)
    runtime=output/(row_id+"_runtime")
    resources.update(row_manifest=str(root/"ROW_MANIFEST.json"),molmo_overlay="")
    materialize(release/"experiments"/TEMPLATES[suite],memory,resources,runtime)
    active=read_json(runtime/"ACTIVE.json")
    recorded=suite=="libero_recovery"
    base_seed=7 if recorded else 20260730
    disturbance_seed=int(case.get("disturbance_seed",20260730 if recorded else base_seed+case["task_id"]*1000+case["init_state_index"]))
    if not recorded and disturbance_seed!=base_seed+case["task_id"]*1000+case["init_state_index"]:
        raise ValueError("Perturbation disturbance seed differs from the frozen derived seed")
    row=dict(case,row_id=row_id,episode_key=row_id,round="R"+str(generation),
        condition="RECOVERY",base_seed=base_seed,
        disturbance_seed=disturbance_seed,
        max_vla_steps=990,wall_cap_s=600,episode_wall_time_s=600,
        disturbance="none" if recorded else "hold_lost",drop_prob=0.0 if recorded else 1.0,
        drop_scatter_m=0.0,dx_m=0.0 if recorded else .12,dy_m=0.0,
        judge_control_enabled=not recorded,judge_model_calls=0 if recorded else None,
        judge_budget_semantics="disabled" if recorded else "time_bounded",
        protocol_recovery_request=recorded,protocol_trigger="BENCHMARK_START_RECOVERY" if recorded else None,
        reflection_and_writeback=False,S_mode="empty" if generation==0 else "snapshot",
        initial_memory={key:{"sha256":sha256(runtime/"memory"/(key+".jsonl"))} for key in ("PARAMETER","SEMANTIC")},
        snapshot_contract={"path":str(runtime/"contracts/COMMON_CONTRACT.json"),"sha256":active["contract_sha256"]})
    write_json(root/"ROW_MANIFEST.json",{"rows":[row]})
    values={**resources,**{k+"_port":v for k,v in ports.items()},"row":str(root),"runtime":str(runtime)}
    frozen=bind(read_json(release/"experiments"/suite/"config.json"),values)
    write_json(root/"runtime_config.yaml",frozen["runtime"])
    write_json(root/"endpoints.yaml",frozen["endpoints"])
    assembly={"resources":dict(sam3_gpu=resources["sam3_gpu"],molmo_gpu=resources["molmo_gpu"],
        curobo_gpu=resources["curobo_gpu"],curobo_python_exe=resources["curobo_python"],
        curobo_runner_path=str(runtime/"runtime/carryon/recovery/curobo_runner.py"),
        curobo_startup_timeout_s=240,curobo_request_timeout_s=120)}
    write_json(root/"ASSEMBLY.json",assembly)
    worker={"runtime":str(runtime),"root":str(root),"row_manifest_sha256":sha256(root/"ROW_MANIFEST.json"),
        "model_transport":spec["model_transport"],"runtime_config":str(root/"runtime_config.yaml")}
    write_json(root/"WORKER.json",worker)
    return row,root,runtime
