"""Run a Recovery trajectory in isolated simulator and CarryOn processes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from ..configuration import validate_public_config
from ..files import bound_file,read_json,sha256,write_json
from .assembly import prepare,validate_case
from .process import owned
from . import protocol as P

FOLDERS={"libero_perturbation":"libero_perturbation","libero_recovery":"libero_recovery"}
ANCHORS=["[recovery-grounding] armed:","[recovery-selection] armed:",
    "[recovery-runtime] ready: generic_topdown_single_approach",
    "[recovery] parameter+semantic memory reflectors ACTIVE","cuRobo subprocess READY",
    "prewarm molmo","prewarm point_sam3","registry visual memory + SoM selector ACTIVE",
    "loaded authored policy recover_hold_lost@COMMON_NATIVE_SHORT_LONG_V2-isolated",
    "[recovery] actuating hook armed"]


def verified_benchmark(spec):
    root=Path(spec["root"]).resolve()
    receipt=read_json(root/"BENCHMARK_MANIFEST.json")
    for name,digest in receipt["files"].items():bound_file({"path":name,"sha256":digest},root)
    pythonpath=(root/spec["pythonpath"]).resolve();config=(root/spec["config"]).resolve()
    if not pythonpath.is_relative_to(root) or not config.is_relative_to(root):
        raise ValueError("Benchmark code and configuration must be inside the prepared benchmark")
    if not pythonpath.is_dir() or not config.is_dir():raise ValueError("Missing prepared benchmark code/config")
    return root,pythonpath,config


def verified_policy_client(release):
    root=Path(release)/"third_party/starvla_client"
    index=read_json(root/"SOURCE_INDEX.json")
    for record in index["files"]:bound_file(record,root)
    return root.resolve()


def preflight(spec,release):
    validate_public_config(spec);validate_case(spec["suite"],spec["case"])
    if os.name!="posix":raise ValueError("Recovery trajectories require Linux")
    from ..runner import probe_environment
    probes={}
    for key,modules in [("sim_python",["numpy","torch","mujoco","robosuite","zmq","PIL","cv2","websockets","msgpack","typing_extensions","matplotlib"]),
                        ("runtime_python",["numpy","scipy","aiohttp","httpx","yaml","jsonschema","cv2","zmq"])]:
        probes[key]=probe_environment(spec[key],modules,release)
    for key,modules in [("sam3_python",["torch","sam3","numpy"]),("molmo_python",["torch","transformers","numpy"]),("curobo_python",["torch","curobo","yaml"])]:
        probes[key]=probe_environment(spec["resources"][key],modules,release)
    verified_benchmark(spec["benchmark"])
    verified_policy_client(release)
    for port in spec["ports"].values():
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:sock.bind(("127.0.0.1",port))
            except OSError as exc:raise ValueError("Required local port is occupied: "+str(port)) from exc
    route=spec["model_transport"]
    expected={"planner","reground_selector","profile_summary"}|({"pose_classifier"} if spec["suite"]=="libero_recovery" else {"judge"})
    if set(route["routes"])!=expected:raise ValueError("Model roles differ from the selected suite")
    from .model_transport import valid_route
    for value in route["routes"].values():valid_route(value)
    if not valid_route(route["routes"]["profile_summary"])[0]:
        raise ValueError("Post-task profile summary uses the local Planner model")
    limits=route["per_episode_limits"]
    if limits.get("profile_summary")!=1:
        raise ValueError("One completed-plan profile summary is allowed per episode")
    if limits.get("planner")!=1 or limits.get("reground_selector")!=1:
        raise ValueError("Frozen Recovery planner/selector caps are one")
    if spec["suite"]=="libero_recovery" and limits.get("pose_classifier")!=1:
        raise ValueError("Recorded pose classifier cap is one")
    if spec["suite"]=="libero_perturbation" and limits.get("judge") is not None:
        raise ValueError("Temporal Judge uses the frozen wall-time budget, not a call-count cutoff")
    return probes


def shim_command(spec,row,root,release):
    ports=spec["ports"];sim=release/"experiments"/FOLDERS[spec["suite"]]/"simulator"
    cmd=[spec["sim_python"],"-B",str(sim/"simulator_entrypoint.py"),
        "--task-suite-name","libero_object","--task-id",str(row["task_id"]),
        "--episode-idx",str(row["init_state_index"]),"--seed",str(row["base_seed"]),
        "--disturbance",row["disturbance"],
        "--label","natural_failure" if row["disturbance"]=="none" else "hold_lost",
        "--vla-host",spec["policy"]["host"],"--vla-port",str(spec["policy"]["port"]),
        "--horizon","990","--drop-prob",str(row["drop_prob"]),"--drop-scatter","0.0",
        "--drop-offset-x",str(row["dx_m"]),"--drop-offset-y","0.0",
        "--control-port",str(ports["control"]),"--camera-port",str(ports["camera"]),
        "--bridge-host","127.0.0.1","--rep-port",str(ports["rep"]),
        "--pull-port",str(ports["pull"]),"--pub-port",str(ports["pub"]),
        "--defer-episode-start","--percep-resolution","512",
        "--result-file",str(root/"SHIM_RESULT.json"),"--timing-file",str(root/"EPISODE_TIMING.json"),
        "--video-file",str(root/"ROLLOUT.mp4"),"--video-fps","10"]
    if spec["suite"]=="libero_recovery":
        cmd += ["--disturbance-seed",str(row["disturbance_seed"])]
    elif row["disturbance_seed"]!=row["base_seed"]+row["task_id"]*1000+row["init_state_index"]:
        raise ValueError("Perturbation disturbance seed differs from the frozen derived seed")
    return cmd,sim


def child_environment():
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(("CARRYON_","QUALIFICATION_","LIBERO_","HANDOFF_","CARRYON_MEMORY_")) or key in ("PYTHONPATH","PYTHONHOME","ANCHOR_BUDGET_ROOT"):
            env.pop(key,None)
    env.update(PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    return env


def run(spec,memory,output,release,generation=0):
    release,output=Path(release).resolve(),Path(output).resolve()
    for source in (release,Path(memory).resolve(),Path(spec["benchmark"]["root"]).resolve()):
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Episode output must not overlap source resources")
    output.mkdir(parents=True,exist_ok=False)
    root=None;result={"status":"service_error","success":None,"suite":spec["suite"],"case":spec["case"]}
    try:
        probes=preflight(spec,release);write_json(output/"ENVIRONMENT.json",probes)
        row,root,runtime=prepare(release,spec,memory,output,generation)
        command,sim=shim_command(spec,row,root,release)
        _,benchmark_code,benchmark_config=verified_benchmark(spec["benchmark"])
        env=child_environment()
        for key in {"OPENROUTER_API_KEY","HF_TOKEN","HUGGING_FACE_HUB_TOKEN",*[r["api_key_env"] for r in spec["model_transport"]["routes"].values() if r.get("api_key_env")]}:
            env.pop(key,None)
        env.update({k:str(v) for k,v in spec["sim_environment"].items()})
        env.update(PYTHONPATH=str(benchmark_code),LIBERO_CONFIG_PATH=str(benchmark_config),
            VLA_BACKEND="starvla",STARVLA_DIR=str(verified_policy_client(release)),SHIM_MAX_LIFETIME_S="600",
            CARRYON_SIM_STARTUP_LOG=str(root/"SIM_STARTUP.json"),
            PROTOCOL_RECOVERY_LEDGER=str(root/"PROTOCOL_RECOVERY_EVENTS.jsonl"),
            QUALIFICATION_HISTORY_OUTPUT=str(root/"qualification_history"),QUALIFICATION_HISTORY_SESSION=root.name,
            QUALIFICATION_REAL_WIDTH_CONTRACT=str(runtime/"bindings/REAL_WIDTH_CONTRACT.json"))
        if spec["suite"]=="libero_recovery":
            scene=spec["recorded_scene"]
            bound_file(scene["protocol"])
            env.update(CARRYON_LIBERO_RECOVERY_ASSETS=str(Path(scene["assets"]).resolve()),
                CARRYON_LIBERO_RECOVERY_PROTOCOL_SOURCE=scene["protocol"]["path"],
                LIBERO_RECOVERY_CASE=spec["case"]["scene_case"],
                LIBERO_RECOVERY_LOAD_RECEIPT=str(root/"SCENE_LOAD.json"))
        ports=spec["ports"];base="http://127.0.0.1:"+str(ports["control"])
        cameras=[{"name":"overhead_rgb","snapshot_url":"http://127.0.0.1:"+str(ports["camera"])+"/cameras/overhead_rgb/latest.jpg"}]
        expected=dict(host="127.0.0.1",rep_port=ports["rep"],pull_port=ports["pull"],pub_port=ports["pub"],ports_source="explicit configuration",host_source="loopback")
        started=time.time()
        with owned(command,cwd=root,env=env,log=root/"shim.log",receipt=root/"shim") as shim:
            P.wait_log(root/"shim.log",shim,["shim up:","VLA warm inference PASS"],300)
            proof=read_json(root/"SIM_STARTUP.json")
            if (proof["status"]!="started" or proof["pid"]!=shim.pid):
                raise RuntimeError("Actual simulator entry identity mismatch")
            bridge=P.validate_shim_bridge_announcement(expected,P.parse_shim_bridge_announcement(root/"shim.log"))
            checks=[P.probe_sim_readiness(base,phase="post_shim_anchor")]
            if checks[-1]["verdict"]!="PASS":raise RuntimeError("Simulator is not ready")
            worker_env=child_environment();worker_env.update({k:str(v) for k,v in spec.get("runtime_environment",{}).items()})
            for route in spec["model_transport"]["routes"].values():
                name=route.get("api_key_env")
                if name and name in os.environ:worker_env[name]=os.environ[name]
            argv=[spec["runtime_python"],"-B",str(Path(__file__).with_name("worker.py")),"--config",str(root/"WORKER.json")]
            with owned(argv,cwd=root,env=worker_env,log=root/"carryon.log",receipt=root/"carryon",stdin=True) as worker:
                P.wait_log(root/"carryon.log",worker,ANCHORS,420)
                boundary=P.http_json("POST",base+"/episode/start",{"services_ready_ts":time.time()})
                if boundary.get("accepted") is not True:raise RuntimeError("Episode start rejected")
                checks.append(P.probe_sim_readiness(base,phase="pre_first_submit"))
                camera=P.probe_camera_readiness(cameras,phase="pre_first_submit_camera")
                write_json(root/"SIM_READINESS.json",dict(checks=checks,camera=camera,bridge=bridge))
                if checks[-1]["verdict"]!="PASS" or camera["verdict"]!="PASS":raise RuntimeError("Pre-submit readiness failed")
                worker.stdin.write(row["instruction"]+"\n");worker.stdin.flush()
                deadline=time.monotonic()+615
                while not (root/"SHIM_RESULT.json").exists() and time.monotonic()<deadline:
                    if shim.poll() is not None or worker.poll() is not None:raise RuntimeError("Execution process exited without official result")
                    time.sleep(.5)
                if not (root/"SHIM_RESULT.json").exists():raise RuntimeError("No official result after the native wall cap")
                if worker.poll() is None:
                    worker.stdin.write("exit\n");worker.stdin.flush()
                    try:worker.wait(timeout=45)
                    except __import__("subprocess").TimeoutExpired:pass
        artifacts=[root/x for x in ("SHIM_RESULT.json","EPISODE_TIMING.json","ROLLOUT.mp4","shim.log","carryon.log","SIM_READINESS.json")]
        traces=sorted((root/"recovery_episodes").glob("*.json"));P.assert_files_stable(artifacts+traces)
        raw=read_json(root/"SHIM_RESULT.json")
        for key,value in dict(task_id=row["task_id"],episode_idx=row["init_state_index"],init_state_index=row["init_state_index"],base_seed=row["base_seed"],disturbance_seed=row["disturbance_seed"],max_vla_steps=990).items():
            if raw.get(key)!=value:raise RuntimeError("Official result identity differs: "+key)
        if type(raw.get("success"))is not bool or type(raw.get("timeout"))is not bool or not isinstance(raw.get("terminal_reason"),str):
            raise RuntimeError("Invalid official terminal tuple")
        if raw["success"] and (raw["timeout"] or raw["terminal_reason"]!="normal_success"):
            raise RuntimeError("Inconsistent success terminal tuple")
        model_errors=[]
        for f in (root/"model_requests").glob("*/RESULT.json"):
            v=read_json(f)
            if v["status"]!="completed":model_errors.append({"path":str(f.relative_to(root)),"status":v["status"]})
        if (root/"model_requests/REJECTIONS.jsonl").exists():
            model_errors.append({"path":"model_requests/REJECTIONS.jsonl","status":"request_rejected"})
        if model_errors:
            write_json(root/"MODEL_ERRORS.json",model_errors)
            raise RuntimeError("Model service or budget errors occurred; task outcome is not a valid completed evaluation")
        result=P.build_structured_result(raw,row,traces,root/"FORMAL_RUNTIME_FLAGS.json")
        result.update(status="completed",row_id=row["row_id"],logical_key=row["logical_key"],
            memory_snapshot_sha256=read_json(runtime/"ADMITTED_MEMORY.json")["source_snapshot_sha256"],
            recovery_triggered=bool(traces),startup_cost_s=float(boundary["episode_start_ts"])-started)
        write_json(root/"RESULT.json",result)
    except BaseException as exc:
        result.update(error=type(exc).__name__,message=str(exc))
        write_json(output/"FAILURE.json",result)
        if root is not None and not (root/"RESULT.json").exists():write_json(root/"RESULT.json",result)
        raise
    finally:
        write_json(output/"EPISODE.json",dict(result=result,evidence_dir=None if root is None else root.name))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","memory","output","release-root"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--generation",type=int,choices=(0,1,2),default=0)
    args=parser.parse_args(argv)
    run(read_json(args.config),args.memory,args.output,args.release_root,args.generation)

if __name__=="__main__":main()
