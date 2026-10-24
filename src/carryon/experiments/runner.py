"""Run independently configured CarryOn experiments and retain every result."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from .files import read_json, write_json, sha256, bound_file
from .resources import read_config
from .services import local_service

IMPLEMENTED_SUITES = ("libero_safety","robolab","libero_perturbation","libero_recovery")

def absolute(value, base):
    return os.path.abspath((Path(base)/value).expanduser())

def child_environment(release):
    env = dict(os.environ)
    env.pop("PYTHONHOME",None)
    env.update(PYTHONPATH=str(release/"src"),PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
    return env

def run_command(argv, cwd, env, log):
    with Path(log).open("xb") as stream:
        result = subprocess.run(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                                stdout=stream,stderr=subprocess.STDOUT)
    return result.returncode

def check_interpreter(path):
    if not Path(path).is_file():
        raise ValueError("Missing Python executable: "+str(path))

def probe_environment(python, modules, release):
    check_interpreter(python)
    code = """import importlib.util,json,site,sys
from pathlib import Path
modules=json.loads(sys.argv[1]); origins={}; missing=[]; outside=[]
for name in modules:
 spec=importlib.util.find_spec(name)
 if spec is None: missing.append(name); continue
 origin=spec.origin
 origins[name]=origin
 if origin and origin not in ('built-in','frozen') and not Path(origin).resolve().is_relative_to(Path(sys.prefix).resolve()):
  outside.append(name+': '+origin)
print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,user_site=site.ENABLE_USER_SITE,origins=origins,missing=missing,outside_environment=outside)))
"""
    env=child_environment(release)
    env.pop("PYTHONPATH",None)
    result=subprocess.run([python,"-B","-c",code,json.dumps(modules)],env=env,
                          capture_output=True,text=True,timeout=60)
    if result.returncode:
        raise ValueError("Environment inspection failed for "+python+": "+result.stderr[-2000:])
    receipt=json.loads(result.stdout)
    executable=Path(os.path.abspath(python))
    if (executable.parent.parent/"pyvenv.cfg").is_file() and Path(receipt["prefix"]).resolve()!=executable.parent.parent.resolve():
        raise ValueError("Configured virtual environment was bypassed: "+python)
    if receipt["missing"] or receipt["outside_environment"] or receipt["user_site"]:
        raise ValueError("Invalid isolated environment: "+json.dumps(receipt))
    return receipt


def probe_safety_kinematics(python, release):
    env = child_environment(release)
    env.pop("PYTHONPATH", None)
    env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    result = subprocess.run([python, "-B", str(release / "src/carryon/experiments/safety/kinematics.py")],
                            env=env, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise ValueError("Shield kinematics initialization failed for " + python + ": " + result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def robolab(config, base, release, output, preset):
    python = absolute(config["python"],base)
    probe = probe_environment(python,["httpx","aiohttp","yaml","PIL","jsonschema","requests","zmq"],release)
    write_json(output/"ENVIRONMENT.json",probe)
    # Build and verify the exact request inventory before starting a model.
    for suite in ("main","text-stale","image-stale"):
        rc=run_command([python,"-B",str(release/"experiments/robolab/run_evaluation.py"),
            "--suite",suite,"--preset",preset,"--lane",str(config.get("lane",1000)),
            "--manifest-only","--output",str(output/(suite+"-plan"))],output,
            child_environment(release),output/(suite+"-plan.log"))
        if rc:
            raise ValueError("RoboLab protocol preflight failed: "+suite+"; see plan log")
    service = config.get("service")
    if service:
        model_probe=probe_environment(service["argv"][0],["vllm","torch","transformers","numpy"],release)
        write_json(output/"MODEL_ENVIRONMENT.json",model_probe)
        if service.get("model_id")!="qwen/qwen3-vl-32b-instruct":
            raise ValueError("RoboLab requires the configured Qwen3-VL-32B-Instruct model identity")
    context = local_service(service,output/"model_service") if service else nullcontext()
    runs = []
    with context:
        for index,suite in enumerate(("main","text-stale","image-stale")):
            destination = output/suite
            argv = [python,"-B",str(release/"experiments/robolab/run_evaluation.py"),
                    "--suite",suite,"--preset",preset,"--lane",str(config.get("lane",1000)),
                    "--output",str(destination)]
            if config.get("local_api_base_url"):
                argv += ["--local-api-base-url",config["local_api_base_url"]]
            rc = run_command(argv,output,child_environment(release),output/(suite+".log"))
            receipt = read_json(destination/"EVALUATION.json") if (destination/"EVALUATION.json").exists() else None
            runs.append({"suite":suite,"returncode":rc,"evaluation":receipt})
            print(suite,"completed" if rc==0 else "error",flush=True)
    return {"status":"completed" if all(r["returncode"]==0 for r in runs) else "incomplete",
            "runs":runs,"preset":preset}

def safety(config, base, release, output, preset, cache):
    from .safety.materialize import runtime_files
    if config.get("method_environment",{}).get("SAFE_ROLES_MODE","live") != "live":
        raise ValueError("Experiment evaluation requires online instruction-role inference; fixed responses are diagnostic only")
    runtime = cache/"safety"
    runtime_files(runtime)
    benchmark = cache/"safety_benchmark"
    bound = read_json(benchmark/"BENCHMARK_MANIFEST.json")
    for path,digest in bound["files"].items():
        bound_file({"path":path,"sha256":digest},benchmark)
    cases = read_json(release/"experiments/libero_safety/cases.json")
    rows = cases["rows"] if preset=="paper" else [r for r in cases["rows"] if r["id"] in cases["smoke_ids"]]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate Safety cases")
    policies = config["policies"]
    if not policies or set(policies)-{"pi05","pi05_sft","openvla_oft","vlact"}:
        raise ValueError("Unknown or empty Safety policy selection")
    requirements={"pi05":["numpy","scipy","requests","yaml","cv2","openpi_client"],
        "pi05_sft":["numpy","scipy","requests","yaml","cv2","openpi_client"],
        "openvla_oft":["numpy","scipy","requests","yaml","cv2","torch","transformers","flash_attn","robosuite","bddl"],
        "vlact":["numpy","scipy","requests","yaml","cv2","websockets","msgpack"]}
    probes={name:probe_environment(absolute(policy["python"],base),list(dict.fromkeys(requirements[name]+["mujoco","robosuite"])),release)
            for name,policy in policies.items()}
    probes["simulator"]=probe_environment(absolute(config["sim_python"],base),
        ["numpy","torch","mujoco","robosuite","bddl","cv2","yaml"],release)
    for name, probe in probes.items():
        probe["kinematics"] = probe_safety_kinematics(probe["python"], release)
    write_json(output/"ENVIRONMENT.json",probes)
    write_json(output/"PLAN.json",{"preset":preset,"policies":list(policies),"rows":rows,
               "control_steps":900,"settle_steps":10,"model_answers_replayed":False})
    method_environment = dict(config["method_environment"])
    if config.get("role_budget"):
        role_budget = dict(config["role_budget"], budget_directory=str(output/"role_budget"))
        budget_path = output/"ROLE_BUDGET.json"
        write_json(budget_path, role_budget)
        method_environment["CARRYON_ROLE_REQUEST_BUDGET"] = str(budget_path)
    results = []
    def record(result):
        results.append(result)
        with (output/"episodes.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(result,allow_nan=False)+"\n")
        write_json(output/("PROGRESS_%03d.json"%len(results)),{"recorded":len(results),"expected":len(rows)*len(policies),"latest":result})
    for name,policy in policies.items():
        policy_output = output/name
        policy_output.mkdir()
        context = local_service(policy["service"],policy_output/"model_service") if policy.get("service") else nullcontext()
        completed_ids=set();attempted_ids=set()
        try:
            with context:
                for row in rows:
                    episode = policy_output/row["id"]
                    spec = dict(config.get("episode",{}),runtime=str(runtime),benchmark=str(benchmark),
                        benchmark_config=str(benchmark/"config"),sim_python=absolute(config["sim_python"],base),
                        sim_environment=config["sim_environment"],method_environment=method_environment,
                        policy={**policy,"kind":name},row=row,control_steps=900)
                    path = policy_output/(row["id"]+".config.json")
                    write_json(path,spec)
                    env = child_environment(release)
                    env.update({k:str(v) for k,v in policy.get("environment",{}).items()})
                    attempted_ids.add(row["id"])
                    rc = run_command([absolute(policy["python"],base),"-B","-m","carryon.experiments.safety.episode",
                                      "--config",str(path),"--output",str(episode)],output,env,
                                      policy_output/(row["id"]+".log"))
                    if (episode/"RESULT.json").exists():
                        result = read_json(episode/"RESULT.json")
                    else:
                        result = {"status":"service_error","row":row,"policy":name,"returncode":rc}
                        if (episode/"FAILURE.json").exists():
                            result["failure"] = read_json(episode/"FAILURE.json")
                    record(result)
                    completed_ids.add(row["id"])
                    print(name,row["id"],result["status"],flush=True)
        except Exception as exc:
            failure={"error":type(exc).__name__,"message":str(exc)}
            write_json(policy_output/"SERVICE_FAILURE.json",failure)
            for row in rows:
                if row["id"] not in completed_ids:
                    record({"status":"service_error","trajectory_complete":False,"row":row,"policy":name,
                        "execution_attempted":row["id"] in attempted_ids,"failure":failure})
    return {"status":"completed" if len(results)==len(rows)*len(policies) and all(
            r["status"]=="completed" for r in results) else "incomplete",
            "expected_episodes":len(rows)*len(policies),"recorded_episodes":len(results),
            "preset":preset,"results":results}

def main(argv=None, *, release_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",required=True,choices=IMPLEMENTED_SUITES)
    parser.add_argument("--config",required=True)
    parser.add_argument("--preset",choices=("paper","smoke"),default="smoke")
    parser.add_argument("--output",required=True)
    args = parser.parse_args(argv)
    config,base = read_config(args.config)
    if not isinstance(config["suites"],dict) or args.suite not in config["suites"]:
        parser.error("The configuration must contain settings for the selected suite")
    release,output = Path(release_root).resolve(),Path(args.output).resolve()
    cache = Path(absolute(config["resource_cache"],base))
    prepared = read_json(cache/"RESOURCES.json")
    if prepared.get("schema")!="carryon-prepared-resources/v1" or prepared["config"]!=config:
        parser.error("Resource cache belongs to a different configuration; run prepare_resources first")
    if output.exists() or output.is_relative_to(release) or output.is_relative_to(cache):
        parser.error("Use a new output directory outside the release and resource cache")
    output.mkdir(parents=True)
    write_json(output/"RUN_CONFIG.json",{"suite":args.suite,"preset":args.preset,
               "config":str(Path(args.config).resolve()),"resources":str(cache/"RESOURCES.json")})
    try:
        if args.suite=="robolab":
            result = robolab(config["suites"][args.suite],base,release,output,args.preset)
        elif args.suite=="libero_safety":
            result = safety(config["suites"][args.suite],base,release,output,args.preset,cache)
        else:
            from .recovery.backend import run as recovery
            result=recovery(config["suites"][args.suite],base,release,output,args.preset,cache,args.suite)
    except BaseException as exc:
        write_json(output/"INCOMPLETE.json",{"status":"incomplete","error":type(exc).__name__,"message":str(exc)})
        raise
    write_json(output/"RESULT.json",result)
    if result["status"]!="completed":
        raise SystemExit(1)
