"""Create explicit experiment and service configurations for a new installation."""
from __future__ import annotations
import argparse
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from carryon.experiments.files import new_directory, read_json, sha256, write_json
from carryon.experiments.request_budget import digest

SUITES = ("libero_perturbation", "libero_safety", "libero_recovery", "robolab")


def create(args):
    envs, cache, output = (Path(x).expanduser().absolute() for x in (args.environments, args.cache, args.output))
    for number in (args.execution_max_attempts, args.reflection_max_attempts):
        if number < 0:raise ValueError("Request limits must be nonnegative")
    for number in (args.execution_budget_usd, args.reflection_budget_usd):
        if not Decimal(number).is_finite() or Decimal(number) < 0:raise ValueError("Budgets must be nonnegative")
    py = lambda profile: str(envs / profile / "venv/bin/python")
    model_root = Path(args.model_cache).expanduser().absolute() if args.model_cache else cache / "models"
    model = lambda name: str((cache / "models" / name).resolve() if name == "openpi_tokenizer" else model_root / name)
    catalog = read_json(ROOT / "requirements/models.json")
    environment_versions = read_json(ROOT / "requirements/environments.json")
    model_spec = lambda name: catalog[name]
    source = lambda profile, name: str(envs / profile / "sources" / name)
    models = {}
    def require(name):
        models[name] = {"model_spec":model_spec(name)}
        if args.model_cache:
            models[name]["path"] = str(model_root / name)
            if name == "openpi_tokenizer":models[name]["copy_to_cache"] = True
        return model(name)
    common = {"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "WANDB_MODE":"disabled",
              "NO_ALBUMENTATIONS_UPDATE":"1", "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}
    sim_env = {"CUDA_VISIBLE_DEVICES":args.perception_device, "MUJOCO_EGL_DEVICE_ID":args.perception_device,
               "MUJOCO_GL":"egl", "PYOPENGL_PLATFORM":"egl"}
    policy_env = dict(common, CUDA_VISIBLE_DEVICES=args.policy_device, XLA_PYTHON_CLIENT_MEM_FRACTION=".18")
    config = {"schema":"carryon-experiment/v1", "resource_cache":str(cache), "resources":models, "suites":{}}
    support = {}
    def planner():
        weights = require("qwen3_vl_32b")
        return {"argv":[py("planner"),"-m","vllm.entrypoints.openai.api_server","--model",weights,
                "--served-model-name","qwen/qwen3-vl-32b-instruct","--host","127.0.0.1","--port",str(args.planner_port),
                "--dtype","bfloat16","--tensor-parallel-size","1","--max-model-len","65536",
                "--gpu-memory-utilization","0.70","--max-num-seqs","1","--generation-config","vllm",
                "--no-enable-prefix-caching","--enforce-eager","--limit-mm-per-prompt",'{"image":4,"video":0}',
                "--mm-processor-kwargs",'{"max_pixels":1048576}'], "port":args.planner_port,
                "model_id":"qwen/qwen3-vl-32b-instruct", "startup_timeout_s":900,
                "environment":dict(common, CUDA_VISIBLE_DEVICES=args.planner_device, VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")}
    if args.suite == "robolab":
        settings={"python":py("planner"), "service":planner(), "lane":1000,
                  "local_api_base_url":"http://127.0.0.1:"+str(args.planner_port)+"/v1"}
    elif args.suite == "libero_safety":
        models["safety_assets"]={"url":"https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets/resolve/94ccca773d1bab760d1aecec4f62c8d4b8797d40/assets.zip"}
        method={"SAFE_ROLES_MODE":"live", "SAFE_SAM3_MODEL":require("sam3"), "SAFE_SAM3_PYTHON":py("sam3"),
                "SAFE_SAM3_GPU":args.perception_device, "SAFE_MOLMO_MODEL":require("molmo2"),
                "SAFE_MOLMO_PYTHON":py("molmo"), "SAFE_MOLMO_GPU":args.perception_device,
                "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"}
        policies={}
        for name in args.policies:
            if name in ("pi05", "pi05_sft"):
                checkpoint=require(name);require("openpi_tokenizer")
                identity={"model_id":model_spec(name).get("repo_id",model_spec(name).get("source")), "model_revision":model_spec(name).get("revision", "pi05_libero")}
                port=18472 if name=="pi05" else 18473
                service={"argv":[py("openpi"),"-u","-B",str(ROOT/"experiments/libero_safety/runtime_sources/services/serve_pi05.py"),
                         "--checkpoint",checkpoint,"--model-id",identity["model_id"],"--model-revision",identity["model_revision"],"--receipt","POLICY_IDENTITY.json","--port",str(port)],
                         "port":port,"startup_timeout_s":600,"environment":dict(policy_env,OPENPI_DATA_HOME=model("openpi_tokenizer"))}
                policies[name]={"python":py("safety-client"),"port":port,"environment":policy_env,"service":service,
                    "metadata":{"protocol":"paired_rng_persistent_v2",**identity}}
            elif name=="openvla_oft":
                policies[name]={"python":py("oft"),"checkpoint":require(name),"upstream":source("oft","openvla-oft"),"environment":policy_env}
            else:
                weights=require("vlact");base=require("vlact_base");expert=require("vlact_expert")

                service={"argv":[py("starvla"),"-u","-B",str(ROOT/"experiments/libero_safety/vlact_adapter/serve_policy.py"),
                    "--upstream",source("starvla","vlact"),"--checkpoint-root",weights,"--base-vlm",base,"--expert",expert,
                    "--output","deployment","--port","18474"],"port":18474,"startup_timeout_s":600,"environment":policy_env}
                policies[name]={"python":py("safety-client"),"port":18474,"upstream":source("starvla","vlact"),
                                "checkpoint_revision":model_spec("vlact")["revision"],"environment":policy_env,"service":service}
        path=ROOT/"experiments/libero_safety/runtime_sources/shield/carryon/planner/instruction_roles.py"
        spec=importlib.util.spec_from_file_location("carryon_role_configuration",path);roles=importlib.util.module_from_spec(spec);spec.loader.exec_module(roles)
        instructions=read_json(path.parent/"data/instructions.json")["instructions"]
        budget={"endpoint":"https://openrouter.ai/api/v1/chat/completions", "max_attempts":args.execution_max_attempts,
                "budget_usd":args.execution_budget_usd,"per_request_ceiling_usd":"0.01", "response_provider_aliases":["Venice"],
                "envelope":{"model":roles.ROLE_PARSE_MODEL,"provider":roles.ROLE_PARSE_PROVIDER,"temperature":0.0,"max_tokens":200},
                "system_prompt_sha256":digest(roles.ROLE_PARSE_SYS),
                "allowed_user_message_sha256":sorted({digest("INSTRUCTION: "+instruction) for instruction in instructions})}
        settings={"sim_python":py("simulation"),"sim_environment":sim_env,"method_environment":method,"policies":policies,"role_budget":budget}
    else:
        models["libero_assets"]={"url":"https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/8f1084e3132a39270c3a13ebe37270a43ece2a01"}
        if args.suite=="libero_recovery":
            models["recorded_scenes"]={"url":"https://www.modelscope.cn/api/v1/datasets/ataier/LIBERO_Recovery_Assets/repo?Source=SDK&Revision=4183a3978e848fd8caab56474abfd4f8eb1ee42e&FilePath=scenes.tar.gz"}
            models["recorded_protocol"]={"url":"https://raw.githubusercontent.com/liulin815/LIBERO-Recovery/4027d2bb4bef3902575d53fad1cfa399bbb580bb/examples/LIBERO/eval_files/eval_libero_custom_scene.py"}
        qwen=require("qwen_oft");base=require("qwen3_vl_4b");qwen_files=model_spec("qwen_oft")["files"]
        def binding(name):
            row=next(r for r in qwen_files if r["path"]==name)
            return {"path":str(Path(qwen)/name)}
        support["policy.json"]={"source":{"root":source("starvla","starvla"),"revision":environment_versions["sources"]["starvla"]["revision"]},
            "base_vlm":{"root":base,"files":[r["path"] for r in model_spec("qwen3_vl_4b")["files"]]},
            "checkpoint":binding("checkpoints/steps_50000_pytorch_model.pt"),"model_config":binding("config.yaml"),
            "statistics":binding("dataset_statistics.json"),"device":"cuda:0","use_bf16":True,"port":18360,"idle_timeout":1800}
        policy_service={"argv":[py("starvla"),"-u","-B","-m","carryon.experiments.recovery.policy","--config",str(output/"policy.json"),"--output","deployment"],
            "port":18360,"startup_timeout_s":600,"environment":dict(policy_env,PYTHONPATH=str(ROOT/"src"))}
        provider={"only":["deepinfra/fp8"],"allow_fallbacks":False,"data_collection":"deny","require_parameters":True}
        remote={"endpoint":"https://openrouter.ai/api/v1/chat/completions","api_key_env":"OPENROUTER_API_KEY","provider":provider,
                "response_provider_aliases":["DeepInfra"],"per_request_ceiling_usd":"0.10","timeout_s":180}
        local={"transport":"local","endpoint":"http://127.0.0.1:"+str(args.planner_port)+"/v1/chat/completions","per_request_ceiling_usd":0,"timeout_s":180}
        routes={"planner":local,"profile_summary":local,"reground_selector":remote};limits={"planner":1,"profile_summary":1,"reground_selector":1}
        if args.suite=="libero_recovery":routes["pose_classifier"]=remote;limits["pose_classifier"]=1
        else:routes["judge"]=local;limits["judge"]=None
        episode={"suite":args.suite,"sim_python":py("simulation"),"runtime_python":py("simulation"),"sim_environment":sim_env,
                 "runtime_environment":dict(common,CUDA_VISIBLE_DEVICES=args.perception_device,CUDA_HOME=args.cuda_home),
                 "policy":{"host":"127.0.0.1","port":18360},
                 "ports":{"control":19031,"camera":19032,"rep":19033,"pull":19034,"pub":19035,"callback":19036,"graspgen":19037},
                 "resources":{"sam3_model":require("sam3"),"sam3_python":py("sam3"),"sam3_gpu":args.perception_device,
                              "molmo_model":require("molmo2"),"molmo_python":py("molmo"),"molmo_gpu":args.perception_device,
                              "molmo_overlay":"","curobo_python":py("curobo"),"curobo_gpu":args.perception_device},
                 "model_transport":{"max_requests":10000000,"max_paid_requests":args.execution_max_attempts,"budget_usd":args.execution_budget_usd,
                                    "routes":routes,"per_episode_limits":limits}}
        reflection=dict(remote,model="qwen/qwen3-vl-235b-a22b-instruct",max_requests=args.reflection_max_attempts,
                        budget_usd=args.reflection_budget_usd,per_request_ceiling_usd="0.15")
        settings={"episode":episode,"reflection":reflection,"services":[planner(),policy_service]}
    config["suites"][args.suite]=settings
    with new_directory(output,inputs=[ROOT,cache,envs]) as staging:
        for name,doc in support.items():write_json(staging/name,doc)
        write_json(staging/"experiment.json",config)
    return output/"experiment.json"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",required=True,choices=SUITES)
    p.add_argument("--environments",required=True);p.add_argument("--cache",required=True);p.add_argument("--output",required=True)
    p.add_argument("--policy-device",default="0");p.add_argument("--perception-device",default="1");p.add_argument("--planner-device",default="2")
    p.add_argument("--model-cache",help="Existing model directories named after requirements/models.json entries; otherwise download into the resource cache")
    p.add_argument("--planner-port",type=int,default=18134);p.add_argument("--cuda-home",default="/usr/local/cuda")
    p.add_argument("--policies",nargs="+",choices=("pi05","pi05_sft","openvla_oft","vlact"),default=["pi05","pi05_sft","openvla_oft","vlact"])
    p.add_argument("--execution-max-attempts",type=int,default=0);p.add_argument("--execution-budget-usd",default="0")
    p.add_argument("--reflection-max-attempts",type=int,default=0);p.add_argument("--reflection-budget-usd",default="0")
    print(create(p.parse_args()))


if __name__=="__main__":main()
