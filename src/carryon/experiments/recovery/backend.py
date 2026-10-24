"""Recovery execution with reflection on the current run's failures."""
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import shutil
from ..files import bound_file,read_json,sha256,write_json
from ..reflection import memory
from ..services import local_service
from . import rounds
from .episode import run as run_episode, preflight

CASE_FILES={suite:"experiments/"+suite+"/cases.json" for suite in ("libero_perturbation","libero_recovery")}


def inventory(release,suite):
    release=Path(release)
    doc=read_json(release/CASE_FILES[suite])
    if doc["suite"]!=suite:raise ValueError("Case suite mismatch")
    rows=copy.deepcopy(doc["rows"])
    if len({r["logical_key"] for r in rows})!=len(rows):raise ValueError("Duplicate case")
    for row in rows:
        if "scene_case" in row:
            path=bound_file({"path":row["scene_case"]},release)
            row["scene_case"]=str(path)
    return doc,rows


def run(config,base,release,output,preset,cache,suite):
    release,output,cache=map(Path,(release,output,cache))
    document,all_cases=inventory(release,suite);index={r["logical_key"]:r for r in all_cases}
    if preset not in ("paper","smoke"):raise ValueError("Unknown protocol")
    spec=copy.deepcopy(config["episode"]);spec["suite"]=suite
    spec["model_transport"]["budget_directory"]=str((output/"model_budget").resolve())
    # All path resources are explicit; config may use environment expansion from read_config.
    for key in ("sim_python","runtime_python"):
        spec[key]=os.path.abspath((Path(base)/spec[key]).expanduser())
    for key in ("sam3_python","sam3_model","molmo_python","molmo_model","curobo_python"):
        spec["resources"][key]=os.path.abspath((Path(base)/spec["resources"][key]).expanduser())
    if "benchmark" not in spec:
        spec["benchmark"]={"root":str((cache/"libero_benchmark").resolve()),"pythonpath":".","config":"config"}
    else:spec["benchmark"]["root"]=str((Path(base)/spec["benchmark"]["root"]).resolve())
    if suite=="libero_recovery" and "recorded_scene" not in spec:
        prepared=cache/"recorded_scenes"
        receipt=read_json(prepared/"RECORDED_MANIFEST.json")
        for path,digest in receipt["files"].items():bound_file({"path":path,"sha256":digest},prepared)
        spec["recorded_scene"]={"assets":str((prepared/"scenes").resolve()),
            "protocol":{"path":str((prepared/"protocol.py").resolve())}}

    cases=all_cases if preset=="paper" else [index[k] for k in document["smoke"]]
    write_json(output/"PLAN.json",{"suite":suite,"preset":preset,"memory":"fresh",
        "round_zero_cases":[r["logical_key"] for r in cases],
        "retry_rule":"only preceding-round completed failures", "historical_answers_consumed":False})

    def run_round(pending,parent,path,generation):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        state=memory.load(parent);rows=[]
        registry=read_json(release/"experiments"/suite/"references/MANIFEST.json")
        (path/"references").mkdir()
        for reference in registry["files"]:
            reference["sha256"] = reference.pop("reference_id", reference.get("sha256"))
            source=bound_file(reference,release/"experiments"/suite/"references")
            target=path/"references"/source.name;shutil.copyfile(source,target)
            reference["path"]=str(target.relative_to(path))
        write_json(path/"REFERENCES.json",registry)
        for ordinal,case in enumerate(pending):
            item=copy.deepcopy(spec);item["case"]=case
            destination=path/"rows"/case["logical_key"]
            try:run_episode(item,parent,destination,release,generation)
            except Exception as exc:
                if not (destination/"EPISODE.json").is_file():
                    destination.mkdir(parents=True,exist_ok=True)
                    write_json(destination/"EPISODE.json",{"result":{"status":"service_error","success":None,
                        "error":type(exc).__name__,"message":str(exc)},"evidence_dir":None})
            receipt=read_json(destination/"EPISODE.json");result=receipt["result"]
            evidence=destination/receipt["evidence_dir"] if receipt["evidence_dir"] else destination
            result_path=evidence/"RESULT.json"
            if not result_path.exists():write_json(result_path,result)
            row=dict(case,row_id=case["logical_key"]+"_R"+str(generation),status=result["status"],
                success=result.get("success"),recovery_triggered=result.get("recovery_triggered",False),
                evidence_dir=str(evidence.relative_to(path)),
                result={"path":str(result_path.relative_to(path)),"sha256":sha256(result_path)})
            if suite=="libero_perturbation" and generation==1:
                # Reflection uses the current R1 schedule, not the source case ordinal.
                row["ordinal"]=ordinal
            rows.append(row)
            write_json(path/("PROGRESS_"+str(len(rows)).zfill(4)+".json"),{"expected":len(pending),"recorded":len(rows),"cases":rows})
            print(suite,row["row_id"],row["status"],row["success"],flush=True)
        closed=all(r["status"]=="completed" for r in rows)
        write_json(path/"ROUND.json",{"status":"closed" if closed else "incomplete","round":"R"+str(generation),
            "memory_snapshot_sha256":state["snapshot_sha256"],"expected_cases":[r["logical_key"] for r in pending],
            "cases":rows,"references":{"path":"REFERENCES.json","sha256":sha256(path/"REFERENCES.json")}})
        return rows

    # Validate the simulator and recovery environments before loading any GPU service.
    preflight_case = cases[0]
    probe = preflight(dict(spec, case=preflight_case), release)
    write_json(output/"ENVIRONMENT.json", probe)
    with ExitStack() as stack:
        for i,service in enumerate(config.get("services",[])):
            stack.enter_context(local_service(service,output/("service_"+str(i))))
        return rounds.execute(cases,output/"rounds",suite=suite,
            run_round=run_round,model_config=config.get("reflection"))
