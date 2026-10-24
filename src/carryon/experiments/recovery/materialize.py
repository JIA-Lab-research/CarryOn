"""Bind the simulation runtime to the selected recovery memory."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re
import shutil
from ..configuration import validate_public_config
from ..files import bound_file, inside, json_bytes, new_directory, read_json, sha256, write_json
from ..reflection.memory import load


def materialize(template, memory_root, resources, output):
    template,output=Path(template).resolve(),Path(output).resolve()
    state=load(memory_root)
    if state["manifest"]["fixture_only"] or state["manifest"]["provenance"].get("complete") is False:
        raise ValueError("Runtime memory must be complete and not a response fixture")
    validate_public_config(resources)
    index=read_json(template/"SOURCE_INDEX.json")
    if index.get("suite") != state["manifest"]["suite"]:
        raise ValueError("Recovery runtime and memory suite do not match")
    files={r["path"]:r for r in index["files"]}
    if len(files)!=len(index["files"]):raise ValueError("Duplicate runtime source path")
    sources={}
    for relative,row in files.items():
        namespace=row.get("source", "runtime")
        if namespace not in ("runtime", "experiment") or (namespace=="experiment" and not relative.startswith("references/")):
            raise ValueError("Invalid runtime source namespace")
        sources[relative]=bound_file(row,template.parent if namespace=="experiment" else template)
    with new_directory(output,inputs=[template,memory_root]) as root:
        pending={}
        for relative in files:
            source=sources[relative]
            if source.suffix==".json":pending[relative]=read_json(source)
            else:
                target=inside(root,relative);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
        memory_values={}
        resolving=set()
        def resolve(value):
            if isinstance(value,dict):return {k:resolve(v) for k,v in value.items()}
            if isinstance(value,list):return [resolve(v) for v in value]
            if not isinstance(value,str):return value
            match=re.fullmatch(r"\$\{(file|sha256|memory):([^}]+)\}",value)
            if not match:return value
            kind,name=match.groups()
            if kind=="memory":return memory_values[name]
            if name not in files:raise ValueError("Unbound runtime resource: "+name)
            if kind=="file":return str(inside(output,name))
            ensure(name);return sha256(inside(root,name))
        def ensure(relative):
            if (root/relative).exists():return
            if relative in resolving:raise ValueError("Cyclic runtime data binding: "+relative)
            resolving.add(relative)
            write_json(inside(root,relative),resolve(pending[relative]))
            resolving.remove(relative)
        ensure("references/FROZEN_REFERENCE_REGISTRY.json")
        registry_sha=sha256(root/"references/FROZEN_REFERENCE_REGISTRY.json")
        ref_by_sha={sha256(root/x):x for x in files if x.startswith("references/") and x.endswith(".png")}
        semantic=copy.deepcopy(state["S"])
        relocation=[]
        for row in semantic:
            name=ref_by_sha.get(row["reference_sha256"])
            if name is None:raise ValueError("Memory reference is absent from the registered runtime assets")
            relocation.append({"object_key":row["object_key"],"original_reference_path":row["reference_path"],
                "reference_sha256":row["reference_sha256"],"original_registry_sha256":row["registry_sha256"]})
            row["reference_path"]=str(output/name);row["registry_sha256"]=registry_sha
        bank=root/"memory";bank.mkdir()
        (bank/"PARAMETER.jsonl").write_bytes(state["P_bytes"])
        (bank/"SEMANTIC.jsonl").write_bytes(b"".join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode() + b"\n" for row in semantic))
        (bank/"POSE_MEMORY.json").write_bytes(state["pose_bytes"])
        memory_values.update(P_sha256=sha256(bank/"PARAMETER.jsonl"),S_sha256=sha256(bank/"SEMANTIC.jsonl"),
            pose_path=str(output/"memory/POSE_MEMORY.json"),pose_sha256=sha256(bank/"POSE_MEMORY.json"))
        for relative in pending:ensure(relative)
        write_json(root/"ADMITTED_MEMORY.json",{"source_snapshot_sha256":state["snapshot_sha256"],
            "suite":state["manifest"]["suite"],"generation":state["manifest"]["generation"],
            "source_parent_by_round":state["manifest"]["provenance"].get("source_parent_by_round",{}),
            "reference_relocation":relocation,"parameter_values_unchanged":True,"pose_values_unchanged":True})
        write_json(root/"RESOURCES.json",resources)
        contract_sha=sha256(root/"contracts/COMMON_CONTRACT.json")
        report={"kind":"prepared_memory_load","status":"MEMORY_LOADED",
            "new_knowledge":False,"contract_sha256":contract_sha,"initial_store_sha256":
            {"P":memory_values["P_sha256"],"S":memory_values["S_sha256"]},
            "store_sha256":{"PARAMETER.jsonl":memory_values["P_sha256"],"SEMANTIC.jsonl":memory_values["S_sha256"]},
            "source_memory_sha256":state["snapshot_sha256"]}
        write_json(bank/"LOAD_REPORT.json",report)
        write_json(root/"ACTIVE.json",{"snapshot":str(output/"memory"),"report_file":"LOAD_REPORT.json",
            "report_sha256":sha256(bank/"LOAD_REPORT.json"),"contract_sha256":contract_sha})
        write_json(root/"RUNTIME.json", {"schema": "carryon-recovery-runtime/v1",
            "suite": state["manifest"]["suite"], "generation": state["manifest"]["generation"],
            "required_files": [p.relative_to(root).as_posix() for p in sorted(root.rglob('*')) if p.is_file()]})
    return read_json(output/"RUNTIME.json")
