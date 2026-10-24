"""Versioned absolute parameter memory, with the admitted run's parent lineage."""
import json
import types
from dataclasses import is_dataclass,replace
from runtime_paths import required_file

SCHEMA="carryon-two-round-vlm-z-object-memory/v2"

def install(inner):
    from parameter_rationale_preference import LengthPreferenceReflector as C
    if getattr(C,"_carryon_versioned_reader",False):return
    metadata=json.loads(required_file("ADMITTED_MEMORY.json").read_text())
    parents=metadata["source_parent_by_round"]
    old=C._parse_persistent_row
    def read(self,value,*,line_number):
        if value.get("schema")!=SCHEMA:return old(self,value,line_number=line_number)
        required={"schema","created_utc","object_key","corrected_z_m","prior_z_m","rationale","value_source","source_v2"}
        if set(value)!=required:raise ValueError("Versioned source fields")
        source=value["source_v2"];key=value["object_key"]
        if not isinstance(key,str) or not key or key!=key.strip().lower():raise ValueError("Object key")
        if value["value_source"]!="vlm_runtime" or value["prior_z_m"]!=0:raise ValueError("Absolute value provenance")
        if source.get("schema")!="bound-mechanical-source/v2" or not isinstance(source.get("classes"),list):raise ValueError("Source schema")
        allowed={"DESCENT_JAM_ACTUALLY_EXECUTED","TERMINAL_MEASURED_EMPTY","MEASURED_LOSS_AFTER_NONEMPTY","MEASURED_WIDTH_UNSTABLE"}
        if not set(source["classes"])<=allowed:raise ValueError("Unknown source class")
        generation=source.get("reflection_round")
        if generation not in (1,2) or isinstance(generation,bool) or source.get("parent_snapshot_sha256")!=parents.get(str(generation)):
            raise ValueError("Parameter source parent mismatch")
        digest=source.get("batch_sha256")
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):raise ValueError("Batch SHA")
        decision=C._numeric_parse(self,json.dumps({"corrected_z_m":value["corrected_z_m"],"rationale":value["rationale"]}),current_z_m=source["anchor_z_m"])
        changes=dict(prior_z_m=0.,source="vlm_persistent_memory",latch_reused=True,write_succeeded=True)
        if is_dataclass(decision):decision=replace(decision,**changes)
        else:
            for key2,value2 in changes.items():setattr(decision,key2,value2)
        return key,decision
    C._parse_persistent_row=read;C._carryon_versioned_reader=True
    base=inner.selection_module.memory_module;loader=base.load_parameter_memory
    native_rows=loader.__globals__["_jsonl_rows"]
    def rows(path):
        result=[]
        for n,value in enumerate(native_rows(path),1):
            if value.get("schema")==SCHEMA:
                read(types.SimpleNamespace(min_z_m=0.,max_z_m=.25,max_abs_delta_m=.20),value,line_number=n)
                value=dict(value,schema="carryon-fixv2-vlm-z-object-memory/v1")
            result.append(value)
        return result
    env=dict(loader.__globals__,_jsonl_rows=rows)
    replacement=types.FunctionType(loader.__code__,env,loader.__name__,loader.__defaults__,loader.__closure__)
    replacement.__kwdefaults__=loader.__kwdefaults__;base.load_parameter_memory=replacement
    return {"original_loader_code_preserved":replacement.__code__ is loader.__code__,"dynamic_parent_binding":True}
