"""Isolated real Recovery worker. Invoke this file with the simulation interpreter."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True)
    args=parser.parse_args(argv);cfg=json.loads(Path(args.config).read_text())
    runtime=Path(cfg["runtime"]).resolve();root=Path(cfg["root"]).resolve()
    if hashlib.sha256((root/"ROW_MANIFEST.json").read_bytes()).hexdigest()!=cfg["row_manifest_sha256"]:
        raise ValueError("Row manifest changed")
    # Sender reads credentials from the configured environment variable.
    # Constructors receive a placeholder; model requests use the bound transport.
    os.environ["CARRYON_COLLECTION_ONLY"]="1"
    os.environ.update({k:str(v) for k,v in {
        "CARRYON_RECOVERY_ENABLE":"1",
        "CARRYON_RECOVERY_PARAMETER_LEDGER_PATH":root/"PARAMETER_LEDGER.jsonl",
        "CARRYON_RECOVERY_SEMANTIC_LEDGER_PATH":root/"SEMANTIC_LEDGER.jsonl",
        "CARRYON_RECOVERY_V7_COMPAT_MANIFEST":runtime/"bindings/semantic_memory_config.json",
        "CARRYON_REFERENCE_REGISTRY_ID":hashlib.sha256((runtime/"references/FROZEN_REFERENCE_REGISTRY.json").read_bytes()).hexdigest(),
        "CARRYON_FORMAL_ASSEMBLY_CONFIG":root/"ASSEMBLY.json",
        "CARRYON_FORMAL_CUROBO_IPC_DIR":root/"curobo_ipc",
        "CARRYON_FORMAL_RUNTIME_FLAGS":root/"FORMAL_RUNTIME_FLAGS.json",
        "CARRYON_RECOVERY_EPISODE_DIR":root/"recovery_episodes",
        "CARRYON_VISUAL_OBJECT_MEMORY_DEBUG_DIR":root/"visual_object_memory",
        "CARRYON_SEMANTIC_EPISODE_KEY":root.name,
        "CARRYON_VISUAL_OBJECT_MEMORY_MOLMO_POINTER":"1",
        "CARRYON_VISUAL_OBJECT_MEMORY_POINT_RECOVERY":"0",
        "CARRYON_USE_STRATEGY_SELECTOR":"0","CARRYON_REFLECTOR":"0",
        "CARRYON_VERIFY_IDENTITY":"1","CARRYON_DISABLE_OBJECT_CATALOG":"0",
        "CARRYON_GOALSET_SCENE":"0","CARRYON_GRASP_Z_OFFSET":"0","CARRYON_GRASP_YAW_OFFSET_DEG":"0",
        "QUALIFICATION_HISTORY_OUTPUT":root/"qualification_history",
        "QUALIFICATION_HISTORY_SESSION":root.name,
        "QUALIFICATION_REAL_WIDTH_CONTRACT":runtime/"bindings/REAL_WIDTH_CONTRACT.json",
        "ANCHOR_BUDGET_ROOT":root,
    }.items()})
    here=Path(__file__).resolve().parent
    transport=load("carryon_recovery_explicit_transport",here/"model_transport.py")
    sender=transport.Sender(cfg["model_transport"],root/"model_requests",root.name)
    isolation=load("carryon_recovery_isolation",here/"isolated.py")
    inner=isolation.configure(runtime,root,pose_send=sender.pose,
        pose_route=cfg["model_transport"]["routes"].get("pose_classifier"))
    sender.install()
    # Collection hooks must not invoke an inline reflector.
    from carryon.recovery import jam_z_reflection,semantic_reflection
    class CollectionClient:
        def __init__(self,**kwargs): pass
        def complete(self,*args,**kwargs): raise RuntimeError("Inline reflection is disabled; use the round collector")
        async def _complete_async(self,*args,**kwargs): raise RuntimeError("Inline reflection is disabled")
    jam_z_reflection.OpenRouterJamZClient=CollectionClient
    semantic_reflection.OpenRouterSemanticClient=CollectionClient
    sys.argv=[str(runtime),"--config",cfg["runtime_config"]]
    return inner.main()

if __name__=="__main__":
    raise SystemExit(main())
