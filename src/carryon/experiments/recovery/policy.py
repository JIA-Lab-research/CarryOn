"""Serve QwenOFT from its installed source and model cache."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
from ..configuration import validate_public_config
from ..files import bound_file,bound_cached_file,new_directory,read_json,write_json


def resource_directory(spec,*,model_cache=False):
    root=Path(spec["root"]).resolve()
    if not root.is_dir():raise ValueError("Policy resource directory is missing: "+str(root))
    for relative in spec.get("files",[]):
        (bound_cached_file if model_cache else bound_file)({"path":relative},root)
    return root


def prepare(spec,output):
    import yaml
    validate_public_config(spec)
    source=resource_directory(spec["source"])
    checkpoint=bound_file(spec["checkpoint"]).resolve()
    config=bound_file(spec["model_config"])
    statistics=bound_file(spec["statistics"])
    base=resource_directory(spec["base_vlm"],model_cache=True)
    original=yaml.safe_load(config.read_text())
    if original["framework"]["name"]!="QwenOFT":raise ValueError("Expected QwenOFT checkpoint")
    if original["framework"]["action_model"]["future_action_window_size"]!=7:
        raise ValueError("Frozen Recovery policy requires eight-action chunks")
    if checkpoint.suffix not in (".pt",".safetensors"):raise ValueError("Unsupported checkpoint format")
    output=Path(output).resolve()
    with new_directory(output,inputs=[source,checkpoint.parent,base,config.parent]) as root:
        (root/"checkpoints").mkdir()
        # The target is shared read-only data. Configuration is always a fresh copy.
        (root/"checkpoints"/checkpoint.name).symlink_to(checkpoint)
        # The pinned StarVLA factory selects its VLM from the path spelling.
        # Keep its Qwen3-VL selector while allowing an arbitrary shared cache name.
        (root/"models").mkdir()
        (root/"models/Qwen3-VL-base").symlink_to(base, target_is_directory=True)
        original["framework"]["qwenvl"]["base_vlm"]=str(output/"models/Qwen3-VL-base")
        original.pop("wandb_entity",None)
        original.pop("wandb_project",None)
        original["output_dir"]=str(output/"unused-training-output")
        original["run_root_dir"]=str(output/"unused-training-output")
        (root/"config.yaml").write_text(yaml.safe_dump(original),encoding="utf-8")
        shutil.copyfile(statistics,root/"dataset_statistics.json")
        write_json(root/"MODEL_BINDING.json",{"checkpoint":str(checkpoint),
            "source_revision":spec["source"].get("revision"),"base_model":str(base),"chunk_size":8,
            "edits":["base model path","local training output paths","omit training telemetry identity"]})
    return source,output/"checkpoints"/checkpoint.name


def serve(spec,output):
    if os.name!="posix":raise ValueError("QwenOFT model service requires Linux")
    source,checkpoint=prepare(spec,output)
    sys.path.insert(0,str(source))
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    wrapper=PolicyServerWrapper(ckpt_path=str(checkpoint),device=spec["device"],use_bf16=spec["use_bf16"])
    if wrapper.metadata["action_chunk_size"]!=8:raise ValueError("Loaded policy has an unexpected chunk size")
    write_json(Path(output)/"METADATA.json",wrapper.metadata)
    WebsocketPolicyServer(policy=wrapper,host="127.0.0.1",port=spec["port"],
        idle_timeout=spec.get("idle_timeout",1800),metadata=wrapper.metadata).serve_forever()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    serve(read_json(args.config),args.output)

if __name__=="__main__":main()
