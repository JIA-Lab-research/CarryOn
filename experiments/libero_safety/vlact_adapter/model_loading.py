"""Load the configured VLAct checkpoint for the local policy service."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from policy import VLActPolicy, UPSTREAM_COMMIT, CHECKPOINT_REVISION

REQUIRED = ('config.yaml', 'dataset_statistics.json', 'checkpoints/steps_50000_pytorch_model.pt')


def path_only_config(config, base_vlm, expert):
    """Only dependency locations change; architecture and numerical settings do not."""
    cfg = copy.deepcopy(config)
    cfg['framework']['qwenvl']['base_vlm'] = str(Path(base_vlm).absolute())
    cfg['framework']['action_model']['expert_model_path'] = str(Path(expert).resolve())
    restored = copy.deepcopy(cfg)
    restored['framework']['qwenvl']['base_vlm'] = config['framework']['qwenvl']['base_vlm']
    restored['framework']['action_model'].pop('expert_model_path')
    if restored != config: raise RuntimeError('NON_PATH_CONFIG_CHANGE')
    return cfg


def base_model_alias(deployment, base_vlm):
    # The pinned factory dispatches by the Qwen3-VL spelling in this path.
    # Preserve that spelling without renaming or modifying the shared weights.
    alias = Path(deployment) / 'Qwen3-VL-base'
    alias.symlink_to(Path(base_vlm).resolve(), target_is_directory=True)
    return alias


def load_policy(args):
    import torch
    from omegaconf import OmegaConf
    from accelerate import PartialState
    upstream, checkpoint = Path(args.upstream).resolve(), Path(args.checkpoint_root).resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT: raise RuntimeError('UPSTREAM_REVISION_DRIFT')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True):
        raise RuntimeError('UPSTREAM_TRACKED_CHANGES')
    for rel in REQUIRED:
        if not (checkpoint / rel).is_file(): raise RuntimeError("Missing checkpoint file: " + rel)
    for root in (Path(args.base_vlm), Path(args.expert)):
        if not (root / 'config.json').is_file(): raise RuntimeError('Dependency unavailable: ' + str(root))
    # Save the path-only deployment config outside the immutable upstream checkpoint.
    deployment = Path(args.output).resolve() / 'deployment'
    deployment.mkdir()
    cfg = OmegaConf.to_container(OmegaConf.load(checkpoint / 'config.yaml'), resolve=True)
    base_alias = base_model_alias(deployment, args.base_vlm)
    cfg = path_only_config(cfg, base_alias, args.expert)
    OmegaConf.save(OmegaConf.create(cfg), deployment / 'config.yaml')
    (deployment / 'dataset_statistics.json').symlink_to(checkpoint / 'dataset_statistics.json')
    (deployment / 'checkpoints').mkdir()
    weight = deployment / 'checkpoints/steps_50000_pytorch_model.pt'
    weight.symlink_to(checkpoint / 'checkpoints/steps_50000_pytorch_model.pt')
    sys.path.insert(0, str(upstream))
    if cfg['framework']['name'] != 'QwenPI_v4': raise RuntimeError('VLACT_FRAMEWORK_DRIFT')
    from framework_loading import load_vlact_framework
    load_vlact_framework(upstream)
    PartialState()
    from starVLA.model.framework.base_framework import baseframework
    model = baseframework.from_pretrained(str(weight)).to(torch.bfloat16).to('cuda').eval()
    if (model.action_dim, model.action_horizon, model.num_inference_timesteps) != (7, 8, 4):
        raise RuntimeError('MODEL_CONFIG_DRIFT')
    policy = VLActPolicy(model, model.norm_stats)
    identity = {
        'upstream_commit': revision, 'checkpoint_revision': CHECKPOINT_REVISION,
        'configuration_changes': 'dependency paths only', 'model_class': type(model).__name__,
        'framework_import_scope': 'QwenPI_v4; unchanged upstream factory and model implementation',
        'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'source_modules': {}, 'dependency_files': {},
    }
    for name, module in list(sys.modules.items()):
        if name.startswith('starVLA.') and getattr(module, '__file__', None):
            p = Path(module.__file__).resolve()
            if not p.is_relative_to(upstream): raise RuntimeError('UPSTREAM_IMPORT_ESCAPE: ' + str(p))
            identity['source_modules'][name] = str(p)
    for root in (Path(args.base_vlm), Path(args.expert)):
        for p in sorted(root.iterdir()):
            if p.is_file(): identity['dependency_files'][str(p)] = p.stat().st_size
    return policy, identity
