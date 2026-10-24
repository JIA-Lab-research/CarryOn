"""VLAct official LIBERO I/O adapter; no shield or benchmark truth inputs."""
from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager

import cv2
import numpy as np

UPSTREAM_COMMIT = '621b01bb830e1a400f12f4c05262add55ae3003a'
CHECKPOINT_REPO = 'StarVLA/VLAct_Qwen3PI_Libero_Plus_Finetune'
CHECKPOINT_REVISION = '2e288218ddcda26b16c5306f49b9e93799926cea'


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def prepare_example(obs):
    """Official order: native RGB -> rotate 180 degrees -> INTER_AREA 224x224."""
    images = []
    for key in ('policy_agentview_raw', 'policy_wrist_raw'):
        a = np.asarray(obs[key])
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError('Expected native uint8 RGB: ' + key)
        images.append(cv2.resize(np.ascontiguousarray(a[::-1, ::-1]),
                                 (224, 224), interpolation=cv2.INTER_AREA))
    instruction = obs['task_description']
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Nonempty task instruction required')
    # The upstream LIBERO entry computes proprio but does not send it.
    # The model applies its packaged CoT template; do not apply it twice here.
    return {'image': images, 'lang': instruction}


def to_environment_actions(normalized, stats):
    """Match ModelClient.unnormalize_actions, then eval_libero gripper mapping.

    Intentionally min/max, NOT the q01/q99 helper on baseframework.
    The latter is not invoked by the pinned official LIBERO evaluation client.
    """
    a = np.asarray(normalized)
    if a.shape != (8, 7) or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError('Expected finite floating-point 8x7 normalized chunk')
    lo, hi = np.asarray(stats['min']), np.asarray(stats['max'])
    mask = np.asarray(stats.get('mask', np.ones_like(lo, dtype=bool)))
    if lo.shape != (7,) or hi.shape != (7,) or mask.shape != (7,):
        raise ValueError('Invalid action statistics shape')
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo):
        raise ValueError('Invalid action statistics bounds')
    if mask.dtype != np.bool_:
        raise ValueError('Action mask must be boolean')
    a = np.clip(a, -1, 1)
    a[:, 6] = np.where(a[:, 6] < 0.5, 0, 1)
    raw = np.where(mask, 0.5 * (a + 1) * (hi - lo) + lo, a)
    # The official environment entry casts each component to float32.
    actions = raw.astype(np.float32)
    actions[:, 6] = 1.0 - 2.0 * (actions[:, 6] > 0.5)
    return actions


def pair_seed(episode_key, control_step):
    if not isinstance(control_step, int) or control_step < 0:
        raise ValueError('control_step must be a nonnegative integer')
    payload = json.dumps({'episode': episode_key, 'control_step': control_step},
                         sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


@contextmanager
def seeded_inference(seed, cuda_devices):
    """Restore RNG state after synchronous inference; no shared concurrent calls."""
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # Seed only generators whose state is saved above.
            torch.random.default_generator.manual_seed(seed)
            for device in cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


class VLActPolicy:
    """Thin wrapper around an already-loaded official QwenPI_v4 model."""
    def __init__(self, model, norm_stats):
        if set(norm_stats) != {'franka'}:
            raise ValueError('Pinned checkpoint requires one franka statistics block')
        self.model = model
        self.stats = norm_stats['franka']['action']

    def query(self, obs, episode_key, control_step):
        import torch
        example = prepare_example(obs)
        seed = pair_seed(episode_key, control_step)
        devices = sorted({p.device.index for p in self.model.parameters() if p.is_cuda})
        with seeded_inference(seed, devices), torch.inference_mode():
            result = self.model.predict_action(examples=[example], do_sample=False,
                                               use_ddim=True, num_ddim_steps=10)
        normalized = np.asarray(result['normalized_actions'])
        if normalized.shape != (1, 8, 7):
            raise ValueError('Official model must return 1x8x7')
        actions = to_environment_actions(normalized[0], self.stats)
        return actions, {
            'pair_seed': seed, 'model_chunk_length': 8, 'unnorm_key': 'franka',
            'normalization': 'official_libero_client_min_max',
            'input_keys': sorted(example), 'uses_proprio': False,
            'image_sha256': [array_sha(x) for x in example['image']],
            'instruction_sha256': hashlib.sha256(example['lang'].encode()).hexdigest(),
            'normalized_actions_sha256': array_sha(normalized),
            'environment_actions_sha256': array_sha(actions),
        }
