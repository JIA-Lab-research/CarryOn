"""Local-only official VLAct service, with arm-independent paired RNG."""
import argparse
import json
import os
from pathlib import Path

from model_loading import load_policy


class ServicePolicy:
    def __init__(self, policy): self.policy = policy

    def predict_action(self, observation, episode_key, control_step, **metadata):
        # Only prepare_example's RGB and instruction allowlist reaches the model.
        actions, receipt = self.policy.query(observation, episode_key, control_step)
        return {'actions': actions, 'receipt': receipt}


def main():
    p = argparse.ArgumentParser()
    for name in ('upstream', 'checkpoint-root', 'base-vlm', 'expert', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--port', type=int, required=True)
    args = p.parse_args()
    if not 1024 <= args.port <= 65535: raise ValueError('Invalid service port')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    policy, identity = load_policy(args)
    (out / 'LOADED_IDENTITY.json').write_text(json.dumps(identity, indent=2) + '\n')
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(ServicePolicy(policy), host='127.0.0.1', port=args.port,
                          metadata={'backbone': 'VLAct_Qwen3PI', 'chunk': 8,
                                    'checkpoint_revision': identity['checkpoint_revision']}).serve_forever()


if __name__ == '__main__': main()
