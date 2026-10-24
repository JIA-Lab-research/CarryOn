"""Policy-specific execution gates; simulator outcomes never enter these gates."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

DUMMY = np.array([0., 0., 0., 0., 0., 0., -1.])


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NativeEight:
    """OFT/VLAct policy query, scan, trim and retreat operations."""
    def __init__(self, runtime, query, runtime_root):
        self.runtime, self.query = runtime, query
        module = load_module(Path(runtime_root) / 'adapter/chunk_controller.py', 'carryon_native8_control')
        self.live = None
        self.step = 0
        self.controller = module.OpenVLAChunkController(lambda chunk: runtime.decide(self.live, chunk, self.step))
        self.pending = None
        self.initialized = self.halted = False
        self.queries, self.dispatches = [], []
        self.initial_identity = None

    def build(self, obs, phase, sequence):
        if not self.initialized:
            grounding = self.runtime.reset(obs)
            self.initial_identity = {'grounding_status': grounding, **self.runtime.identity()}
            self.initialized = True
        elif self.pending is not None:
            self.runtime.commit(self.pending, obs)
            self.pending = None
        self.live = obs
        if phase == 'settle':
            self.pending = DUMMY.copy()
            return self.pending.tolist()
        if not self.halted and self.controller.needs_policy_query:
            actions, receipt = self.query(obs, self.step, sequence)
            self.controller.install_native_chunk(actions)
            self.queries.append({'sequence': sequence, **receipt})
        dispatch = None if self.halted else self.controller.next()
        if dispatch is None:
            self.halted = True
            self.pending = DUMMY.copy()
        else:
            self.pending = np.asarray(dispatch.action, dtype=np.float64)
        self.dispatches.append({'sequence': sequence, 'control_step': self.step,
            'source': 'fail_closed' if dispatch is None else dispatch.source,
            'cache_index': None if dispatch is None else dispatch.cache_index,
            'action': self.pending.tolist(), 'decision': None if dispatch is None else dispatch.decision})
        self.step += 1
        return self.pending.tolist()


class Pi05Gate:
    """Run the C=5/E=3 loop while retaining absolute step indices and executed history."""
    def __init__(self, runtime, query, runtime_root, *, sft):
        self.runtime, self.query, self.sft = runtime, query, sft
        module = load_module(Path(runtime_root) / 'adapter/pi05_control.py', 'carryon_pi05_control')
        self.controller = module.Batch2ActionController()
        self.pending = None
        self.initialized = self.halted = self.fail_closed = False
        self.queries, self.dispatches = [], []
        self.initial_identity = None
        self.step = 0

    def observe(self, obs, phase, sequence):
        if not self.initialized:
            grounding = self.runtime.reset(obs)
            self.initial_identity = {'grounding_status': grounding, **self.runtime.identity()}
            self.fail_closed = self.runtime.state()['fail_closed']
            self.initialized = True
        elif self.pending is not None:
            self.runtime.commit(self.pending, obs)
            self.pending = None

    def adapt(self, values):
        actions = np.asarray(values, dtype=np.float64).copy()
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 5 or not np.isfinite(actions).all():
            raise ValueError('pi0.5 needs at least five finite 7D action rows')
        if self.sft:
            actions[:, :3] *= 20.0
            actions[:, 3:6] *= 2.0
            actions[:, :6] = np.clip(actions[:, :6], -1.0, 1.0)
        return actions

    def build(self, obs, phase, sequence):
        t = int(sequence)
        decision = None
        source = 'settle'
        if phase == 'settle' or self.halted:
            action = DUMMY.copy()
            source = 'settle' if phase == 'settle' else 'fail_closed'
        else:
            controller = self.controller
            if controller.decision_due(t):
                gap = controller.begin_decision(t)
                if self.fail_closed:
                    chunk = np.repeat(DUMMY[None, :], 5, axis=0)
                    kind = 'fail_closed_no_policy'
                elif controller.in_hold:
                    chunk = controller.hold_lookahead()
                    kind = 'hold_lookahead_no_policy'
                else:
                    actions, receipt = self.query(obs, self.step, t)
                    chunk = self.adapt(actions)
                    self.queries.append({'sequence': t, **receipt})
                    controller.note_fresh_inference(t)
                    kind = 'fresh_policy'
                mode, evade, decision = self.runtime.decide(obs, chunk, t)
                decision.update(t=t, decision_cadence=5, decision_gap=gap,
                    inference_kind=kind, hold_left_before_decision=controller.hold_left)
                if mode == 'fail_closed':
                    self.halted = True
                    controller.clear_all()
                elif mode == 'evade':
                    controller.start_or_refresh_hold(evade, t)
                elif mode in ('pass_through', 'trim_grant'):
                    grant = int(decision['grant_steps'])
                    controller.commit_scan_grant(t, grant)
                    if not controller.in_hold:
                        controller.set_policy_plan(chunk, grant_steps=grant)
                else:
                    raise RuntimeError('Undeclared Shield mode: ' + str(mode))
                decision.update(next_decision_step=controller.next_decision_step,
                                hold_left_after_decision=controller.hold_left)
            if self.halted:
                action, source = DUMMY.copy(), 'fail_closed'
            else:
                action, override = controller.action_for_step(t)
                source = 'evade' if override is not None else 'nominal'
        self.pending = np.asarray(action, dtype=np.float64)
        self.dispatches.append({'sequence': t, 'source': source, 'action': self.pending.tolist(), 'decision': decision})
        if phase != 'settle':
            self.step += 1
        return self.pending.tolist()


def input_fingerprint(observation):
    """Preserve paired-policy input dtype and shape."""
    fields = {}
    def add(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                add(value[key], path + "/" + str(key))
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            fields[path] = {"dtype": str(array.dtype), "shape": list(array.shape),
                            "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        else:
            fields[path] = {"type": type(value).__name__, "value": value}
    add(observation, "")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pi05_query(client, row, runtime_root):
    from openpi_client import image_tools
    module = load_module(Path(runtime_root) / 'adapter/pi05_control.py', 'carryon_pi05_preprocess')
    def query(obs, control_step, sequence):
        agent = np.ascontiguousarray(obs['policy_agentview_raw'][::-1, ::-1])
        wrist = np.ascontiguousarray(obs['policy_wrist_raw'][::-1, ::-1])
        state = np.concatenate((obs['robot0_eef_pos'], module.quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']))
        element = {'observation/image': image_tools.convert_to_uint8(image_tools.resize_with_pad(agent, 224, 224)),
            'observation/wrist_image': image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
            'observation/state': state, 'prompt': obs['task_description']}
        key = {'pair_key': {'suite': row['suite'], 'level': row['level'], 'task': row['task_index'],
                           'seed': row['seed'], 'init': row['init_index']}, 'absolute_step': sequence}
        expected_input = input_fingerprint(element)
        answer = client.infer(dict(element, __pair_rng__=key))
        if answer.get('pair_input_sha256') != expected_input:
            raise RuntimeError('Policy input receipt mismatch')
        seed = int.from_bytes(hashlib.sha256(json.dumps(key, sort_keys=True, separators=(',', ':')).encode()).digest()[:4], 'big')
        action = np.ascontiguousarray(answer['actions'])
        if answer['pair_rng_seed'] != seed or answer['pair_action_sha256'] != hashlib.sha256(action.tobytes()).hexdigest():
            raise RuntimeError('Policy action/RNG receipt mismatch')
        return action, {k: answer[k] for k in ('pair_rng_seed', 'pair_input_sha256', 'pair_action_sha256')}
    return query
