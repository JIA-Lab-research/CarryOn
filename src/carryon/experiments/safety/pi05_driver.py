"""pi0.5 environment lifecycle and full-horizon evaluation."""
import numpy as np


def make_driver_class(native):
    class Pi05Driver(native.SafetySimulationDriver):
        def __init__(self, config):
            # The native8 driver uses an explicit marker scene. pi0.5 uses the
            # benchmark's own default scene and never moves an extra marker.
            self.config = config
            if config.settle_steps != 10 or config.max_control_steps != 900 or config.control_hz != 20:
                raise ValueError('Frozen pi0.5 execution budget mismatch')
            import torch
            previous = torch.load
            def compatible(*args, **kwargs):
                kwargs.pop('weights_only', None)
                return previous(*args, **kwargs)
            torch.load = compatible
            try:
                from libero.libero import benchmark
                from libero.libero.envs import OffScreenRenderEnv
                suite = benchmark.get_benchmark_dict()[config.suite_name]()
                task = suite.get_task_by_level_id(config.level, config.task_id)
                self.initial_states = suite.get_task_init_states(config.level, config.task_id)
                task_file = suite.get_task_bddl_file_path(config.level, config.task_id)
            finally:
                torch.load = previous
            self.task_description = str(task.language)
            self.env = OffScreenRenderEnv(bddl_file_name=str(task_file), camera_heights=256,
                                          camera_widths=256, camera_depths=True)
            self.env.seed(config.seed)
            self.base = self.env.env
            self.step_index = 0
            self.constraint_events, self.success_step = [], None
            self.native_stop_seen = False
            self.shield_static = None

        def reset(self):
            np.random.seed(native.episode_rng_seed(self.config.seed, self.config.init_index))
            self.env.reset()
            obs = self.env.set_init_state(self.initial_states[self.config.init_index])
            from robosuite.utils.camera_utils import get_camera_transform_matrix
            from safety_eval.camera_calibration import calibrated_camera
            w2p = get_camera_transform_matrix(self.base.sim, 'agentview', 256, 256)
            self.shield_static = {'world_to_pixel': w2p, 'camera_to_world': np.linalg.inv(w2p),
                'calibration': calibrated_camera(self.base.sim, {}, camera_name='agentview',
                    height=256, width=256, image_rotation_deg=180)}
            self.step_index = 0
            self.constraint_events, self.success_step = [], None
            self.native_stop_seen = False
            return self._project(obs)

        def step(self, action):
            obs, reward, stop, info = self.env.step(action)
            if 'cost' not in (info or {}):
                raise RuntimeError('Frozen cost mapping absent')
            from safety_eval.constraint_timeline import evaluate_step
            self.constraint_events.extend(evaluate_step(self.step_index,
                self.base.parsed_problem.get('constraints', []) or [], self.base._eval_predicate))
            if bool(stop) and self.success_step is None:
                self.success_step = self.step_index
                self.native_stop_seen = True
            self.step_index += 1
            # pi0.5 runs all 910 steps. Scoring uses the interval
            # from step 10 through first success, or step 909 if unsuccessful.
            return self._project(obs), reward, False, info

        def coordinator_payload(self, sequence, closure_class):
            from safety_eval.constraint_timeline import finalize
            from score_execution_window import score_arm
            result = super().coordinator_payload(sequence, closure_class)
            full = finalize(self.constraint_events, self.success_step, self.step_index)
            result['full_horizon_record'] = full
            result['execution_window'] = score_arm(full)
            return result
    return Pi05Driver
