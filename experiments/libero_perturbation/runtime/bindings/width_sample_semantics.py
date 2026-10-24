"""Robot-state provenance only; no task rules, thresholds or physics control."""
import copy
PHYSICS_SOURCE = 'robot_observation_after_physics_step'
FROZEN_SOURCE = 'robot_sim_data_read_no_physics_step'
UNKNOWN_SNAPSHOT_SOURCE = 'robot_observation_snapshot'

def width_sample(state):
    return {'ts': state['ts'], 'width_m': state['w_meas'], 'measurement_source': 'w_meas', 'state_source': copy.deepcopy(state.get('state_source'))}

def dynamic_samples(samples):
    return [copy.deepcopy(s) for s in samples if isinstance(s, dict) and s.get('state_source') == PHYSICS_SOURCE]
