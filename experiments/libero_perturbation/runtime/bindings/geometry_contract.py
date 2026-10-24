"""Exact common geometry structural admission, no efficacy or new robot threshold."""
import json
from pathlib import Path
SCHEMA = 'common-native-two-direction/v1'

def check_common_contract(c, require, sha):
    g = c['common_geometry']
    require(c['template'] == 'REGROUND_V7_ALL_CANDIDATE_V1', 'unchanged semantic template')
    require(g['schema'] == SCHEMA and g['generator'] == 'common_grasp_geometry/v2', 'common generator')
    require(c['physical_grasps'] == 2 and c['planning_candidates'] == 2, 'common family/physical budget')
    require(g['max_recovery_invocations'] == 1 and g['per_direction_planning_candidates'] == 1, 'nested budget')
    require(g['P_read_count'] == 1 and g['same_xyz_z'] is True and (g['depth_retry'] is False), 'P family contract')
    require(g['candidate_order'] == ['short', 'long'] and g['long_trigger'] == ['verified_jam_exit', 'fresh_measured_empty', 'unstable_then_fresh_measured_empty'], 'retry trigger contract')
    require(g['required_wrapper_order'] == ['original_policy_guard', 'common_family_and_retry', 'v4_measurement', 'v5_holding', 'attempt_retention'], 'wrapper contract')
    require(g['module_files'] and len({x['name'] for x in g['module_files']}) == len(g['module_files']), 'module bindings')
    for x in g['module_files']:
        require(Path(x['path']).name == x['name'] and Path(x['path']).is_file(), 'common module drift: ' + x['name'])

def check_common_policy(p, c, require):
    steps = p['steps']
    ops = [s['op'] for s in steps]
    g = c['common_geometry']
    require(ops.count('retry') == 1 and ops.count('perceive_grasps') == 1 and ('close_gripper' not in ops), 'common outer policy')
    r = steps[ops.index('retry')]['params']
    body = r['body']
    bops = [s['op'] for s in body]
    require(r['mode'] == 'common_short_long_v1' and type(r['max_attempts']) is int and (r['max_attempts'] == 2), 'common retry budget')
    require(bops == ['open_gripper', 'goto_grasp_topdown_family', 'close_gripper', 'lift', 'assert_holding'], 'common primitive body')
    q = body[1]['params']
    require(q['verify'] is True and q['fresh_retry_max_attempts'] == 1 and (q['frozen_candidate_only'] is True) and (q['yaw_degs'] == []), 'exact one-target descent')
    require(next((s for s in steps if s['op'] == 'perceive_grasps'))['params']['top_k'] == 15, 'legacy perception request budget; produced family separately pinned to two')
    require(ops.index('perceive_grasps') < ops.index('retry') < ops.index('goto_place'), 'common order')
    require(p['handback_mode'] == 'pre_place' and c['failure_handback'] == ['failure_terminal_destination', 'failure_handback_open_gripper', 'failure_handback_ack'], 'single-destination failure handback')
