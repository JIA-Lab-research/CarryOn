"""Isolated recorded-scene entry; no ordinary LIBERO task/init fallback."""
from pathlib import Path
import ast,hashlib,json,os,re,logging
import numpy as np
BENCHMARK='LIBERO-Recovery-recorded-scene'
ASSET_ROOT=Path(os.environ['CARRYON_LIBERO_RECOVERY_ASSETS'])
SOURCE=Path(os.environ['CARRYON_LIBERO_RECOVERY_PROTOCOL_SOURCE'])
HELPERS=('_parse_language_from_bddl','build_object_xy_qpos_adrs','perturb_object_xy','settle_gripper_to','reset_controller_goal_to_current','_construct_env','_reset_env')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()
def checked_case(path):
 p=Path(path);c=json.loads(p.read_text())
 for key in ('bddl','state'):
  q=Path(c[key]);c[key]=str((ASSET_ROOT/q).resolve()) if not q.is_absolute() else str(q)
 if c.get('benchmark')!=BENCHMARK:raise ValueError('recorded-scene benchmark marker required; no ordinary-init fallback')
 for key in ('bddl','state'):
  q=Path(c[key]).resolve()
  if not q.is_relative_to(ASSET_ROOT.resolve()):raise ValueError('scene outside pinned asset root')
  if not q.is_file():raise FileNotFoundError(q)
 if Path(c['state'])!=Path(c['bddl']).with_name(Path(c['bddl']).stem+'_sim_state.npy'):raise ValueError('BDDL/state sibling mismatch')
 if c['gripper_init'] not in ('open','close'):raise ValueError('invalid gripper split')
 if not 0 <= c['trial_index'] < 10:raise ValueError('full benchmark requires ten trials per scene')
 if c['gripper_init'] != ('close' if c['trial_index'] < 5 else 'open'):raise ValueError('five close then five open required')
 if c.get('global_index') != c['scene_index']*10+c['trial_index']:raise ValueError('global case identity mismatch')
 if c.get('construction_seed') != 7+1000001*(c['scene_index']+1):raise ValueError('construction seed mismatch')
 if c.get('reset_seed') != c['construction_seed']+10001*(c['trial_index']+1):raise ValueError('reset seed mismatch')
 if c['seed']!=7 or c['obj_noise_xy']!=0.02 or c['settle_steps']!=20 or c['budget']!=990:raise ValueError('frozen protocol mismatch')
 return c

def load_helpers(factory,randomization_error):
 text=SOURCE.read_text();tree=ast.parse(text);ns={'np':np,'os':os,'re':re,'logging':logging,'OffScreenRenderEnv':factory,'RandomizationError':randomization_error}
 for name in HELPERS:
  node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  exec(compile(ast.get_source_segment(text,node),str(SOURCE),'exec'),ns)
 return ns

class ObserveSettle:
 def __init__(self,env,obs):self.env=env;self.last=obs
 def step(self,action):
  result=self.env.step(action);self.last=result[0];return result

def build_recorded_scene(world,task_id,episode_idx,*,factory=None,randomization_error=None):
 case_path=os.environ.get('LIBERO_RECOVERY_CASE')
 if not case_path:raise RuntimeError('LIBERO_RECOVERY_CASE required; ordinary suite init is forbidden')
 c=checked_case(case_path)
 if int(task_id)!=c['task_id'] or int(episode_idx)!=c['trial_index']:raise ValueError('shim case identity mismatch')
 if world.horizon_override!=c['budget']:raise ValueError('shared step budget mismatch')
 if factory is None:
  from libero.libero.envs import OffScreenRenderEnv
  from robosuite.utils.errors import RandomizationError
  factory=OffScreenRenderEnv;randomization_error=RandomizationError
 h=load_helpers(factory,randomization_error)
 lang=h['_parse_language_from_bddl'](c['bddl'])
 if lang!=c['instruction']:raise ValueError('BDDL task language mismatch')
 # Exactly the benchmark initialization protocol, evaluated one scene per invocation.
 # Frozen RNG state preserves the original across-scene stream under arbitrary sharding.
 # Initialization metadata is never exposed to policy, selector, or reflection.
 env_args={'bddl_file_name':c['bddl'],'camera_names':[world.camera_name,'robot0_eye_in_hand'],'camera_heights':[world.resolution,world.resolution],'camera_widths':[world.resolution,world.resolution],'camera_depths':True,'horizon':c['budget']+c['settle_steps']}
 env=h['_construct_env'](env_args,c['case_id'],c['construction_seed'],12)
 if env is None:raise RuntimeError('benchmark scene construction failed')
 try:
  env.seed(c['seed'])
  if not h['_reset_env'](env,c['case_id'],c['trial_index'],c['gripper_init'],c['reset_seed'],12):raise RuntimeError('benchmark scene reset failed')
  base=np.load(c['state'],allow_pickle=False);base_before=array_sha(base);addresses=h['build_object_xy_qpos_adrs'](env)
  if [list(x) for x in addresses] != c['expected_object_address_order']:raise ValueError('initialization object order differs from frozen CPU model')
  rs=c['rng_state_before'];rng=np.random.RandomState()
  rng.set_state((rs[0],np.asarray(rs[1],dtype=np.uint32),rs[2],rs[3],rs[4]))
  state,deltas=h['perturb_object_xy'](base,addresses,c['obj_noise_xy'],rng)
  if array_sha(state)!=c['expected_perturbed_tensor_sha256']:raise ValueError('global initialization RNG replay mismatch')
  if array_sha(base)!=base_before:raise AssertionError('benchmark state was mutated')
  obs=env.set_init_state(state);h['reset_controller_goal_to_current'](env)
  settling=ObserveSettle(env,obs);h['settle_gripper_to'](settling,close=c['gripper_init']=='close',num_steps=c['settle_steps']);obs=settling.last
  world._env=env;world._task_description=lang;world.num_steps_wait=c['settle_steps'];world._last_grip=1.0 if c['gripper_init']=='close' else -1.0
  world._producer_frame_seq=0;world._qualification_rgb_history.clear();world._idle_refresh_steps=0;world._last_idle_refresh_s=0.0
  world._build_controllers();world._update_snapshots(obs)
  receipt={'benchmark':BENCHMARK,'case_id':c['case_id'],'bddl':c['bddl'],'state':c['state'],'base_tensor_sha256':base_before,'perturbed_tensor_sha256':array_sha(state),'obj_deltas':deltas,'obj_address_order':addresses,'gripper_init':c['gripper_init'],'settle_steps':c['settle_steps'],'post_settle_observation_refreshed':True,'instruction':lang,'shared_post_settle_budget':c['budget'],'physical_horizon_including_settle':env_args['horizon'],'ordinary_task_init_used':False,'runtime_forced_disturbance':False}
  try:receipt['post_settle_state_sha256']=array_sha(np.asarray(env.env.sim.get_state().flatten()))
  except AttributeError:receipt['post_settle_state_sha256']=None
  output=Path(os.environ['LIBERO_RECOVERY_LOAD_RECEIPT']);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
  return obs
 except BaseException:
  env.close();raise
