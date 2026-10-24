"""Shield and response predictor assembly for OpenVLA; no policy action adaptation here."""
import importlib.util,json,os,pathlib,sys,types
import numpy as np

SNAPSHOT_ROOT=pathlib.Path(__file__).resolve().parent.parent
METHOD=SNAPSHOT_ROOT/'shield'
RP1=SNAPSHOT_ROOT/'predictor'
CONTRACTS_SNAPSHOT=METHOD/'carryon/libero_safety/contracts.py'
TRIGGER=0.0667424414807498
FLOOR=0.0151


def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def configure_environment(diagnostics):
 values={'SAFE_RELATIONAL_ON':'1','SAFE_SHIELD_MODE':'relational','SAFE_REPLAN':'5','H':'12','EVADE_STEPS':'3','TAU':'.20','SAFE_ROLES_MODE':os.environ.get('SAFE_ROLES_MODE','live'),'SAFE_TRACKING_ENABLE':'0','SAFE_ROLLOUT_PREDICTOR':'1','SAFE_ROLLOUT_PREDICTOR_DIR':str(METHOD/'safety_eval'),'SAFE_V2_STAGE':str(METHOD/'safety_eval'),'SAFE_TRIGGER_GRIPPER_HELD_M':str(TRIGGER),'SAFE_MAX_STEPS':'900','SAFE_G_BATCH':'0','SAFE_RAW_MODE':'run','SAFE_CAPTURE_TELEMETRY':'1','SAFE_CB3_DIAGNOSTICS':str(pathlib.Path(diagnostics).resolve())}
 os.environ.update(values);return values

class FrozenRuntime:
 def __init__(self,loader_path,diagnostics):
  self.environment=configure_environment(diagnostics)
  self.roles_events=[];self.roles_provenance=None
  self.policy_family=os.environ.get("SAFE_POLICY_FAMILY","native8")
  if self.policy_family not in ("native8","pi05"):raise ValueError("Unknown policy family")
  self.arm_trigger=(FLOOR+5*0.010328488296149959) if self.policy_family=="pi05" else TRIGGER
  os.chdir(METHOD)
  sys.path[:0]=[str(METHOD),str(METHOD/'safety_eval')]
  for name,path in [('carryon',METHOD/'carryon'),('carryon.recovery',METHOD/'carryon/recovery'),('carryon.planner',METHOD/'carryon/planner')]:
   module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
   if '.' in name:setattr(sys.modules[name.rpartition('.')[0]],name.rpartition('.')[2],module)
  libero_pkg=types.ModuleType('carryon.libero_safety');libero_pkg.__path__=[str(CONTRACTS_SNAPSHOT.parent)];sys.modules['carryon.libero_safety']=libero_pkg;setattr(sys.modules['carryon'],'libero_safety',libero_pkg)
  load_module('carryon.libero_safety.contracts',CONTRACTS_SNAPSHOT)
  load_module('carryon.recovery.schema',METHOD/'carryon/recovery/schema.py')
  load_module('carryon.recovery.primitives',METHOD/'carryon/recovery/primitives.py')
  loader=load_module('openvla_frozen_loader',loader_path);self.imports=loader.load_method(METHOD)
  sys.path.insert(0,str(RP1))
  from online_predictor import ResponsePredictor,declared_terminal_spec,install_native_block
  import relational_scene as RSC,relational_shield as RSH,degraded_record as DR,shield_adapter as ADA
  from carryon.recovery.geometry import MolmoConfig,MolmoPointer,Sam3Config,Sam3Segmenter
  self.RSC,self.RSH,self.ADA=RSC,RSH,ADA;self.calibration=None
  self.native_rollout_calls=install_native_block(sys.modules['rollout_predictor'])
  seg=Sam3Segmenter(Sam3Config(model_path=os.environ['SAFE_SAM3_MODEL'],python_exe=os.environ['SAFE_SAM3_PYTHON'],threshold=.35,retry_threshold=-1.0,mask_threshold=.5,cuda_visible_devices=os.environ['SAFE_SAM3_GPU'],persistent=True))
  _molmo_gpu=os.environ.get('SAFE_MOLMO_GPU','').strip()
  _point_pointer,_point_pointer_unavailable=None,None
  if not _molmo_gpu:
   _point_pointer_unavailable='SAFE_MOLMO_GPU_EMPTY'
  else:
   try:
    _point_pointer=MolmoPointer(MolmoConfig(model_path=os.environ['SAFE_MOLMO_MODEL'],python_exe=os.environ['SAFE_MOLMO_PYTHON'],pythonpath_prepend=os.environ.get('SAFE_MOLMO_PYTHONPATH',''),cuda_visible_devices=_molmo_gpu,max_new_tokens=128))
   except Exception as _exc:
    _point_pointer_unavailable=type(_exc).__name__
  self.point_pointer,self.point_pointer_unavailable=_point_pointer,_point_pointer_unavailable
  def smask(pr,depth,w2p,H,W):
   sph=ADA._fk_spheres(pr)[0]
   fk=RSC.fk_self_mask([c for c,r in sph],[r for c,r in sph],depth,w2p,H,W)
   mount,_=ADA.relational_robot_mount_self_masks(pr,depth,self.calibration)
   base=RSC.base_link0_static_self_mask(depth,w2p,np.asarray(sph[0][0],float),H,W)
   return np.asarray(fk,bool)|np.asarray(mount,bool)|np.asarray(base,bool)
  def labels(rgb,depth,c2w,w2p,proprio,roles,segmenter,fk_spheres_fn,**kwargs):
   return RSC.build_labels(rgb,depth,c2w,w2p,proprio,roles,segmenter,fk_spheres_fn,self_mask_fn=smask,point_pointer=_point_pointer,pointer_unavailable_reason=_point_pointer_unavailable,**kwargs)
  rec=DR.DestGroundingRecorder(labels)
  links,radii=declared_terminal_spec(METHOD/'safety_eval/franka_attach32.yml');online_core=ResponsePredictor(links,radii)
  class ActionWindowPredictor:
   def __init__(self,core):self.core=core;self.window_records=[]
   def __getattr__(self,name):return getattr(self.core,name)
   def predict(self,actions):
    arr=np.asarray(actions,float);n=len(arr)
    if arr.shape!=(n,7) or n<1 or n>5 or not np.isfinite(arr).all():raise ValueError('OPENVLA_PREDICTOR_WINDOW_MUST_BE_FINITE_1_TO_5x7')
    valid=arr.copy()
    if n<5:
     pad=np.zeros((5-n,7),dtype=float);pad[:,6]=arr[-1,6];arr=np.vstack([arr,pad])
    out=list(self.core.predict(arr))[:n]
    self.window_records.append({'valid_len':n,'padding_rows':5-n,'padding_executed':False,'padding_committed':False})
    return out
  online=ActionWindowPredictor(online_core)
  shield=RSH.RelationalShield(seg,rec,lambda pr:ADA._fk_spheres(pr)[0],lambda pr:ADA._fk_spheres(pr)[1],smask,ADA._predicted_offsets,FLOOR,ADA.GRIPPER_CLOSED_WIDTH,execution_window=5,trigger_distance=self.arm_trigger,trigger_distance_gripper=TRIGGER,trigger_distance_held=TRIGGER,relational_on=True,tracking_adapter=None,predictor=online)
  shield.dest_record=rec
  original=shield.fk_grip_fn
  def current_grip(pr):
   spheres=original(pr);online.prepare(pr,spheres);return spheres
  shield.fk_grip_fn=current_grip
  self.shield,self.predictor,self.loader=shield,online,loader
 def reset(self,obs):
  self.shield.reset_episode();self.calibration=obs['shield_camera_calibration']
  self.predictor.reset(obs,obs['shield_time_s'])
  from carryon.planner import instruction_roles as roles_module
  self.roles_events=[]
  emit=roles_module._emit_role_parse_infra
  def record(event):
   self.roles_events.append(dict(event));emit(event)
  roles_module._emit_role_parse_infra=record
  try:roles=roles_module.parse_roles(obs['task_description'])
  finally:roles_module._emit_role_parse_infra=emit
  self.roles_provenance=roles_module.get_roles_provenance(obs['task_description'])
  return self.shield.initialize_episode(obs['shield_agentview_rgb'],obs['shield_agentview_depth_m'],obs['shield_camera_to_world'],obs['shield_world_to_pixel'],obs['shield_proprio'],roles)
 def commit(self,action,obs_after):
  self.predictor.commit_executed(np.asarray(action,float),obs_after,obs_after['shield_time_s'])
 def decide(self,obs,chunk,step):
  return self.shield.decide(obs['shield_agentview_depth_m'],obs['shield_proprio'],np.asarray(chunk,float),rgb=obs['shield_agentview_rgb'],absolute_action_step=int(step))
 def state(self):
  return {"fail_closed":bool(self.shield.fail_closed)}
 def identity(self):
  return {'runtime_class':type(self.shield).__module__+'.'+type(self.shield).__name__,'predictor_class':type(self.predictor).__module__+'.'+type(self.predictor).__name__,'trigger_m':self.arm_trigger,'terminal_trigger_m':TRIGGER,'policy_family':self.policy_family,'floor_m':FLOOR,'execution_window':5,'evade_steps':3,'native_rollout':False,'native_rollout_calls':list(self.native_rollout_calls),'point_pointer_available':self.point_pointer is not None,'point_pointer_unavailable_reason':self.point_pointer_unavailable,'grounding_trace':dict(self.shield.dest_record.record),'roles_provenance':self.roles_provenance,'roles_events':list(self.roles_events)}
 def integration_records(self):
  return {'rp1_windows':list(self.predictor.window_records),'native_rollout_calls':list(self.native_rollout_calls),'executed_steps':int(self.predictor.executed_steps)}
 def close(self):
  cb3=getattr(self.shield,'_cb3',None)
  if cb3 is not None:cb3.close()
  for obj in (getattr(self.shield,'seg',None),self.point_pointer):
   stop=getattr(obj,'_stop_server',None)
   if stop is not None:stop()