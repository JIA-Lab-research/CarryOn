#!/usr/bin/env python3
"""Conclusion-free LIBERO-Safety driver projection for OpenVLA-OFT raw."""
from __future__ import annotations
import hashlib,pathlib,re
from dataclasses import dataclass
from typing import Any,Mapping
import numpy as np

DUMMY_ACTION=[0.0,0.0,0.0,0.0,0.0,0.0,-1.0]
SCENES={0:"aegis_phase1_libero_tabletop_base_style_v1.xml",1:"aegis_phase1_libero_study_base_style_v1.xml",2:"aegis_phase1_libero_kitchen_tabletop_base_style_v1.xml",3:"aegis_phase1_libero_floor_base_style_v1.xml",4:"aegis_phase1_libero_kitchen_tabletop_base_style_v1.xml"}
PROBLEMS={0:"LIBERO_Tabletop_Manipulation",1:"LIBERO_Study_Tabletop_Manipulation",2:"LIBERO_Kitchen_Tabletop_Manipulation",3:"LIBERO_Floor_Manipulation",4:"LIBERO_Kitchen_Tabletop_Manipulation"}

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def episode_rng_seed(seed,init_index):return (int(seed)*100003+7919*int(init_index))%(2**32)

@dataclass(frozen=True)
class RawDriverConfig:
 safety_repo_root:str;reuse_tag:str;ordinal:int;suite_name:str;level:int;task_id:int;seed:int;init_index:int;problem_class:str;scene_xml_path:str;settle_steps:int=10;max_control_steps:int=900;control_hz:int=20
 def validate(self):
  if self.suite_name not in ("obstacle_avoidance","obstacle_avoidance_human"):raise ValueError("suite drift")
  if self.level not in (0,1,2) or self.task_id not in range(5):raise ValueError("cell drift")
  if self.seed not in (7,17,27) or self.init_index not in range(10):raise ValueError("population drift")
  if self.problem_class!=PROBLEMS[self.task_id]:raise ValueError("problem-class drift")
  scene=pathlib.Path(self.scene_xml_path).resolve()
  if scene.name!=SCENES[self.task_id] or not scene.is_file():raise ValueError("scene drift")
  if self.settle_steps!=10 or self.max_control_steps!=900 or self.control_hz!=20:raise ValueError("method constants drift")

class SafetySimulationDriver:
 def __init__(self,config):
  config.validate();self.config=config
  import torch as tm
  old=tm.load
  def compat(*args,**kwargs):kwargs.pop("weights_only",None);return old(*args,**kwargs)
  tm.load=compat
  try:
   from libero.libero import benchmark
   from libero.libero.envs import OffScreenRenderEnv
   suite=benchmark.get_benchmark_dict()[config.suite_name]();task=suite.get_task_by_level_id(config.level,config.task_id)
   self.initial_states=suite.get_task_init_states(config.level,config.task_id);task_file=pathlib.Path(suite.get_task_bddl_file_path(config.level,config.task_id)).resolve()
  finally:tm.load=old
  match=re.search(r"\(problem\s+([^\)]+)\)",task_file.read_text(encoding="utf-8"))
  if (match.group(1) if match else None)!=config.problem_class:raise RuntimeError("BDDL problem-class drift")
  self.task_description=str(task.language)
  self.env=OffScreenRenderEnv(bddl_file_name=str(task_file),scene_xml=str(pathlib.Path(config.scene_xml_path).resolve()),camera_names=["agentview","robot0_eye_in_hand"],camera_heights=[256,256],camera_widths=[256,256],camera_depths=[True,True])
  self.env.seed(config.seed);self.base=self.env.env;self.step_index=0;self.eef_body_id=None;self.constraint_events=[];self.success_step=None;self.native_stop_seen=False;self.shield_static=None
 def _set_marker(self,obs):
  from scipy.spatial.transform import Rotation
  if self.eef_body_id is None:raise RuntimeError("eef marker missing")
  p=np.asarray(obs["robot0_eef_pos"]);q=np.asarray(obs["robot0_eef_quat"]);r=Rotation.from_quat(q).as_matrix();center=p+r@np.array([0.0,0.0,-0.08])
  self.env.sim.model.body_pos[self.eef_body_id]=center;self.env.sim.model.body_quat[self.eef_body_id]=q[[3,0,1,2]]
 def _project(self,obs):
  from robosuite.utils.camera_utils import get_real_depth_map
  agent=np.array(obs["agentview_image"],copy=True,order="C")
  depth=get_real_depth_map(self.base.sim,obs["agentview_depth"]);depth=depth[...,0] if depth.ndim==3 else depth
  proprio=np.concatenate((np.asarray(obs["robot0_joint_pos"],dtype=np.float64),np.asarray(obs["robot0_eef_pos"],dtype=np.float64),np.asarray(obs["robot0_eef_quat"],dtype=np.float64),np.asarray(obs["robot0_gripper_qpos"],dtype=np.float64)))
  if proprio.shape!=(16,):raise RuntimeError("shield proprio shape drift")
  out={"robot0_eef_pos":np.array(obs["robot0_eef_pos"],copy=True),"robot0_eef_quat":np.array(obs["robot0_eef_quat"],copy=True),"robot0_gripper_qpos":np.array(obs["robot0_gripper_qpos"],copy=True),"task_description":self.task_description,"shield_agentview_rgb":np.ascontiguousarray(agent[::-1]),"shield_agentview_depth_m":np.ascontiguousarray(depth[::-1]),"shield_proprio":proprio,"shield_world_to_pixel":np.array(self.shield_static["world_to_pixel"],copy=True),"shield_camera_to_world":np.array(self.shield_static["camera_to_world"],copy=True),"shield_camera_calibration":self.shield_static["calibration"],"shield_time_s":float(self.base.sim.data.time)}
  if self.step_index>=self.config.settle_steps:
   wrist=np.array(obs["robot0_eye_in_hand_image"],copy=True,order="C")
   if agent.shape!=(256,256,3) or wrist.shape!=(256,256,3) or agent.dtype!=np.uint8 or wrist.dtype!=np.uint8:raise RuntimeError("raw camera contract drift")
   out.update({"task_description":self.task_description,"policy_agentview_raw":agent,"policy_wrist_raw":wrist})
  return out
 def reset(self):
  np.random.seed(episode_rng_seed(self.config.seed,self.config.init_index));self.env.reset();obs=self.env.set_init_state(self.initial_states[self.config.init_index])
  from robosuite.utils.camera_utils import get_camera_transform_matrix
  from safety_eval.camera_calibration import calibrated_camera
  w2p=get_camera_transform_matrix(self.base.sim,"agentview",256,256)
  self.shield_static={"world_to_pixel":w2p,"camera_to_world":np.linalg.inv(w2p),"calibration":calibrated_camera(self.base.sim,{},camera_name="agentview",height=256,width=256,image_rotation_deg=180)}
  self.eef_body_id=self.env.sim.model.body_name2id("eef_marker");self._set_marker(obs);self.step_index=0;self.constraint_events=[];self.success_step=None;self.native_stop_seen=False
  return self._project(obs)
 def step(self,action):
  record=self.env.step(action)
  if not isinstance(record,(tuple,list)) or len(record)!=4:raise RuntimeError("LIBERO-Safety step schema drift")
  obs,reward,stop,info=record
  if "cost" not in (info or {}):raise RuntimeError("frozen cost mapping absent")
  scoring_active=self.step_index>=self.config.settle_steps
  if scoring_active:
   from safety_eval.constraint_timeline import evaluate_step
   self.constraint_events.extend(evaluate_step(self.step_index,self.base.parsed_problem.get("constraints",[]) or [],self.base._eval_predicate))
   if bool(stop) and self.success_step is None:self.success_step=self.step_index;self.native_stop_seen=True
  effective_stop=bool(stop) if scoring_active else False
  self.step_index+=1
  if self.step_index>self.config.settle_steps:self._set_marker(obs)
  return self._project(obs),reward,effective_stop,info
 def coordinator_payload(self,sequence,closure_class):
  from safety_eval.constraint_timeline import first_violation_steps,protocol_success
  fixed,stock=first_violation_steps(self.constraint_events)
  return {"schema":"carryon.safety.episode_endpoints.v1","reuse_tag":self.config.reuse_tag,"ordinal":self.config.ordinal,"init_index":self.config.init_index,"suite":self.config.suite_name,"level":self.config.level,"task_id":self.config.task_id,"seed":self.config.seed,"episode_rng_seed":episode_rng_seed(self.config.seed,self.config.init_index),"closure_class":closure_class,"supervisor_sequence":int(sequence),"env_steps_observed":self.step_index,"native_method_stop_seen":self.native_stop_seen,"success":int(self.success_step is not None),"success_step":self.success_step,"fixed_violation":int(fixed is not None),"fixed_violation_step":fixed,"stock_violation":int(stock is not None),"stock_violation_step":stock,"paper_success_fixed":protocol_success(self.success_step,fixed),"paper_success_stock":protocol_success(self.success_step,stock),"constraint_events":self.constraint_events}
 def close(self):self.env.close()

