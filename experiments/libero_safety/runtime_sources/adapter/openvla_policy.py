"""OpenVLA-OFT policy I/O using configured upstream resources and cache paths."""
import hashlib,inspect,math,os,pathlib,sys
import numpy as np
OPENVLA=None
OPENVLA_LIBERO=None

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()

def array_sha(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()

def quat2axisangle_official(quat):
 q=np.array(quat,copy=True);q[3]=np.clip(q[3],-1.0,1.0);den=np.sqrt(1.0-q[3]*q[3])
 if math.isclose(den,0.0):return np.zeros(3)
 return (q[:3]*2.0*math.acos(q[3]))/den

class OpenVLAPolicy:
 def __init__(self,checkpoint,upstream):
  global OPENVLA,OPENVLA_LIBERO
  OPENVLA=pathlib.Path(upstream).resolve();OPENVLA_LIBERO=OPENVLA/"experiments/robot/libero/LIBERO_PRO"
  sys.path[:0]=[str(OPENVLA),str(OPENVLA_LIBERO)];os.chdir(OPENVLA)
  from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model
  self.cfg=GenerateConfig(model_family="openvla",pretrained_checkpoint=str(pathlib.Path(checkpoint).resolve()),use_l1_regression=True,use_diffusion=False,use_film=False,num_images_in_input=2,use_proprio=True,center_crop=True,num_open_loop_steps=8,load_in_8bit=False,load_in_4bit=False,task_suite_name="libero_spatial",use_wandb=False)
  self.model,self.action_head,self.proprio_projector,self.noisy,self.processor=initialize_model(self.cfg)
  if self.noisy is not None:raise RuntimeError("diffusion component unexpectedly loaded")
  self.available_keys=sorted(self.model.norm_stats)
  for key in ("libero_spatial_no_noops","libero_10_no_noops"):
   if key not in self.available_keys:raise RuntimeError("normalization key absent "+key)
  from experiments.robot import robot_utils
  from experiments.robot.libero import run_libero_eval
  self.get_action=robot_utils.get_action;self.process_action=run_libero_eval.process_action
  self.module_paths={}
  components=(("robot_utils",robot_utils),("run_libero_eval",run_libero_eval),("model",inspect.getmodule(type(self.model))),("action_head",inspect.getmodule(type(self.action_head))),("proprio_projector",inspect.getmodule(type(self.proprio_projector))),("processor",inspect.getmodule(type(self.processor))))
  for name,module in components:
   if module is None or not getattr(module,"__file__",None):raise RuntimeError("component module identity absent "+name)
   p=pathlib.Path(module.__file__).resolve();self.module_paths[name]={"path":str(p)}
  cache=pathlib.Path(os.environ.get("HF_MODULES_CACHE","")).resolve()
  if not os.environ.get("HF_MODULES_CACHE"):raise RuntimeError("An explicit isolated HF_MODULES_CACHE is required")
  for name in ("model","processor"):
   if os.path.commonpath((str(cache),self.module_paths[name]["path"]))!=str(cache):raise RuntimeError("dynamic module escaped pinned cache")
 def identity(self,checkpoint):
  cp=pathlib.Path(checkpoint).resolve();return {"schema":"openvla_oft.raw.loaded_identity.v3","checkpoint_realpath":str(cp),"model_class":type(self.model).__module__+"."+type(self.model).__name__,"action_head_class":type(self.action_head).__module__+"."+type(self.action_head).__name__,"proprio_projector_class":type(self.proprio_projector).__module__+"."+type(self.proprio_projector).__name__,"processor_class":type(self.processor).__module__+"."+type(self.processor).__name__,"available_normalization_keys":self.available_keys,"module_paths":self.module_paths,"hf_modules_cache":str(pathlib.Path(os.environ["HF_MODULES_CACHE"]).resolve()),"python":sys.version,"torch":__import__("torch").__version__,"external_network_guard":True}
 def query(self,obs,task_id):
  key="libero_spatial_no_noops" if int(task_id)==0 else "libero_10_no_noops"
  self.cfg.task_suite_name="libero_spatial" if int(task_id)==0 else "libero_10";self.cfg.unnorm_key=key
  agent=np.ascontiguousarray(obs["policy_agentview_raw"][::-1,::-1]);wrist=np.ascontiguousarray(obs["policy_wrist_raw"][::-1,::-1])
  proprio=np.concatenate((np.array(obs["robot0_eef_pos"],copy=True),quat2axisangle_official(obs["robot0_eef_quat"]),np.array(obs["robot0_gripper_qpos"],copy=True)))
  if proprio.shape!=(8,):raise RuntimeError("proprio shape drift")
  policy_obs={"full_image":agent,"wrist_image":wrist,"state":proprio}
  raw=np.asarray(self.get_action(self.cfg,self.model,policy_obs,obs["task_description"],processor=self.processor,action_head=self.action_head,proprio_projector=self.proprio_projector,noisy_action_projector=None,use_film=False))
  if raw.shape!=(8,7) or not np.isfinite(raw).all():raise RuntimeError("model chunk shape/nonfinite drift")
  actions=np.stack([np.asarray(self.process_action(a,"openvla"),dtype=np.float64) for a in raw])
  if actions.shape!=(8,7) or not np.isfinite(actions).all():raise RuntimeError("processed chunk drift")
  return actions,{"unnorm_key":key,"raw_agent_sha256":array_sha(obs["policy_agentview_raw"]),"rotated_agent_sha256":array_sha(agent),"raw_wrist_sha256":array_sha(obs["policy_wrist_raw"]),"rotated_wrist_sha256":array_sha(wrist),"proprio_sha256":array_sha(proprio),"raw_action_sha256":array_sha(raw),"environment_action_sha256":array_sha(actions),"model_chunk_length":8}
