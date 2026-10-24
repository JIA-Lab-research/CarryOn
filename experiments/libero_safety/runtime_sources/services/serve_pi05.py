"""Serve pi0.5 with the experiment paired random seeds."""
import argparse,hashlib,json,pathlib,os,sys,logging,time
import numpy as np
import jax
from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
p=argparse.ArgumentParser()
for flag in ['checkpoint','receipt','model-id','model-revision']:p.add_argument('--'+flag,required=True)
p.add_argument('--port',type=int,required=True);args=p.parse_args()
logging.basicConfig(level=logging.INFO)
ckpt=pathlib.Path(args.checkpoint).resolve(); conf=cfg.get_config('pi05_libero')
dc=conf.data.create(conf.assets_dirs,conf.model);norm=ckpt/'assets'/dc.asset_id/'norm_stats.json'
identity={'protocol':'paired_rng_persistent_v2','checkpoint_realpath':str(ckpt),'params_realpath':str((ckpt/'params').resolve()),'normalization_path':str(norm),'serving_config':repr(conf),'data_config':repr(dc),'argv':sys.argv,'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'policy_rng':'SHA256(sorted JSON of pair_key + absolute_step), first 32 bits big endian -> jax.random.key; reset before each infer; arm omitted','model_id':args.model_id,'model_revision':args.model_revision}
base=policy_config.create_trained_policy(conf,ckpt)
identity['actual_policy_class']=type(base).__module__+'.'+type(base).__name__
identity['input_transform']=repr(base._input_transform);identity['output_transform']=repr(base._output_transform)
def fingerprint(obs):
 fields={}
 def add(x,path):
  if isinstance(x,dict):
   for k in sorted(x):add(x[k],path+'/'+str(k))
  elif isinstance(x,np.ndarray):
   a=np.ascontiguousarray(x);fields[path]={'dtype':str(a.dtype),'shape':list(a.shape),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
  else:fields[path]={'type':type(x).__name__,'value':x}
 add(obs,'')
 return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':')).encode()).hexdigest(),fields
reqpath=pathlib.Path(args.receipt).with_suffix('.requests.jsonl')
class PairPolicy:
 def infer(self,obs):
  obs=dict(obs);key=obs.pop('__pair_rng__')
  digest=hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).digest()
  seed=int.from_bytes(digest[:4],'big');base._rng=jax.random.key(seed)
  input_hash,fields=fingerprint(obs)
  out=base.infer(obs)
  action=np.ascontiguousarray(out['actions']);action_hash=hashlib.sha256(action.tobytes()).hexdigest()
  out.update(pair_rng_seed=seed,pair_input_sha256=input_hash,pair_action_sha256=action_hash)
  with reqpath.open('a') as f:f.write(json.dumps({'key':key,'seed':seed,'input_sha256':input_hash,'input_fields':fields,'action_sha256':action_hash,'time':time.time()})+'\n')
  return out
 def reset(self):pass
metadata={k:identity[k] for k in ['protocol','checkpoint_realpath','normalization_path','pid','policy_rng','model_id','model_revision']}
pathlib.Path(args.receipt).write_text(json.dumps(identity,indent=2))
print('PAIRED_POLICY_READY '+json.dumps(metadata),flush=True)
WebsocketPolicyServer(PairPolicy(),host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()
