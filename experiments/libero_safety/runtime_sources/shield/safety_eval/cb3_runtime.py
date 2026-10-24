"""Destination ownership and completion-state bookkeeping."""
from pathlib import Path
import hashlib,json,os,select,subprocess,sys,time
import numpy as np
from destination_cb1 import replace_destination_claim
from destination_cb3 import complete_boundary

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
class Destination:
 def __init__(self):
  self.output=Path(os.environ['SAFE_CB3_DIAGNOSTICS']);self.output.mkdir(parents=True,exist_ok=False)
  self.proc=None;self.box=None;self.step=None;self.mask=None;self.rows=[];self.claims=[]
 def _read(self,timeout):
  ready,_,_=select.select([self.proc.stdout],[],[],timeout)
  if not ready:raise RuntimeError('CB3_SAM_TIMEOUT')
  line=self.proc.stdout.readline()
  if not line:raise RuntimeError('CB3_SAM_EOF')
  return json.loads(line)
 def _start(self):
  if self.proc is not None:return
  path=Path(__file__).with_name('cb3_sam_worker.py');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=os.environ['SAFE_SAM3_GPU'];env['PYTHONUNBUFFERED']='1'
  self.log=(self.output/'worker.log').open('x')
  self.proc=subprocess.Popen([os.environ['SAFE_SAM3_PYTHON'],'-B',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env=env)
  identity=self._read(300)
  assert identity['ready'] and identity['pid']==self.proc.pid
  (self.output/'worker_identity.json').write_text(json.dumps(identity,indent=2))
 def _request(self,request):
  self._start();start=time.perf_counter();self.proc.stdin.write(json.dumps(request)+'\n');self.proc.stdin.flush();result=self._read(180)
  if not result.get('ok'):raise RuntimeError('CB3_SAM_REQUEST_ERROR:'+json.dumps(result))
  assert result['request_id']==request['request_id'];result['roundtrip_s']=time.perf_counter()-start;return result
 def initialize(self,rgb,initial_mask,fallback):
  self.initial_rgb=np.ascontiguousarray(rgb).copy();point=(fallback or {}).get('point_px')
  # Only the live, already-produced destination fallback point is used.
  inp=self.output/'initial_input.npz';np.savez_compressed(inp,rgb=rgb,initial_mask=initial_mask)
  result=self._request({'request_id':'initial','mode':'descriptor','input':str(inp),'output':str(self.output/'initial_return.npz'),'point':point})
  assert result['rgb_sha256']==digest(rgb);self.box=result['box']
  if point is not None:
   returned=np.load(self.output/'initial_return.npz');top=returned['masks'][int(np.argmax(returned['scores']))];assert np.array_equal(top,initial_mask),'CB3_INITIAL_POINT_RETURN_DRIFT'
  (self.output/'descriptor.json').write_text(json.dumps({'source':'same live initial RGB and original destination grounding result','point':point,**result},indent=2))
 def prepare(self,rgb,depth,w2p,step):
  if rgb is None or step is None:raise RuntimeError('CB3_MISSING_LIVE_INPUT')
  step=int(step)
  if self.step is not None and step<=self.step:raise RuntimeError('CB3_DUPLICATE_OR_NONMONOTONIC_DECISION')
  rgb=np.ascontiguousarray(rgb);depth=np.asarray(depth)
  assert rgb.dtype==np.uint8 and rgb.shape[:2]==depth.shape and rgb.shape[2]==3
  name='decision_%04d'%step;inp=self.output/(name+'_input.npz');mp=self.output/(name+'_sam.npy')
  np.savez_compressed(inp,rgb=rgb,depth=depth,w2p=w2p)
  result=self._request({'request_id':name,'mode':'current','input':str(inp),'output':str(mp),'box':self.box})
  assert result['rgb_sha256']==digest(rgb);raw=np.load(mp);start=time.perf_counter();mask,added=complete_boundary(raw,depth,w2p);elapsed=time.perf_counter()-start
  np.savez_compressed(self.output/(name+'_completion.npz'),mask=mask,added=added)
  self.mask=mask;self.step=step;self.depth_hash=digest(depth);self.w2p=np.asarray(w2p).copy();self.row={'t':step,'rgb_sha256':digest(rgb),'depth_sha256':self.depth_hash,'w2p_sha256':digest(w2p),'sam_mask_sha256':digest(raw),'completed_mask_sha256':digest(mask),'added_pixels':int(added.sum()),'empty_mask':not bool(raw.any()),'completion_s':elapsed,**result}
  self.rows.append(self.row)
  with (self.output/'calls.jsonl').open('a') as f:f.write(json.dumps(self.row)+'\n')
 def apply(self,world,floating,claimed_without_destination,depth):
  assert self.mask is not None and digest(depth)==self.depth_hash,'CB3_STALE_DEPTH_OR_MASK'
  other,dest,dc,om=replace_destination_claim(world,floating,claimed_without_destination,self.w2p,self.mask)
  row={'t':self.step,'world_points':len(world),'destination_points':int(dc.sum()),'other_points':int(om.sum())};self.claims.append(row)
  with (self.output/'claims.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return other,dest
 def close(self):
  if self.proc is not None:
   if self.proc.poll() is None:
    try:self.proc.stdin.write(json.dumps({'cmd':'shutdown'})+'\n');self.proc.stdin.flush();self.proc.wait(timeout=10)
    except Exception:self.proc.terminate();self.proc.wait(timeout=10)
   self.log.close()
