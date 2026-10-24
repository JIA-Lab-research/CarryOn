import os
"""Persistent native SAM worker for the frozen CB1 input/output functions."""
from pathlib import Path
import contextlib,hashlib,importlib.util,json,os,sys,time,traceback,types
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cb3_frozen_sam',ROOT/'carryon/recovery/sam3_runner.py');S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
import destination_cb1 as CB1
with contextlib.redirect_stdout(sys.stderr):
 a=types.SimpleNamespace(model=os.environ['SAFE_SAM3_MODEL'],threshold=.35,mask_threshold=.5,seed=123,device='cuda',backend='native')
 _,model,processor,torch,np,Image=S._load_model(a)
print(json.dumps({'ready':True,'pid':os.getpid(),'worker':str(Path(__file__).resolve()),'settings':vars(a)}),flush=True)
for line in sys.stdin:
 try:
  r=json.loads(line)
  if r.get('cmd')=='shutdown':break
  z=np.load(r['input']);rgb=z['rgb'];image=Image.fromarray(rgb);start=time.perf_counter()
  with contextlib.redirect_stdout(sys.stderr):
   if r['mode']=='descriptor':
    if r.get('point') is not None:
     point=r['point'];state=processor.set_image(image)
     raw,scores,_=model.predict_inst(state,point_coords=np.asarray([point],np.float32),point_labels=np.array([1],np.int64),multimask_output=True)
     masks=S._to_numpy(raw).astype(bool);box=CB1.prompt_box(masks,point)
     np.savez_compressed(r['output'],masks=masks,scores=S._to_numpy(scores),point=point)
    else:
     m=z['initial_mask'].astype(bool);ys,xs=np.where(m);box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],np.float32) if len(xs) else None
     np.savez_compressed(r['output'],masks=m[None])
    result={'box':None if box is None else box.tolist(),'native_model_calls':int(r.get('point') is not None)}
   else:
    mask,scores=CB1.segment_current(model,processor,image,r.get('box'));np.save(r['output'],mask)
    result={'mask_pixels':int(mask.sum()),'scores':scores,'native_model_calls':int(r.get('box') is not None)}
  result.update(ok=True,request_id=r['request_id'],rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),latency_s=time.perf_counter()-start)
  print(json.dumps(result),flush=True)
 except BaseException as e:
  print(json.dumps({'ok':False,'error':repr(e),'traceback':traceback.format_exc()}),flush=True)
  break
