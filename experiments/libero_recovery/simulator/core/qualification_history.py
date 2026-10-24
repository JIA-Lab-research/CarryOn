import asyncio,base64,hashlib,math,threading,copy
from collections import deque
from types import SimpleNamespace
from urllib.parse import parse_qs,urlencode
import numpy as np
class RGBHistory:
 def __init__(self,session):
  if not session:raise ValueError('explicit per-job history session required')
  self.session=session;self.frames=deque(maxlen=16);self.lock=threading.Lock()
 def clear(self):
  with self.lock:self.frames.clear()
 def publish(self,frame):
  # RGB only; no depth, render, env.step or JPEG work here.
  rgb=np.array(frame.rgb,dtype=np.uint8,copy=True,order='C');rgb.flags.writeable=False
  f=SimpleNamespace(rgb=rgb,ts=frame.ts,producer_frame_seq=frame.producer_frame_seq,simulation_step_id=frame.simulation_step_id,simulation_time_s=frame.simulation_time_s,camera_role=frame.camera_role,camera_source=frame.camera_source)
  with self.lock:self.frames.append(f)
 def select(self,session,end):
  with self.lock:
   if session!=self.session:return None
   xs=[f for f in self.frames if f.ts<=end]
   return max(xs,key=lambda f:f.ts) if xs else None
 def rgb_bytes(self):
  with self.lock:return sum(f.rgb.nbytes for f in self.frames)
def history_response(history,query,encode,headers):
 q=parse_qs(query)
 try:
  session=q['session'][0];end=float(q['end'][0]);start=float(q['start'][0])
  if not all(map(math.isfinite,[start,end])) or start>end:raise ValueError()
 except Exception:return 400,b'',{}
 frame=history.select(session,end) if history else None
 if frame is None:return 404,b'',{}
 h=headers(frame);h.update({'X-CarryOn-History-Session':session,'X-CarryOn-Qualification-Delta':str(frame.ts-end),'X-CarryOn-Inside-Window':str(start<=frame.ts<=end).lower()})
 return 200,encode(frame.rgb),h
class FirstQualificationFetch:
 def __init__(self,job,generation,session,url,http,timeout_s):
  self.job=job;self.generation=generation;self.session=session;self.url=url;self.http=http;self.timeout_s=timeout_s;self.used=False;self.active=True;self.task=None;self.result=None
 def close(self):self.active=False
 def on_qualification(self,job,generation,q):
  if not self.active or self.used or (job,generation)!=(self.job,self.generation):return
  self.used=True;self.task=asyncio.create_task(self._fetch(copy.deepcopy(q)))
 async def _fetch(self,q):
  result={'anchor':None,'status':'MISSING','qualification':q}
  try:
   url=self.url+'?'+urlencode({'session':self.session,'start':q['window_start_producer_ts'],'end':q['window_last_producer_ts']})
   response=await self.http.get(url,timeout=self.timeout_s)
   if response.status_code==200:
    h=response.headers;ts=float(h['x-carryon-producer-frame-ts']);seq=int(h['x-carryon-producer-frame-seq'])
    if h.get('x-carryon-history-session')!=self.session or not math.isfinite(ts) or ts>q['window_last_producer_ts'] or seq<0 or not response.content:raise ValueError('invalid history identity')
    inside=q['window_start_producer_ts']<=ts
    result={'status':'WINDOW_OBSERVATION' if inside else 'PRIOR_CONTEXT','qualification':q,'anchor':{'producer_ts':ts,'producer_seq':seq,'session':self.session,'delta_to_qualification_end_s':ts-q['window_last_producer_ts'],'inside_window':inside,'sha256':hashlib.sha256(response.content).hexdigest(),'jpeg_base64':base64.b64encode(response.content).decode(),'held_identity':'UNDETERMINED'}}
   else:result['status']='MISSING_HTTP_'+str(response.status_code)
  except Exception as exc:result['status']='MISSING_'+type(exc).__name__
  if self.active:self.result=result
