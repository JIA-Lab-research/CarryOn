"""Observer-only timestamps. Never change decisions, actions, clocks or counters."""
import os,json,time,functools
from pathlib import Path
def emit(root,event,**v):
 try:
  p=Path(root)/'TIMELY_EVENTS.jsonl';b=(json.dumps(dict(event=event,wall_ts=time.time(),pid=os.getpid(),**v),default=str)+'\n').encode()
  fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
  try:os.write(fd,b)
  finally:os.close(fd)
 except Exception as e:
  print('[timely-instrument-error]',type(e).__name__,flush=True)
def budget(w):
 return dict(vla_steps=getattr(w,'_step',None),horizon=getattr(w,'horizon_override',None),mode=str(getattr(w,'_mode',None)),controller=getattr(w,'_active_controller_type',None),recovery_physical_steps=getattr(w,'_recovery_frozen_steps',None),episode_wall_cap_s=getattr(w,'_episode_wall_cap_s',None))
def install_sim(shim,world_module,root):
 W=world_module.SimWorld;old=W._update_snapshots
 @functools.wraps(old)
 def snapshot(w,*a,**k):
  v=old(w,*a,**k);now=time.time();b=budget(w);key=(b['vla_steps'],b['mode'],b['recovery_physical_steps'])
  if getattr(w,'_timely_last_key',None)!=key or now-getattr(w,'_timely_last_ts',0)>.1:
   emit(root,'SIM_BUDGET_SAMPLE',**b);w._timely_last_key=key;w._timely_last_ts=now
  return v
 W._update_snapshots=snapshot
 oldswap=W._swap_controller
 @functools.wraps(oldswap)
 def swap(w,*a,**k):
  before=budget(w);v=oldswap(w,*a,**k);emit(root,'CONTROLLER_SWITCH_COMPLETED',before=before,**budget(w));return v
 W._swap_controller=swap
 oldcancel=W.request_cancel
 @functools.wraps(oldcancel)
 def cancel(w,*a,**k):
  v=oldcancel(w,*a,**k);emit(root,'CANCEL_REQUEST_APPLIED',standby=k.get('standby',a[0] if a else None),**budget(w));return v
 W.request_cancel=cancel
 oldtick=W._tick_recovery
 @functools.wraps(oldtick)
 def tick(w,*a,**k):
  before=getattr(w,'_recovery_frozen_steps',0);v=oldtick(w,*a,**k)
  if getattr(w,'_recovery_frozen_steps',0)>before and not getattr(w,'_timely_recovery_motion_recorded',False):
   emit(root,'FIRST_RECOVERY_PHYSICS_STEP',**budget(w));w._timely_recovery_motion_recorded=True
  return v
 W._tick_recovery=tick
 emit(root,'SIM_TIMELY_ENTRY_INSTALLED',observer_only=True)
