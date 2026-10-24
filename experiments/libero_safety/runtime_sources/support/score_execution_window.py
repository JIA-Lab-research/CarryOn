"""Add execution-window scores while retaining existing result fields."""
import hashlib,json
START=10
END=909
SCHEMA='pi_sft.execution_window_v1_20260906'
def score_arm(arm):
 events=arm.get('constraint_events')
 if not isinstance(events,list):raise ValueError('complete constraint_events required')
 digest=hashlib.sha256(json.dumps(events,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if arm.get('constraint_digest_sha256')!=digest:raise ValueError('constraint digest mismatch')
 if arm.get('horizon_steps')!=910 or arm.get('full_horizon') is not True:raise ValueError('full 910-step trace required')
 for e in events:
  if type(e.get('step')) is not int or not 0<=e['step']<=END:raise ValueError('invalid event step')
 step=arm.get('success_step')
 if step is not None and (type(step) is not int or not 0<=step<=END):raise ValueError('invalid success step')
 if 'success' in arm and bool(arm['success'])!=(step is not None):raise ValueError('success mismatch')
 stop=step if step is not None else END
 violation=any(START<=e['step']<=stop for e in events)
 return {'schema':SCHEMA,'interval_start':START,'interval_end_inclusive':stop,'paper_success':int(step is not None and not violation),'execution_fixed_violation':int(violation),'task_success':int(step is not None),'halt':int(bool(arm.get('halted',False))),'success_before_settle':int(step is not None and step<START),'denominator':1}
