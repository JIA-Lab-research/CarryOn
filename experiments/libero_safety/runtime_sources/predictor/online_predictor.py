"""Maintain executed action history for the response predictor."""
import copy,hashlib,time
import numpy as np
from response_model import predict_response,goal_from_sent_action,DT
from terminal_geometry import simplified_terminal_steps
VERSION="PREDICTOR_ONLINE_v1"
def robot_pose(obs):
    return {k:np.asarray(obs[k],float).copy() for k in ("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos")}
class ResponsePredictor:
    def __init__(self,terminal_links,terminal_radii):
        self.links=tuple(terminal_links);self.radii=tuple(terminal_radii)
        self.reset_state()
    def reset_state(self):
        self.current=None;self.previous=None;self.clock=None;self.previous_clock=None
        self.goal=None;self.last_goal_step=None;self.executed_steps=0;self.prepared=None
        self.predict_calls=0;self.last_latency_ms=None;self.last_input=None
        self.last_restore_fidelity_ok=False
    def reset(self,observation,time_s=0.0):
        self.reset_state();self.current=robot_pose(observation);self.clock=float(time_s)
    def commit_executed(self,action,observation_after,time_after_s):
        if self.current is None:raise RuntimeError("PREDICTOR_RESET_OBSERVATION_MISSING")
        delta=float(time_after_s)-self.clock
        if not np.isclose(delta,DT,atol=1e-10,rtol=0):raise RuntimeError("PREDICTOR_CONTROL_INTERVAL_MISMATCH")
        goal=goal_from_sent_action(self.current["robot0_eef_quat"],action)
        if goal is not None:self.goal=goal;self.last_goal_step=self.executed_steps
        self.previous=self.current;self.previous_clock=self.clock
        self.current=robot_pose(observation_after);self.clock=float(time_after_s)
        self.executed_steps+=1;self.prepared=None
    def prepare(self,proprio,terminal_spheres):
        if self.current is None:raise RuntimeError("PREDICTOR_HISTORY_NOT_INITIALIZED")
        p=np.asarray(proprio,float)
        for arr,part in [(self.current["robot0_eef_pos"],p[7:10]),(self.current["robot0_eef_quat"],p[10:14]),(self.current["robot0_gripper_qpos"],p[14:16])]:
            if not np.array_equal(arr,part):raise RuntimeError("PREDICTOR_CURRENT_PROPRIO_NOT_OBSERVED")
        if len(terminal_spheres)!=len(self.links):raise RuntimeError("PREDICTOR_TERMINAL_COUNT_MISMATCH")
        if not np.array_equal([r for c,r in terminal_spheres],self.radii):raise RuntimeError("PREDICTOR_TERMINAL_RADIUS_ORDER_MISMATCH")
        self.prepared=[(np.asarray(c,float).copy(),float(r)) for c,r in terminal_spheres]
    def history_signature(self):
        def stable(v):
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,dict):return {k:stable(x) for k,x in v.items()}
            return v
        import json
        fields=[self.current,self.previous,self.clock,self.previous_clock,self.goal,self.last_goal_step,self.executed_steps]
        return hashlib.sha256(json.dumps([stable(v) for v in fields],sort_keys=True).encode()).hexdigest()
    def predict(self,actions):
        if self.previous is None or self.prepared is None:raise RuntimeError("PREDICTOR_OBSERVED_HISTORY_OR_CURRENT_GEOMETRY_MISSING")
        before=self.history_signature();start=time.perf_counter()
        cur=self.current;pr=self.previous
        self.last_input=dict(current=copy.deepcopy(cur),previous=copy.deepcopy(pr),
          elapsed_s=self.clock-self.previous_clock,retained_goal=None if self.goal is None else self.goal.copy(),
          last_goal_step=self.last_goal_step,executed_steps=self.executed_steps,
          candidate_actions=np.asarray(actions,float).copy())
        out=predict_response(cur["robot0_eef_pos"],cur["robot0_eef_quat"],cur["robot0_gripper_qpos"],
          pr["robot0_eef_pos"],pr["robot0_eef_quat"],self.clock-self.previous_clock,
          actions,self.goal)
        result=simplified_terminal_steps(self.prepared,self.links,cur["robot0_eef_pos"],cur["robot0_eef_quat"],out["steps"])
        # In particular do not commit out["retained_orientation_goal"].
        if self.history_signature()!=before:raise RuntimeError("PREDICTOR_PREDICTION_MUTATED_EXECUTED_HISTORY")
        self.predict_calls+=1;self.last_latency_ms=(time.perf_counter()-start)*1000
        return result

def declared_terminal_spec(yaml_path):
    import yaml
    from pathlib import Path
    blob=Path(yaml_path).read_bytes()
    def find(n):
        if isinstance(n,dict):
            if isinstance(n.get("collision_spheres"),dict):return n["collision_spheres"]
            for x in n.values():
                r=find(x)
                if r is not None:return r
    smap=find(yaml.safe_load(blob))
    selected={"panda_link6","panda_link7","panda_hand","panda_leftfinger","panda_rightfinger"}
    links=[];radii=[]
    for name,entries in smap.items():
        if name in selected:
            for e in entries:links.append(name);radii.append(float(e["radius"]))
    if set(links)!=selected:raise RuntimeError("PREDICTOR_DECLARED_LINKS_INCOMPLETE")
    return links,radii

def install_native_block(module):
    """Only in candidate process; lazy parent wrapper must never construct impl."""
    calls=[]
    def prohibited(*args,**kwargs):
        calls.append("native_rollout_attempt")
        raise RuntimeError("PREDICTOR_NATIVE_ROLLOUT_FORBIDDEN")
    module.RolloutPredictor.__init__=prohibited
    module.RolloutPredictor.predict=prohibited
    return calls
