"""Fixed critically damped end-effector response from executed action history."""
import numpy as np
VERSION = "PREDICTOR_EE_RESPONSE_v1"
KP = 150.0
DAMPING_RATIO = 1.0
DT = 0.05
XYZ_SCALE = 0.05
ROT_SCALE = 0.5

def vec(x, n):
    a = np.asarray(x, dtype=float).copy()
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError("finite vector required")
    return a

def unit(q):
    q = vec(q, 4)
    if np.linalg.norm(q) <= np.finfo(float).eps:
        raise ValueError("zero quaternion")
    return q / np.linalg.norm(q)

def mul(a, b):
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return np.r_[aw*bv + bw*av + np.cross(av,bv), aw*bw-np.dot(av,bv)]

def inv(q):
    return np.r_[-q[:3], q[3]]

def exp(v):
    v = vec(v, 3)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return unit(np.r_[v/2,1.0])
    return np.r_[v*np.sin(n/2)/n,np.cos(n/2)]

def log(q):
    q = unit(q)
    if q[3] < 0:
        q = -q
    n = np.linalg.norm(q[:3])
    if n < 1e-10:
        return 2*q[:3]
    return q[:3]*(2*np.arctan2(n,q[3])/n)

def mat(q):
    q = unit(q)
    x,y,z,w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def left_jacobian(v):
    x,y,z = v
    A = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    a = np.linalg.norm(v)
    if a < 1e-6:
        return np.eye(3)+(0.5-a*a/24)*A+(1/6-a*a/120)*(A@A)
    return np.eye(3)+(1-np.cos(a))/(a*a)*A+(a-np.sin(a))/(a**3)*(A@A)

def critical_response(position, velocity, goal, dt=DT):
    """Exact x''=KP*(goal-x)-2*sqrt(KP)*x' for fixed goal."""
    x,v,g = np.asarray(position,float),np.asarray(velocity,float),np.asarray(goal,float)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive interval required")
    lam = np.sqrt(KP)
    e = x-g
    b = v+lam*e
    decay = np.exp(-lam*dt)
    return g+(e+b*dt)*decay, (v-lam*b*dt)*decay

def measured_velocity(previous_position, previous_quaternion, position, quaternion, elapsed_s):
    if not np.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("measured elapsed interval required")
    v = (vec(position,3)-vec(previous_position,3))/elapsed_s
    w = log(mul(unit(quaternion), inv(unit(previous_quaternion))))/elapsed_s
    return v,w

def goal_from_sent_action(observed_before_quaternion, adapted_sent_action):
    """Return replacement world orientation goal, or None for exact zero rot.
    The caller must retain an earlier goal when None is returned.
    """
    a = vec(adapted_sent_action,7)
    d = ROT_SCALE*np.clip(a[3:6],-1,1)
    return unit(mul(exp(d),unit(observed_before_quaternion))) if np.any(d != 0) else None

def predict_response(position_m, quaternion_xyzw, gripper_qpos,
                     previous_position_m, previous_quaternion_xyzw, elapsed_s,
                     adapted_actions, retained_orientation_goal_xyzw=None):
    """Finite 50ms response for each action, left/world SO(3), F1 hand quat.
    Position is measured control-site origin; constant site/hand orientation
    offset cancels under world-left operations. No world-axis compensation.
    History is measured, elapsed_s is the observation interval, not wall delay.
    Exact zero rotation retains goal; missing goal is an explicit error unless
    a nonzero candidate command establishes it before use.
    Rotation is a local log-coordinate PD approximation; J_left converts its
    final coordinate derivative to world angular velocity for the next step.
    Fixed current finger width; gripper commands do not predict attachment.
    """
    p,q,g = vec(position_m,3),unit(quaternion_xyzw),vec(gripper_qpos,2)
    v,w = measured_velocity(previous_position_m,previous_quaternion_xyzw,p,q,elapsed_s)
    actions = np.asarray(adapted_actions,float).copy()
    if actions.shape != (5,7) or not np.isfinite(actions).all():
        raise ValueError("finite 5x7 already-adapted actions required")
    goal = None if retained_orientation_goal_xyzw is None else unit(retained_orientation_goal_xyzw)
    steps=[]
    for k,a in enumerate(actions):
        p_goal = p+XYZ_SCALE*np.clip(a[:3],-1,1)
        replacement = goal_from_sent_action(q,a)
        if replacement is not None:
            goal = replacement
        if goal is None:
            raise ValueError("orientation goal history unavailable")
        rot_goal = log(mul(goal,inv(q)))
        dtheta,dtheta_dot = critical_response(np.zeros(3),w,rot_goal)
        p,v = critical_response(p,v,p_goal)
        q = unit(mul(exp(dtheta),q))
        w = left_jacobian(dtheta)@dtheta_dot
        steps.append(dict(k=k,ee_pos_k=p.copy(),ee_quat_k=q.copy(),
                          linear_velocity=v.copy(),angular_velocity=w.copy(),
                          gripper_command=float(a[6]),gripper_qpos_k=g.copy()))
    return dict(version=VERSION,steps=steps,current_gripper_qpos=g.copy(),
                retained_orientation_goal=goal.copy(),
                coverage="EE finite response; fingers frozen, no future arm kinematics")
