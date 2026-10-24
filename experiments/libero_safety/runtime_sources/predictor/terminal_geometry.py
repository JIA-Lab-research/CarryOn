"""Transform the declared terminal collision spheres along predicted poses."""
import numpy as np
from response_model import mat, vec
DYNAMIC_LINKS = frozenset(("panda_hand","panda_leftfinger","panda_rightfinger"))
STATIC_TERMINAL_LINKS = frozenset(("panda_link6","panda_link7"))

def simplified_terminal_steps(current_spheres, sphere_links, position, quaternion, pose_steps):
    """Preserve all terminal spheres/order/radii/actor. Only hand/fingers move.
    Link names must come from the frozen sphere specification, never coordinates.
    link6/link7 remain in the terminal actor at their current positions.
    Outer caller preserves original arm/wrist static_spheres and all groups.
    """
    if len(current_spheres) != len(sphere_links) or not len(sphere_links):
        raise ValueError("complete declared terminal link identities required")
    if set(sphere_links) - DYNAMIC_LINKS - STATIC_TERMINAL_LINKS:
        raise ValueError("unexpected terminal link")
    p0 = vec(position,3);r0 = mat(quaternion)
    result=[]
    for k,s in enumerate(pose_steps):
        if s["k"] != k: raise ValueError("noncontiguous steps")
        r = mat(s["ee_quat_k"]);p = vec(s["ee_pos_k"],3)
        spheres=[]
        for (c,rad),link in zip(current_spheres,sphere_links):
            c = vec(c,3)
            if link in DYNAMIC_LINKS:
                c = p+r@r0.T@(c-p0)
            spheres.append((c.copy(),float(rad)))
        result.append(dict(k=k,ee_pos_k=p.copy(),ee_quat_k=np.asarray(s["ee_quat_k"]).copy(),spheres_k=spheres))
    return result
