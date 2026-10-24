#!/usr/bin/env python3
"""Build target, source-support, destination and residual scene point groups."""
import numpy as np

import relational_grounding as RG

SELF_DEPTH_TOL_M = 0.03   # |observed - expected robot depth| within this = self pixel

# Outward margin of the local window used to label SOURCE_SUPPORT.
# This bounds the support region beneath the target.
SOURCE_LOCAL_WINDOW_MARGIN_M = 0.05

# Maximum number of instances retained per noun.
TARGET_CANDIDATES_PER_NOUN_CEILING = 6
TARGET_CANDIDATES_UNION_CEILING = 12
# B" geometric identity gate. A TARGET mask this close to the DESTINATION
# mask IS the destination, whatever noun asked for it. Certified offline at
# 9/9 (b3cbda2a) with a trip side of 0.9933-0.9969 and a keep side whose max
# is 0.2731 (a small cargo legitimately inside a large destination), so the
# separation margin is 0.7202 and this threshold sits in open space.
# The quantity is MASK IoU; a point-set IoU would not be the certified one.
IDENTITY_GATE_IOU = 0.80



BASE_LINK0_STATIC_AABB_MIN_REL_FIRST_FK_M = np.array([-0.09, -0.15, -0.09])
BASE_LINK0_STATIC_AABB_MAX_REL_FIRST_FK_M = np.array([0.09, 0.08, 0.16])


def base_link0_static_self_mask(depth_disp, w2p, base_anchor, H, W):
    """Delete observed RGB-D points inside the fixed robot0_link0/base volume."""
    z = np.asarray(depth_disp, float)
    if z.shape[:2] != (H, W):
        raise ValueError("BASE_LINK0_DEPTH_SHAPE:%r!=%r" % (z.shape[:2], (H, W)))
    anchor = np.asarray(base_anchor, float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("BASE_LINK0_ANCHOR_INVALID")
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    out = np.zeros((H, W), bool)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return out
    zz = z[ys, xs]
    cam = np.stack([xs * zz, ys * zz, zz, np.ones_like(zz)], axis=1)
    c2w = np.linalg.inv(np.asarray(w2p, float))
    world = (c2w @ cam.T).T[:, :3]
    rel = world - anchor[None, :]
    inside = ((rel >= BASE_LINK0_STATIC_AABB_MIN_REL_FIRST_FK_M) &
              (rel <= BASE_LINK0_STATIC_AABB_MAX_REL_FIRST_FK_M)).all(axis=1)
    out[ys[inside], xs[inside]] = True
    return out


def _project(w2p, pts):
    h = (w2p @ np.c_[pts, np.ones(len(pts))].T).T
    z = h[:, 2:3]
    z[np.abs(z) < 1e-9] = 1e-9
    uv = h[:, :2] / z
    return uv  # (u=col, v=row) in display orientation


def fk_self_mask(fk_sphere_centers, fk_sphere_radii, depth_disp, w2p, H, W):
    """Per-pixel FK expected-depth self mask: project robot FK spheres to the
    image; a pixel is self if the observed metric depth matches the nearest
    projected sphere-surface depth within tolerance."""
    mask = np.zeros((H, W), bool)
    if len(fk_sphere_centers) == 0:
        return mask
    uv = _project(w2p, np.asarray(fk_sphere_centers))
    # expected depth per sphere = distance from camera center to sphere surface
    cam_center = np.linalg.inv(w2p)[:3, 3] if w2p.shape == (4, 4) else None
    for (cu, cv), c, r in zip(uv, fk_sphere_centers, fk_sphere_radii):
        u, v = int(round(cu)), int(round(cv))
        rad_px = max(2, int(round(r * 900)))  # Coarse footprint.
        u0, u1 = max(0, u - rad_px), min(W, u + rad_px + 1)
        v0, v1 = max(0, v - rad_px), min(H, v + rad_px + 1)
        if u0 >= u1 or v0 >= v1:
            continue
        # expected surface depth: distance camera->center minus radius (coarse)
        if cam_center is not None:
            exp = float(np.linalg.norm(np.asarray(c) - cam_center) - r)
        else:
            exp = float(depth_disp[v0:v1, u0:u1].mean())
        patch = depth_disp[v0:v1, u0:u1]
        mask[v0:v1, u0:u1] |= np.abs(patch - exp) <= SELF_DEPTH_TOL_M
    return mask


def geometric_source_mask(instance_masks, depth_disp, c2w, exclude_mask=None,
                          margin_m=SOURCE_LOCAL_WINDOW_MARGIN_M):
    """Derive source support directly beneath the selected target instances.

    For each instance, expand its xy footprint by margin_m and retain points below
    its lowest z within RELATIONAL_ABOVE_MARGIN_M. Union the resulting support
    patches, excluding target pixels and the destination mask."""
    z = np.asarray(depth_disp, float)
    H, W = z.shape[:2]
    out = np.zeros((H, W), bool)
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    vv, uu = np.mgrid[0:H, 0:W]
    ys, xs = vv[valid], uu[valid]
    if len(xs) == 0:
        return out
    zz = z[ys, xs]
    cam = np.stack([xs * zz, ys * zz, zz, np.ones_like(zz)], axis=1)
    world = (np.asarray(c2w) @ cam.T).T[:, :3]
    sel = np.zeros(len(world), bool)
    for m in instance_masks:
        inst = RG.mask_world(m, depth_disp, c2w)
        if inst is None or len(inst) == 0:
            continue
        lo = inst[:, :2].min(0) - float(margin_m)
        hi = inst[:, :2].max(0) + float(margin_m)
        ceil_z = float(inst[:, 2].min()) + RG.RELATIONAL_ABOVE_MARGIN_M
        sel |= (((world[:, :2] >= lo) & (world[:, :2] <= hi)).all(1)
                & (world[:, 2] <= ceil_z))
    out[ys[sel], xs[sel]] = True
    for m in instance_masks:
        out &= ~np.asarray(m, bool)
    if exclude_mask is not None:
        out &= ~np.asarray(exclude_mask, bool)
    return out


def held_instance_index(target_instances, ee_xyz):
    """Which plural instance the gripper is holding: the one whose nearest point
    is closest to the end effector. A threshold-free arg-min -- it introduces no
    constant and so cannot silently widen an exemption. Returns None when there
    is no non-empty instance."""
    best_i, best_d = None, float("inf")
    for i, p in enumerate(target_instances):
        p = np.asarray(p, float)
        if len(p) == 0:
            continue
        d = float(np.min(np.linalg.norm(p - np.asarray(ee_xyz, float), axis=1)))
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def split_carry_target(target_instances, held_index, ee_now, init_ee):

    off = np.asarray(ee_now, float) - np.asarray(init_ee, float)
    held = np.empty((0, 3))
    rest = []
    for i, p in enumerate(target_instances):
        p = np.asarray(p, float)
        if i == held_index:
            held = p + off
        elif len(p):
            rest.append(p)
    return held, (np.vstack(rest) if rest else np.empty((0, 3)))


def carry_groups(target_instances, held_index, ee_now, init_ee):
    """Label the held instance HELD_TARGET and retain TARGET on the other instances.

    The held instance follows swept gripper geometry and uses the destination
    contact exemption."""
    held, remaining = split_carry_target(target_instances, held_index, ee_now, init_ee)
    return {"HELD_TARGET": held, "TARGET": remaining}


def tracking_mask_world(mask, depth_disp, c2w):

    return RG.mask_world(np.asarray(mask, bool), depth_disp, c2w)


def tracking_carry_groups(candidate_records, held_candidate_id):

    held = np.empty((0, 3))
    remaining = []
    seen = set()
    for record in candidate_records or []:
        stable_id = int(record["stable_candidate_id"])
        if stable_id in seen:
            raise ValueError("duplicate stable_candidate_id in tracker snapshot")
        seen.add(stable_id)
        points = np.asarray(record["world_points"], float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError("tracker candidate has invalid world points")
        if stable_id == held_candidate_id:
            held = points
        else:
            remaining.append(points)
    if held_candidate_id is None or not len(held):
        raise ValueError("motion-emergent held candidate is absent from snapshot")
    target = np.vstack(remaining) if remaining else np.empty((0, 3))
    return {"HELD_TARGET": held, "TARGET": target}



def _destination_error_class(value):
    """Sanitized, bounded failure class only; no prompts, paths or secrets."""
    if value is None:
        return None
    text = str(value).strip()
    return (text.split(":", 1)[0] or "UNKNOWN")[:80]


def _molmo_point_fallback(rgb, phrase, segmenter, point_pointer,
                          pointer_unavailable_reason=None):
    """Ground a noun with Molmo pointing and the SAM3 point interface.

    Propagate failures explicitly; an empty mask is not a successful result."""
    rec = {
        "attempted": True,
        "policy": "MOLMO_POINT_THEN_SAM3_POINT_API",
        "phrase_source": "FROZEN_ROLE_NOUN_EXACT",
        "molmo_calls": 0,
        "sam3_point_calls": 0,
        "point_px": None,
        "mask_px": 0,
        "molmo_error_class": None,
        "sam3_point_error_class": None,
        "outcome": None,
        "declared_fail_closed": True,
    }
    if point_pointer is None:
        rec["outcome"] = "FAIL_POINTER_UNAVAILABLE"
        rec["molmo_error_class"] = _destination_error_class(
            pointer_unavailable_reason or "POINTER_UNAVAILABLE")
        return None, rec
    rec["molmo_calls"] = 1
    try:
        point = point_pointer.point_at(rgb, str(phrase))
    except Exception as exc:
        rec["outcome"] = "FAIL_MOLMO_EXCEPTION"
        rec["molmo_error_class"] = type(exc).__name__
        return None, rec
    if point is None:
        rec["outcome"] = "FAIL_MOLMO_NO_POINT"
        rec["molmo_error_class"] = _destination_error_class(
            getattr(point_pointer, "last_error", None) or "NO_POINT")
        return None, rec
    rec["point_px"] = [int(point[0]), int(point[1])]
    rec["sam3_point_calls"] = 1
    try:
        mask = segmenter.segment_at_point(rgb, point_px=point)
    except Exception as exc:
        rec["outcome"] = "FAIL_SAM3_POINT_EXCEPTION"
        rec["sam3_point_error_class"] = type(exc).__name__
        return None, rec
    if mask is None or not np.asarray(mask, bool).any():
        rec["outcome"] = "FAIL_SAM3_POINT_NO_MASK"
        rec["sam3_point_error_class"] = _destination_error_class(
            getattr(segmenter, "last_error", None) or "NO_MASK")
        return None, rec
    mask = np.asarray(mask, bool)
    rec["mask_px"] = int(mask.sum())
    rec["outcome"] = "MOLMO_POINT_SUCCESS"
    rec["declared_fail_closed"] = False
    return mask, rec


def _destination_text_success():
    return {
        "attempted": False,
        "policy": "SAM3_TEXT_SUCCESS_NO_POINT_ATTEMPTED",
        "phrase_source": "FROZEN_ROLE_NOUN_EXACT",
        "molmo_calls": 0,
        "sam3_point_calls": 0,
        "point_px": None,
        "mask_px": 0,
        "molmo_error_class": None,
        "sam3_point_error_class": None,
        "outcome": "TEXT_SUCCESS",
        "declared_fail_closed": False,
    }

def build_labels(rgb, depth_disp, c2w, w2p, proprioception, roles_payload, segmenter,
                 fk_spheres_fn, *, cand_threshold=0.15, use_self_mask=True,
                 self_mask_fn=None, point_pointer=None,
                 pointer_unavailable_reason=None):

    H, W = depth_disp.shape[:2]
    try:
        roles, plural = roles_payload
    except Exception:
        roles, plural = None, False
    if not roles or not all(roles.get(k) for k in ("TARGET", "DESTINATION")):
        return {"status": "FAIL_roles_unavailable", "sam3_calls": 0}
    raw_targets = roles["TARGET"]
    # Accept a scalar role value as well as the list-of-strings representation.
    if isinstance(raw_targets, str) and raw_targets.strip():
        target_nouns = [raw_targets.strip()]
    elif (isinstance(raw_targets, list) and raw_targets and
          all(isinstance(v, str) and v.strip() for v in raw_targets)):
        target_nouns = [v.strip() for v in raw_targets]
    else:
        return {"status": "FAIL_roles_unavailable", "sam3_calls": 0}
    if len(set(target_nouns)) != len(target_nouns):
        return {"status": "FAIL_roles_unavailable", "sam3_calls": 0}
    # Derive SOURCE_SUPPORT from target geometry. No source phrase is segmented.
    dst_mask = segmenter.segment(rgb, prompt=roles["DESTINATION"])
    sam3_calls, molmo_calls = 1, 0
    # DESTINATION grounding provenance, captured HERE -- immediately after its
    # own request and BEFORE the TARGET batch. Sam3Segmenter._run resets and
    # rewrites last_prompt_* on EVERY call, so reading these fields after
    # build_labels returns would record the TARGET's prompt under a DESTINATION
    # name. Same bar as the A2 amendment: a recorder that lies quietly is worse
    # than no recorder.
    dest_prov = (getattr(segmenter, "last_prompt_used", None),
                 getattr(segmenter, "last_prompt_was_alternate", None),
                 getattr(segmenter, "last_prompt_px", None))
    # Capture destination metadata before the target request overwrites segmenter state.
    # The reason describes only the last fallback phrase; it distinguishes no detections
    # from detections rejected by the score threshold for that phrase.
    dest_fail_reason = (getattr(segmenter, "last_result", None) or {}).get("reason")
    dest_text_fail_reason = dest_fail_reason
    dest_fallback = _destination_text_success()
    if dst_mask is None or not np.asarray(dst_mask, bool).any():
        dst_mask, dest_fallback = _molmo_point_fallback(
            rgb, roles["DESTINATION"], segmenter, point_pointer,
            pointer_unavailable_reason)
        sam3_calls += int(dest_fallback["sam3_point_calls"])
        molmo_calls += int(dest_fallback["molmo_calls"])
        dest_fail_reason = (None if dst_mask is not None else
                            "DESTINATION_" + str(dest_fallback["outcome"]))
    else:
        dest_fail_reason = None
    dest_fallback["text_failure_reason"] = _destination_error_class(
        dest_text_fail_reason)
    # Compute destination fields before target-side early returns so every path records them.
    dest_degraded = dst_mask is None or not np.asarray(dst_mask, bool).any()
    dest_mask = (np.zeros((H, W), bool) if dest_degraded
                 else np.asarray(dst_mask, bool))
    segmenter._cb3_initial_destination = (dest_mask.copy(), dict(dest_fallback))
    n_dest_px = int(dest_mask.sum())
    dest_record_fields = {"dest_degraded": bool(dest_degraded),
                          "degraded_reason": ("dest_empty" if dest_degraded else None),
                          "n_dest_px": n_dest_px,
                          "dest_prompt_used": dest_prov[0],
                          "dest_prompt_was_alternate": dest_prov[1],
                          "dest_prompt_px": dest_prov[2],
                          "dest_fail_reason": dest_fail_reason}







    candidate_entries, candidate_records = [], {}
    for target_index, target_noun in enumerate(target_nouns):









        sam3_calls += 1
        raw_masks = [np.asarray(c, bool) for c in
                     (segmenter.segment_candidates(rgb, prompt=target_noun,
                                                   threshold=cand_threshold) or [])
                     if np.asarray(c, bool).sum() > 0]
        if not raw_masks:
            # text returned nothing: the ONLY path to the pointer.
            point_mask, target_fallback = _molmo_point_fallback(
                rgb, target_noun, segmenter, point_pointer,
                pointer_unavailable_reason)
            sam3_calls += int(target_fallback["sam3_point_calls"])
            molmo_calls += int(target_fallback["molmo_calls"])
            if point_mask is not None:
                raw_masks = [np.asarray(point_mask, bool)]
        else:
            target_fallback = _destination_text_success()
            target_fallback["policy"] = "SAM3_TEXT_FIRST_NO_POINT_ATTEMPTED"
            target_fallback["text_first_px"] = int(raw_masks[0].sum())
        raw_px = [int(mask.sum()) for mask in raw_masks]
        retained_masks = raw_masks[:TARGET_CANDIDATES_PER_NOUN_CEILING]
        retained_px = raw_px[:TARGET_CANDIDATES_PER_NOUN_CEILING]
        dropped_px = raw_px[TARGET_CANDIDATES_PER_NOUN_CEILING:]
        candidate_records[target_noun] = {
            "target_index": target_index,

            "count": len(raw_masks),
            "candidate_px": raw_px,

            "raw_count": len(raw_masks),
            "retained_count": len(retained_masks),
            "dropped_count": len(raw_masks) - len(retained_masks),
            "raw_candidate_px": raw_px,
            "retained_candidate_px": retained_px,
            "dropped_candidate_px": dropped_px,
            "cap_applied": len(raw_masks) > len(retained_masks),
            "grounding_fallback": target_fallback,
        }
        for candidate_index, mask in enumerate(retained_masks):
            candidate_entries.append({
                "target_index": target_index,
                "target_noun": target_noun,
                "candidate_index": candidate_index,
                "mask": mask,
            })
    union_retained = candidate_entries[:TARGET_CANDIDATES_UNION_CEILING]
    union_dropped = candidate_entries[TARGET_CANDIDATES_UNION_CEILING:]
    cand_masks = [entry["mask"] for entry in union_retained]
    def _candidate_provenance(entry):
        return {key: entry[key] for key in
                ("target_index", "target_noun", "candidate_index")}
    target_candidate_union = {
        "raw_after_per_noun_count": len(candidate_entries),
        "retained_count": len(union_retained),
        "dropped_count": len(union_dropped),
        "retained_provenance": [_candidate_provenance(v) for v in union_retained],
        "dropped_provenance": [_candidate_provenance(v) for v in union_dropped],
        "cap_applied": bool(union_dropped),
    }
    grounding_fallback = {
        "policy": ("TARGET_SAM3_TEXT_FIRST_THEN_MOLMO_POINT_ON_EMPTY_THEN_IDENTITY_GATE_DROP"
                   "__DEST_SAM3_TEXT_THEN_MOLMO_POINT__THEN_FAIL_CLOSED"),
        "destination": dest_fallback,
        "targets": {noun: candidate_records[noun]["grounding_fallback"]
                    for noun in target_nouns},
        "declared_fail_closed": bool(
            dest_fallback["declared_fail_closed"] or
            any(candidate_records[n]["grounding_fallback"]["declared_fail_closed"]
                for n in target_nouns)),
    }























    identity_iou = []
    if dest_mask.any():
        for _cand in cand_masks:
            _c = np.asarray(_cand, bool)
            _union = int((_c | dest_mask).sum())
            identity_iou.append(int((_c & dest_mask).sum()) / _union
                                if _union else 0.0)
    _dropped_idx = [i for i, v in enumerate(identity_iou) if v >= IDENTITY_GATE_IOU]
    _kept_idx = [i for i in range(len(cand_masks)) if i not in _dropped_idx]
    identity_collision = bool(_dropped_idx) and not _kept_idx
    grounding_fallback["identity_gate"] = {
        "quantity": "mask_iou",
        "threshold": IDENTITY_GATE_IOU,
        "per_candidate_iou": [round(v, 4) for v in identity_iou],
        "max_iou": (round(max(identity_iou), 4) if identity_iou else None),
        "n_candidates": len(cand_masks),
        "dest_px": int(dest_mask.sum()),
        "dropped": [{"candidate": i,
                     "px": int(np.asarray(cand_masks[i], bool).sum()),
                     "iou": round(identity_iou[i], 4)} for i in _dropped_idx],
        "n_remaining": len(_kept_idx),
        "tripped": bool(identity_collision),
    }
    if _dropped_idx and _kept_idx:
        cand_masks = [cand_masks[i] for i in _kept_idx]
    if identity_collision:
        out = {"status": "FAIL_grounding_IDENTITY_COLLISION_FAIL_CLOSED",
               "roles": roles,
               "n_target_candidates": candidate_records,
               "target_candidate_union": target_candidate_union,
               "sam3_calls": sam3_calls, "molmo_calls": molmo_calls,
               "grounding_fallback": grounding_fallback}
        out.update(dest_record_fields)
        return out
    surv, status = RG.relational_target_set(cand_masks, None, depth_disp, c2w, plural)
    if not status.startswith("OK"):
        # Record the target candidate count and destination state on the failure path.
        out = {"status": "FAIL_grounding_" + status, "roles": roles,
               "n_target_candidates": candidate_records,
               "target_candidate_union": target_candidate_union,
               "sam3_calls": sam3_calls, "molmo_calls": molmo_calls,
               "grounding_fallback": grounding_fallback}
        out.update(dest_record_fields)
        return out
    # Keep an empty DESTINATION group when destination grounding fails.
    # Without destination points, no destination claim box is created and residual
    # points remain OTHER. Record the degraded outcome for the episode.
    instance_masks = [cand_masks[i] for i in surv]
    target_mask = np.zeros((H, W), bool)
    for m in instance_masks:
        target_mask |= m
    # geometric SOURCE: derived AFTER the target, bounded by the local window,
    # destination excluded so a place surface is never absorbed into the patch.
    source_mask = geometric_source_mask(instance_masks, depth_disp, c2w,
                                        exclude_mask=dest_mask)




    if not use_self_mask:
        self_mask = np.zeros((H, W), bool)
    elif self_mask_fn is not None:
        self_mask = np.asarray(
            self_mask_fn(proprioception, depth_disp, w2p, H, W), dtype=bool
        )
        if self_mask.shape != (H, W):
            raise ValueError(
                "SELF_MASK_INITIAL_SELF_MASK_SHAPE:%r!=%r" %
                (self_mask.shape, (H, W)))
    else:
        spheres = fk_spheres_fn(proprioception)
        centers = [np.asarray(c, float) for c, r in spheres]
        radii = [float(r) for c, r in spheres]
        self_mask = fk_self_mask(centers, radii, depth_disp, w2p, H, W)
    # --- label the cloud ---------------------------------------------------
    vv, uu = np.mgrid[0:H, 0:W]
    z = np.asarray(depth_disp, float)
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    out = {"status": "OK", "roles": roles, "plural": plural, "survivors": surv,
           "n_self": int((self_mask & valid).sum()),
           "n_source_px": int(source_mask.sum()),
           "source_window_margin_m": SOURCE_LOCAL_WINDOW_MARGIN_M,
           # Record candidate count on successful grounding paths, including unions.
           "n_target_candidates": candidate_records,
           "target_candidate_union": target_candidate_union,
           "sam3_calls": sam3_calls, "molmo_calls": molmo_calls,
           "grounding_fallback": grounding_fallback}
    out.update(dest_record_fields)
    # Keep per-instance TARGET point sets for the held/remaining split.
    # Apply the same validity, self-removal and priority rules as the group labels.
    inst_groups, inst_assigned = [], self_mask.copy()
    for m in instance_masks:
        sel_i = valid & m & (~inst_assigned)
        inst_assigned |= m
        ys_i, xs_i = np.where(sel_i)
        if len(xs_i):
            zi = z[ys_i, xs_i]
            cami = np.stack([xs_i * zi, ys_i * zi, zi, np.ones_like(zi)], axis=1)
            inst_groups.append((np.asarray(c2w) @ cami.T).T[:, :3])
        else:
            inst_groups.append(np.empty((0, 3)))
    out["target_instances"] = inst_groups
    groups = {}
    # priority: ROBOT deleted first, then TARGET, SOURCE, DEST, then OTHER
    assigned = self_mask.copy()
    for name, m in (("TARGET", target_mask), ("SOURCE_SUPPORT", source_mask),
                    ("DESTINATION", dest_mask)):
        sel = valid & m & (~assigned)
        assigned |= m
        ys, xs = np.where(sel)
        if len(xs):
            zz = z[ys, xs]
            cam = np.stack([xs * zz, ys * zz, zz, np.ones_like(zz)], axis=1)
            groups[name] = (np.asarray(c2w) @ cam.T).T[:, :3]
        else:
            groups[name] = np.empty((0, 3))
    other_sel = valid & (~assigned)
    ys, xs = np.where(other_sel)
    zz = z[ys, xs]
    cam = np.stack([xs * zz, ys * zz, zz, np.ones_like(zz)], axis=1)
    groups["OTHER"] = (np.asarray(c2w) @ cam.T).T[:, :3]
    out["groups"] = groups
    return out
