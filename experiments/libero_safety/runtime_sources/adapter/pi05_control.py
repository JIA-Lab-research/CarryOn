"""pi0.5 controller and image-orientation preprocessing."""
import hashlib
import math
import numpy as np
BATCH2_CADENCE=5
EVADE_STEPS=3
BATCH2_E_MAIN_STEPS=3
BATCH2_E_CONTROL_STEPS=6

class Batch2ActionController:
    """Pure action gate for C=5 and registered nonadditive E in {3, 6}."""

    def __init__(self, cadence=BATCH2_CADENCE, hold_steps=EVADE_STEPS):
        if int(cadence) != 5:
            raise ValueError("batch2 controller requires C=5")
        if int(hold_steps) not in (BATCH2_E_MAIN_STEPS, BATCH2_E_CONTROL_STEPS):
            raise ValueError("batch2 controller requires registered E=3 or E=6")
        self.cadence = int(cadence)
        self.hold_steps = int(hold_steps)
        self.reset()

    def reset(self):
        self.action_plan = []
        self.hold_action = None
        self.hold_action_sha256 = None
        self.hold_left = 0
        self.next_decision_step = None
        self.decision_steps = []
        self.fresh_inference_steps = []
        self.post_hold_replan_pending = False
        self.post_hold_replan_steps = []
        self.override_steps = []
        self.trigger_steps = []
        self.scan_grants = []

    @property
    def in_hold(self):
        return self.hold_left > 0

    def decision_due(self, step):
        step = int(step)
        if self.next_decision_step is None:
            return True
        if step > self.next_decision_step:
            raise RuntimeError("BATCH2_CADENCE_SKIPPED:%d>%d" %
                               (step, self.next_decision_step))
        return step == self.next_decision_step

    def begin_decision(self, step):
        step = int(step)
        if not self.decision_due(step):
            raise RuntimeError("BATCH2_EARLY_DECISION:%d" % step)
        gap = None if not self.decision_steps else step - self.decision_steps[-1]
        if gap is not None and not (1 <= gap <= self.cadence):
            raise RuntimeError("BATCH2_CADENCE_GAP:%d" % gap)
        self.decision_steps.append(step)
        # Deadline only.  The scan result immediately replaces this with the
        # authoritative safe-prefix grant (or start_or_refresh_hold does so).
        self.next_decision_step = step + self.cadence
        return gap

    def commit_scan_grant(self, step, grant_steps):
        step = int(step)
        grant_steps = int(grant_steps)
        if not self.decision_steps or self.decision_steps[-1] != step:
            raise RuntimeError("BATCH2_GRANT_WITHOUT_CURRENT_SCAN:%d" % step)
        if not (1 <= grant_steps <= self.cadence):
            raise RuntimeError("BATCH2_GRANT_OUT_OF_RANGE:%d" % grant_steps)
        self.next_decision_step = step + grant_steps
        self.scan_grants.append({"t": step, "grant_steps": grant_steps})

    def note_fresh_inference(self, step):
        if self.in_hold:
            raise RuntimeError("BATCH2_POLICY_INFERENCE_DURING_HOLD")
        step = int(step)
        self.fresh_inference_steps.append(step)
        if self.post_hold_replan_pending:
            self.post_hold_replan_steps.append(step)
            self.post_hold_replan_pending = False

    def set_policy_plan(self, chunk, *, grant_steps):
        if self.in_hold:
            raise RuntimeError("BATCH2_PLAN_OVERWRITE_DURING_HOLD")
        grant_steps = int(grant_steps)
        if not (1 <= grant_steps <= self.cadence):
            raise RuntimeError("BATCH2_GRANT_OUT_OF_RANGE:%d" % grant_steps)
        rows = list(np.asarray(chunk, dtype=np.float64)[:grant_steps])
        if len(rows) != grant_steps:
            raise RuntimeError("BATCH2_SHORT_POLICY_CHUNK")
        self.action_plan = rows

    def start_or_refresh_hold(self, evade, step):
        self.action_plan.clear()
        # Every k=0 retrigger uses the vector computed from that trigger's
        # current geometry.  Dose remains nonadditive via max(), below.
        self.hold_action = np.asarray(evade, dtype=np.float64).copy()
        self.hold_action_sha256 = hashlib.sha256(
            np.ascontiguousarray(self.hold_action).tobytes()
        ).hexdigest()
        self.hold_left = max(self.hold_left, self.hold_steps)
        self.trigger_steps.append(int(step))
        self.next_decision_step = int(step) + min(self.hold_left, self.cadence)

    def hold_lookahead(self):
        if not self.in_hold or self.hold_action is None:
            raise RuntimeError("BATCH2_NO_HOLD_LOOKAHEAD")
        return np.repeat(self.hold_action[None, :], self.hold_left, axis=0)

    def clear_all(self):
        self.action_plan.clear()
        self.hold_action = None
        self.hold_action_sha256 = None
        self.hold_left = 0

    def action_for_step(self, step):
        step = int(step)
        if self.in_hold:
            # Re-clear on every override step: a concurrent policy write cannot
            # survive to execution even if an upstream caller violates the gate.
            self.action_plan.clear()
            before = self.hold_left
            action = self.hold_action.copy()
            self.hold_left -= 1
            row = {
                "t": step,
                "hold_total": self.hold_steps,
                "hold_index": self.hold_steps - min(before, self.hold_steps) + 1,
                "hold_left_before": before,
                "hold_left_after": self.hold_left,
                "plan_cleared": True,
                "action_sha256": self.hold_action_sha256,
            }
            self.override_steps.append(row)
            if self.hold_left == 0:
                self.hold_action = None
                self.hold_action_sha256 = None
                self.post_hold_replan_pending = True
                # E may be shorter than C. The cleared plan cannot supply the
                # remaining window, so fresh inference is due on the next env
                # step; after that intervention seam, native C=5 resumes.
                self.next_decision_step = step + 1
            return action, row
        if not self.action_plan:
            raise RuntimeError("BATCH2_EMPTY_ACTION_PLAN")
        return np.asarray(self.action_plan.pop(0), dtype=np.float64).copy(), None

def quat2axisangle(quat):
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den
