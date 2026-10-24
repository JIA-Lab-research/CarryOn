#!/usr/bin/env python3
"""Track manipulation phases from gripper width and executed gripper commands."""

PRE_GRASP = "PRE_GRASP"
CARRY = "CARRY"
RELEASE_SETTLE = "RELEASE_SETTLE"


class PhaseLatch:
    def __init__(self, gripper_closed_width, gripper_open_is_negative=True):
        # gripper_closed_width: existing GRIPPER_CLOSED_WIDTH from the adapter.
        # gripper_open_is_negative: action gripper-command sign convention for
        # "open/release" (pi05 LIBERO: -1 open, +1 close -> open is negative).
        self._closed_w = float(gripper_closed_width)
        self._open_neg = bool(gripper_open_is_negative)
        self.state = PRE_GRASP
        self.was_holding = False
        # A1: RELEASE_SETTLE records that an open COMMAND was issued; `released`
        # records that the MEASURED width has confirmed the object actually left
        # the fingers. Monotonic like the states above, never cleared.
        self.released = False

    def update(self, gripper_width, action_gripper_command):
        """Advance the latch one control step and return the new latch state.
        gripper_width: |finger1 - finger2| (proprioception).
        action_gripper_command: the executed action's gripper dim."""
        holding = float(gripper_width) < self._closed_w
        stable_holding = holding and self.was_holding

        if self.state == PRE_GRASP:
            if stable_holding:
                self.state = CARRY
        elif self.state == CARRY:
            released = (action_gripper_command < 0) if self._open_neg \
                else (action_gripper_command > 0)
            if released:
                self.state = RELEASE_SETTLE
        # RELEASE_SETTLE is terminal.

        # A1: the open command and the width opening need not fall on the
        # same decision, so this is re-checked on every step while the latch
        # sits in RELEASE_SETTLE.
        if self.state == RELEASE_SETTLE and float(gripper_width) > self._closed_w:
            self.released = True
        self.was_holding = holding
        return self.state

    def regime(self):
        """Collapse the latch to the contact-graph regime."""
        return PRE_GRASP if self.state == PRE_GRASP else CARRY
