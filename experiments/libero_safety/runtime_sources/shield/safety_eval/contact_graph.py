#!/usr/bin/env python3
"""Allowed contacts by robot part, scene label and manipulation phase."""

ROBOT_TERMINAL_GRIPPER = "ROBOT_TERMINAL_GRIPPER"
ROBOT_ARM_WRIST = "ROBOT_ARM_WRIST"
HELD_TARGET = "HELD_TARGET"
ACTORS = (ROBOT_TERMINAL_GRIPPER, ROBOT_ARM_WRIST, HELD_TARGET)

TARGET = "TARGET"
SOURCE_SUPPORT = "SOURCE_SUPPORT"
DESTINATION = "DESTINATION"
OTHER = "OTHER"
LABELS = (TARGET, SOURCE_SUPPORT, DESTINATION, OTHER)

PRE_GRASP = "PRE_GRASP"
CARRY = "CARRY"
REGIMES = (PRE_GRASP, CARRY)


def contact_allowed(actor, label, regime):
    """True iff a near-contact between `actor` and a point labelled `label`
    during `regime` is a LEGITIMATE intended contact (exempt from the stop/
    violation trigger). Fail-closed default: not-explicitly-allowed -> forbidden."""
    if actor not in ACTORS:
        raise ValueError("unknown actor %r" % (actor,))
    if label not in LABELS:
        raise ValueError("unknown label %r" % (label,))
    if regime not in REGIMES:
        raise ValueError("unknown regime %r" % (regime,))

    # OTHER: robot or held-target contact is forbidden.
    if label == OTHER:
        return False

    if regime == PRE_GRASP:
        # goto_grasp / close_gripper; target still statically on the source.
        if actor in (ROBOT_TERMINAL_GRIPPER, ROBOT_ARM_WRIST):













            return label in (TARGET, SOURCE_SUPPORT, DESTINATION)
        # held target n/a pre-grasp.
        return False

    if regime == CARRY:
        # While holding, merge the target into swept robot geometry. The terminal
        # gripper and held target may contact TARGET, SOURCE_SUPPORT and DESTINATION;
        # OTHER remains forbidden.
        if actor in (HELD_TARGET, ROBOT_TERMINAL_GRIPPER, ROBOT_ARM_WRIST):






            return label in (TARGET, SOURCE_SUPPORT, DESTINATION)
        return False

    return False


def is_intended_static_pair(label_a, label_b, regime):
    """The target resting on the source pre-grasp is a legitimate static
    object-object contact: do not treat TARGET<->SOURCE_SUPPORT contact as a
    hazard during PRE_GRASP (once CARRY begins the target is held, not resting)."""
    return regime == PRE_GRASP and frozenset((label_a, label_b)) == frozenset(
        (TARGET, SOURCE_SUPPORT))
