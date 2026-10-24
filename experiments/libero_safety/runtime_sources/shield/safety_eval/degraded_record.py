#!/usr/bin/env python3
"""Record per-episode destination-grounding outcomes.

Wrap build_labels without changing its returned value. Copy its recorded
fields rather than reading segmenter state after the target request overwrites it."""

FIELDS = ("grounding_status", "dest_degraded", "degraded_reason", "n_dest_px",
          "dest_prompt_used", "dest_prompt_was_alternate", "dest_prompt_px",
          "n_target_candidates", "target_candidate_union",
          # Initialize dispatch_branch for every episode, including early exits.
          # The shield writes the branch at the decision point.
          "dispatch_branch",
          "dest_fail_reason", "grounding_fallback", "molmo_calls")


class DestGroundingRecorder:
    """Callable pass-through around build_labels. Zero control flow."""

    def __init__(self, build_labels_fn):
        self._fn = build_labels_fn
        self.record = dict.fromkeys(FIELDS)

    def reset(self):
        """Clear before every call. Without this a miss would leave the previous
        episode's record in place -- the silent-wrong-record failure this seam
        exists to prevent."""
        self.record = dict.fromkeys(FIELDS)

    def __call__(self, *args, **kwargs):
        self.reset()
        out = self._fn(*args, **kwargs)
        self.record["grounding_status"] = out.get("status")
        self.record["dest_degraded"] = bool(out.get("dest_degraded", False))
        self.record["degraded_reason"] = out.get("degraded_reason")
        self.record["n_dest_px"] = out.get("n_dest_px")
        self.record["dest_prompt_used"] = out.get("dest_prompt_used")
        self.record["dest_prompt_was_alternate"] = out.get("dest_prompt_was_alternate")
        self.record["dest_prompt_px"] = out.get("dest_prompt_px")
        # Without this line the count added in relational_scene is computed and
        # then silently discarded, because this recorder copies a FIXED
        # whitelist instead of passing the dict through. A fix that never
        # reaches the artifact is not a fix.
        self.record["n_target_candidates"] = out.get("n_target_candidates")


        self.record["target_candidate_union"] = out.get("target_candidate_union")
        # Retain the destination failure reason even when no masks were detected.
        self.record["dest_fail_reason"] = out.get("dest_fail_reason")
        self.record["grounding_fallback"] = out.get("grounding_fallback")
        self.record["molmo_calls"] = out.get("molmo_calls")
        return out
