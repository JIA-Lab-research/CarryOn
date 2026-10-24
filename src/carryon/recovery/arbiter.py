"""Transfer control between the VLA endpoint and the recovery backend.

Takeover cancels the active request into standby; handback resumes execution
through the configured control client."""

from __future__ import annotations

from typing import Optional

from .gello_client import GelloControlClient


class GelloArbiter:
    """Takeover/handback for one recovery invocation on one gello request.

    ``request_id`` is the in-flight VLA request to cancel (the CarryOn job_key
    sent to gello as ``request_id``, e.g. ``plan-...:t1``; not the returned
    ``gello-...`` job_id);
    ``resume_instruction`` is what gello should run after recovery (typically the
    same subtask instruction). The job_id gello returns on resume is exposed as
    :attr:`resumed_job_id` so the caller can re-key its bookkeeping.
    """

    def __init__(
        self,
        gello: GelloControlClient,
        *,
        request_id: str,
        resume_instruction: str,
        callback_url: str = "",
        resume_inputs: Optional[dict] = None,
    ) -> None:
        self.gello = gello
        self.request_id = request_id
        self.resume_instruction = resume_instruction
        self.callback_url = callback_url
        self.resume_inputs = resume_inputs
        self.resumed_job_id: Optional[str] = None
        self.last_takeover_reason: Optional[str] = None
        self.last_handback_reason: Optional[str] = None

    def takeover(self) -> bool:
        try:
            res = self.gello.cancel(self.request_id, return_home=False, standby=True)
        except Exception as e:  # noqa: BLE001
            self.last_takeover_reason = f"{type(e).__name__}: {e}"
            return False
        # Standby is the ONLY signal that matters: gello must actually yield the
        # shared bridge PULL channel, otherwise it keeps streaming hold_q and we
        # get the dual-writer oscillation. A plain ``ok`` is NOT enough -- an
        # unknown/idle request without standby returns ok=True, standby=False and
        # must be treated as a failed takeover. gello latches standby both when it
        # cancels an active task and (G1) when idle, so this stays robust.
        self.last_takeover_reason = getattr(res, "reason", None)
        return bool(getattr(res, "standby", False))

    def handback(self) -> bool:
        try:
            res = self.gello.execute(
                self.resume_instruction,
                request_id=self.request_id,
                skip_reset=True,
                inputs=self.resume_inputs,
                callback_url=self.callback_url,
            )
        except Exception as e:  # noqa: BLE001
            self.last_handback_reason = f"{type(e).__name__}: {e}"
            return False
        self.resumed_job_id = getattr(res, "job_id", None)
        self.last_handback_reason = getattr(res, "reason", None)
        return bool(getattr(res, "accepted", False))
