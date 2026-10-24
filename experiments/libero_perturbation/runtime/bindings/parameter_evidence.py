"""Attach same-attempt evidence to parameter reflection."""
import json
from pathlib import Path
import hashlib
import parameter_revision as prior
from carryon.recovery import jam_z_reflection as frozen
SYSTEM = 'Reflect only on an unresolved grasp_no_hold or descent_jam physical attempt. Choose NO_UPDATE or UPDATE_Z using the supplied same-attempt evidence. NO_UPDATE retains the current configuration and existing memory. UPDATE_Z proposes only a future absolute grasp contact/TCP target z in metres, not hand/flange height or a lift endpoint. Missing evidence is unknown. A failed hold, visible point cloud or previous memory alone is not proof of a z error. Return strict JSON only; no other parameter changes.'
OriginalClient = frozen.OpenRouterJamZClient

class EvidenceClient(OriginalClient):

    async def _complete_async(self, prompt, envelope):
        from carryon.providers.openrouter import OpenRouterChatClient
        body = json.loads(json.dumps(envelope))
        if body != frozen.OPENROUTER_ENVELOPE:
            raise ValueError('grasp-height reflector OpenRouter envelope mismatch')
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        client = self._client or OpenRouterChatClient(api_key=self.api_key, base_url=self.base_url, timeout_s=self.timeout_s)
        owns_client = self._client is None
        try:
            chat = getattr(client, 'chat_envelope', None)
            if not callable(chat) or not getattr(chat, '__carryon_parameter_transport_retry__', False):
                raise RuntimeError('parameter transport retry handler is not installed')
            result = await chat(envelope=body, messages=messages)
            if not isinstance(getattr(result, 'text', None), str):
                raise RuntimeError('transport choke point returned non-text result')
            return result.text
        finally:
            if owns_client:
                await client.aclose()

class EvidenceReflector(prior.RevisableJamZReflector):

    def _decision_prompt(self, payload, trigger):
        prompt = super()._decision_prompt(payload, trigger)
        ev = getattr(self, 'attempt_evidence', {'status': 'unknown', 'reason': 'same-attempt capture unavailable'})
        self._append_ledger_row({'event': 'PARAMETER_REQUEST_EVIDENCE', 'trigger': trigger, 'object_key': payload['object_key'], 'evidence': ev, 'evidence_sha256': hashlib.sha256(json.dumps(ev, sort_keys=True).encode()).hexdigest(), 'no_images': True, 'write_succeeded': False}, exclusive=False)
        return prompt + (' SAME_ATTEMPT_EVIDENCE=' + json.dumps(ev, sort_keys=True, separators=(',', ':'), allow_nan=False) + ' Existing memory is an unverified hypothesis, not evidence that its height is effective. Signed residual is actual minus contact target. Post-lift holding widths are not contact heights. Cite the specific evidence or missing evidence in the rationale; do not invent missing measurements.')

def install():
    assert frozen.JamZReflector is prior.FrozenReflector
    assert frozen.OpenRouterJamZClient is OriginalClient
    frozen.JamZReflector = EvidenceReflector
    frozen.OpenRouterJamZClient = EvidenceClient
    return {'writer': __file__, 'original_complete_retained': EvidenceClient.complete is OriginalClient.complete, 'scope': 'isolated candidate; no active shared-tree change'}
