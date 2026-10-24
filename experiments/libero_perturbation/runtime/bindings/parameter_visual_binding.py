"""Retain same-attempt visual evidence and construct parameter-reflection requests.
Install after runtime bindings and before factory construction."""
import base64
import copy
import functools
import hashlib
import json
import time
from pathlib import Path
VISUAL_INSTRUCTION = 'Reaching the commanded TCP pose does not prove that the grasp height was suitable. Use the reference and time-labelled scene images to assess possible failure causes; use the supplied geometry, coordinate frame and telemetry for metric scale. Do not estimate an absolute metre height from pixels alone. Do not force identity, XY, orientation or occlusion problems into a z correction. Evidence may be missing or a perception image may predate a retry: inspect its stated age and phase. Choose NO_UPDATE if the evidence does not support a specific absolute contact-TCP z. NO_UPDATE preserves the active value; it does not clear memory. Cite the supplied frame phases and telemetry in the rationale. '

def digest(data):
    return hashlib.sha256(data).hexdigest()

def image_part(data, mime, metadata):
    assert mime in ('image/png', 'image/jpeg')
    return {'metadata': {**metadata, 'sha256': digest(data), 'bytes': len(data)}, 'content': {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')}}}

def messages_for(prompt, system, images):
    assert len(images) <= 3
    assert sum((x['metadata']['phase'] != 'reference' for x in images)) <= 2
    content = [{'type': 'text', 'text': prompt}]
    for item in images:
        content.append({'type': 'text', 'text': 'FRAME=' + json.dumps(item['metadata'], sort_keys=True)})
        content.append(copy.deepcopy(item['content']))
    return [{'role': 'system', 'content': system + ' ' + VISUAL_INSTRUCTION}, {'role': 'user', 'content': content}]

def install_client():
    import parameter_evidence as accepted
    from parameter_rationale_preference import LengthPreferenceReflector
    from parameter_revision import RevisableJamZReflector
    from carryon.recovery import jam_z_reflection as frozen
    assert frozen.JamZReflector is LengthPreferenceReflector
    assert frozen.OpenRouterJamZClient is accepted.EvidenceClient

    class VisualClient(accepted.OriginalClient):

        async def _complete_async(self, prompt, envelope):
            body = copy.deepcopy(envelope)
            assert body == frozen.OPENROUTER_ENVELOPE, 'sampling envelope changed'
            messages = messages_for(prompt, accepted.SYSTEM, getattr(self, 'visual_images', []))
            client = self._client
            owned = False
            if client is None:
                from carryon.providers.openrouter import OpenRouterChatClient
                client = OpenRouterChatClient(api_key=self.api_key, base_url=self.base_url, timeout_s=self.timeout_s)
                owned = True
            try:
                chat = getattr(client, 'chat_envelope', None)
                if not callable(chat) or not getattr(chat, '__carryon_parameter_transport_retry__', False):
                    raise RuntimeError('Model transport binding unavailable')
                result = await chat(envelope=body, messages=messages)
                if not isinstance(getattr(result, 'text', None), str):
                    raise RuntimeError('transport choke point returned non-text result')
                return result.text
            finally:
                if owned:
                    await client.aclose()

    class VisualReflector(LengthPreferenceReflector):

        def _decision_prompt(self, payload, trigger):
            prompt = RevisableJamZReflector._decision_prompt(self, payload, trigger)
            prompt = prompt.replace('rationale at most 240 characters', 'prefer a concise rationale; length is not an action-validity condition')
            evidence = copy.deepcopy(getattr(self, 'attempt_evidence', None)) or {'status': 'unknown'}
            images = copy.deepcopy(getattr(self, 'attempt_images', []))
            self.client.visual_images = images
            self._append_ledger_row({'event': 'PARAMETER_REQUEST_EVIDENCE', 'object_key': payload['object_key'], 'trigger': trigger, 'evidence': evidence, 'evidence_sha256': digest(json.dumps(evidence, sort_keys=True).encode()), 'no_images': not images, 'image_count': len(images), 'write_succeeded': False, 'image_manifest': [x['metadata'] for x in images]}, exclusive=False)
            return prompt + ' SAME_ATTEMPT_EVIDENCE=' + json.dumps(evidence, sort_keys=True) + ' ' + VISUAL_INSTRUCTION
    frozen.OpenRouterJamZClient = VisualClient
    frozen.JamZReflector = VisualReflector
    return (VisualClient, VisualReflector)

def install_passive_evidence(row_root):
    import carryon.recovery as recovery
    import carryon.recovery.backend as backend_module
    import carryon.recovery.orchestrator as orchestrator
    import parameter_attempt_evidence as capture
    root = Path(row_root) / 'parameter_visual'
    root.mkdir(parents=True, exist_ok=False)
    from critical_frame_capture import install as install_critical
    install_critical(backend_module.GelloRecoveryBackend, row_root)
    records = {}
    registry_class = recovery.EpisodeRecorderRegistry
    old_build = registry_class.build

    @functools.wraps(old_build)
    def build(self, job_key, *args, **kwargs):
        record = old_build(self, job_key, *args, **kwargs)
        if record is not None:
            records[id(record)] = (self, job_key)
        return record
    registry_class.build = build
    old_reference = orchestrator.RecoveryOrchestrator._load_scene_reference

    @functools.wraps(old_reference)
    def load_reference(self, record):
        match = records.get(id(record))
        self.backend._parameter_record_supplier = (lambda: old_build(match[0], match[1])) if match else lambda: record
        return old_reference(self, record)
    orchestrator.RecoveryOrchestrator._load_scene_reference = load_reference
    old_fetch = backend_module.GelloRecoveryBackend._fetch_perception_inputs

    @functools.wraps(old_fetch)
    def fetch(self, *args, **kwargs):
        value = old_fetch(self, *args, **kwargs)
        self._parameter_scene_image = None
        if value is not None:
            rgb = value[0]
            if isinstance(rgb, bytes):
                data = rgb
                mime = 'image/jpeg' if rgb.startswith(b'\xff\xd8') else 'image/png'
            else:
                import cv2
                ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError('passive RGB serialization failed')
                data = encoded.tobytes()
                mime = 'image/png'
            self._parameter_scene_image = image_part(data, mime, {'phase': 'perception_for_current_candidate_pool', 'captured_wall_ts': time.time(), 'timestamp_meaning': 'existing fetch completed, not hardware exposure timestamp', 'identity_status': 'algorithm input, not confirmed target truth'})
        return value
    backend_module.GelloRecoveryBackend._fetch_perception_inputs = fetch
    old_assemble = capture.assemble_evidence

    def assemble(backend, steps, attribution):
        evidence = old_assemble(backend, steps, attribution)
        images = []
        missing = []
        if evidence.get('schema') != 'carryon-same-attempt-evidence/v1':
            backend.jam_z_reflector.attempt_images = []
            return evidence
        binding = evidence['binding']
        descent = evidence.get('descent', {})
        start = descent.get('started_wall_ts')
        end = descent.get('finished_wall_ts')
        from carryon.recovery.visual_object_memory import _norm
        key = _norm(getattr(backend, '_last_grasp_object_key', '') or backend.task_instruction)
        detectors = list(getattr(backend, 'detector_sources', {}).values()) + [getattr(backend, 'detector', None)]
        selector = next((getattr(d, 'visual_object_selector', None) for d in detectors if getattr(d, 'visual_object_selector', None) is not None), None)
        ref = selector.memory.reference_for(key) if selector else None
        if ref and Path(ref).is_file():
            p = Path(ref)
            data = p.read_bytes()
            images.append(image_part(data, 'image/png', {'phase': 'reference', 'source_path': str(p)}))
        else:
            missing.append('reference unavailable')
        scene = copy.deepcopy(getattr(backend, '_parameter_scene_image', None))
        if scene:
            scene['metadata']['binding'] = binding
            captured = scene['metadata'].get('captured_wall_ts')
            scene['metadata']['age_at_descent_start_s'] = None if start is None or captured is None else start - captured
            images.append(scene)
        else:
            missing.append('existing pre-grasp perception capture unavailable')
        from critical_frame_capture import for_attempt
        captured, missing_reason = for_attempt(backend, evidence)
        if captured is not None:
            data, metadata = captured
            images.append(image_part(data, metadata['mime'], metadata))
        else:
            missing.append(missing_reason)
        evidence['visual'] = {'manifest': [x['metadata'] for x in images], 'missing': missing, 'critical_frame_in_request': captured is not None, 'critical_frame_source': 'direct camera at descent return', 'maximum_scene_images': 2}
        import tempfile
        archive = Path(tempfile.mkdtemp(prefix='attempt_', dir=root))
        for index, image in enumerate(images):
            url = image['content']['image_url']['url']
            data = base64.b64decode(url.split(',', 1)[1], validate=True)
            assert digest(data) == image['metadata']['sha256']
            extension = '.jpg' if url.startswith('data:image/jpeg;') else '.png'
            path = archive / ('IMAGE_' + str(index) + extension)
            with path.open('xb') as stream:
                stream.write(data)
        path = archive / 'INPUT_EVIDENCE.json'
        data = (json.dumps(evidence, sort_keys=True, indent=2) + '\n').encode()
        with path.open('xb') as stream:
            stream.write(data)
        backend.jam_z_reflector.attempt_images = images
        return evidence
    capture.assemble_evidence = assemble
    return {'fetch': fetch, 'assemble': assemble, 'record_build': build, 'reference': load_reference}

def configure_inner(inner, selector_arm, row_root):
    from parameter_rationale_preference import configure_inner as accepted
    accepted(inner, selector_arm, row_root)
    install_client()
    return install_passive_evidence(row_root)
