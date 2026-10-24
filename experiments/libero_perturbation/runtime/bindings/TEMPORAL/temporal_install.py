import os, json, time, asyncio, hashlib, base64, types, copy, math
from pathlib import Path
from event_gate import SuspectWakeGate
from temporal_detector import SuspectedHoldLoss
from qualification_history import FirstQualificationFetch
from event_parser import parse
from carryon.judge.client import JudgeDecision
REG = None
ACTIVE = {}
GEN = {}
D = Path(__file__).parent
FROZEN = json.loads((D / 'TEMPORAL.json').read_text())
SYSTEM = FROZEN['messages'][0]['content']
PBASE = json.loads(FROZEN['messages'][1]['content'][0]['text'])['proprioception']

def attach_registry(reg):
    global REG
    REG = reg
    if reg is None or reg._stream is None:
        raise RuntimeError('MISSING_REAL_STATE_STREAM')

def log(name, data):
    p = Path(os.environ['QUALIFICATION_HISTORY_OUTPUT'])
    p.mkdir(parents=True, exist_ok=True)
    with (p / 'TEMPORAL_CHAIN.jsonl').open('a') as f:
        f.write(json.dumps({'event': name, 'wall_ts': time.time(), **data}) + '\n')

def inconclusive(reason):
    return JudgeDecision(status='running', error=reason, result={'source': 'temporal_judge', 'certified_P_failure': False})

async def fetch_current(http, camera, timeout_s):
    try:
        response = await http.get(camera['snapshot_url'], timeout=timeout_s)
        if response.status_code != 200 or not response.content:
            return None
        h = response.headers
        meta = {'producer_frame_ts': float(h['x-carryon-producer-frame-ts']), 'producer_frame_seq': int(h['x-carryon-producer-frame-seq']), 'history_session': h.get('x-carryon-history-session')}
        mime = h.get('content-type', 'image/jpeg').split(';')[0]
        return ({'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(response.content).decode()}}, meta)
    except Exception:
        return None

async def judge(self, context):
    job = context['job_key']
    state = ACTIVE.get(job)
    if state is None or state['gate'].terminal:
        return inconclusive('STALE_JOB_NO_HTTP')
    generation = state['gate'].generation
    async with self._temporal_request_lock:
        if ACTIVE.get(job) is not state or state['gate'].terminal:
            return inconclusive('STALE_JOB_NO_HTTP')
        camera = next((c for c in self.cameras if c.get('snapshot_url')), None)
        if camera is None:
            return inconclusive('NO_CURRENT_NO_HTTP')
        started = time.time()
        got = await fetch_current(self._http, camera, timeout_s=self.image_fetch_timeout_s)
        if ACTIVE.get(job) is not state or state['gate'].terminal:
            return inconclusive('LATE_CURRENT_NO_HTTP')
        if got is None:
            return inconclusive('NO_CURRENT_NO_HTTP')
        part, meta = got
        ts = meta.get('producer_frame_ts')
        session = meta.get('history_session')
        if session != state['session']:
            return inconclusive('CURRENT_SESSION_MISMATCH_NO_HTTP')
        if not isinstance(ts, (int, float)) or not math.isfinite(ts):
            return inconclusive('NO_PRODUCER_TIME_NO_HTTP')
        event = context.get('suspected_measurement_event')
        if event and ts < event['producer_ts']:
            return inconclusive('EVENT_FRAME_NOT_FRESH_NO_HTTP')
        frames = []
        parts = []
        anchor = (state['fetch'].result or {}).get('anchor')
        if anchor and anchor['session'] == state['session'] and (anchor['producer_ts'] <= ts):
            frames.append({'id': 'PREVIOUS_1', 'producer_ts': anchor['producer_ts'], 'sha256': anchor['sha256']})
            parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + anchor['jpeg_base64']}})
        raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
        frames.append({'id': 'CURRENT', 'producer_ts': ts, 'sha256': hashlib.sha256(raw).hexdigest()})
        parts.append(part)
        prop = copy.deepcopy(context.get('proprioception') or {})
        samples = prop.get('measured_width_samples', [])
        prop = {'measured_width_samples': [s for s in samples if isinstance(s.get('producer_ts'), (int, float)) and s['producer_ts'] <= ts], 'note': PBASE['note'], 'width_m_definition': PBASE['width_m_definition']}
        ev = None if event is None else {k: event[k] for k in ('kind', 'producer_ts', 'w_cmd', 'w_meas', 'status') if k in event}
        payload = {'instruction': context.get('instruction', ''), 'suspected_measurement_event': ev, 'proprioception': prop, 'frames': frames, 'cutoff_producer_ts': ts}
        content = [{'type': 'text', 'text': json.dumps(payload)}]
        for frame, image in zip(frames, parts):
            content.extend([{'type': 'text', 'text': json.dumps(frame)}, image])
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
        api_start = time.time()
        log('JUDGE_REQUEST', {'job': job, 'generation': generation, 'fetch_start_wall': started, 'current_producer_ts': ts, 'frames': frames, 'suspected_event': ev, 'anchor_missing': anchor is None})
        try:
            result = await self._chat.chat(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, response_format={'type': 'json_object'})
            if ACTIVE.get(job) is not state or state['gate'].terminal:
                return inconclusive('LATE_RESPONSE_REJECTED')
            api_end = time.time()
            parse_start = time.time()
            p = parse(result.text, [f['id'] for f in frames])
            parse_end = time.time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import budget24
            budget24.guard().stop('TEMPORAL_HTTP_OR_PARSE_FAILURE', error_type=type(e).__name__)
            return inconclusive('TEMPORAL_INCONCLUSIVE_' + type(e).__name__)
        failed = p['current_grasp_state'] == 'VISIBLE_GRASP_FAILURE' or p['earlier_failure_visible'] is True
        status = 'failed' if failed else 'running'
        out = {'source': 'vlm_temporal_judge', 'model_classification': p, 'visual_evidence': p['current_evidence'], 'certified_P_failure': False, '_observation': {'producer_frame_ts': ts, 'producer_frame_seq': meta.get('producer_frame_seq'), 'image_fetch_start_wall_ts': started, 'image_fetch_end_wall_ts': api_start, 'image_fetch_ms': (api_start - started) * 1000, 'judge_api_request_wall_ts': api_start, 'judge_api_response_wall_ts': api_end, 'judge_api_ms': (api_end - api_start) * 1000, 'parse_ms': (parse_end - parse_start) * 1000}}
        log('JUDGE_PARSED', {'job': job, 'generation': generation, 'raw_text': result.text, 'mapped_status': status, 'certified_P_failure': False, 'observation_timing': out['_observation']})
        return JudgeDecision(status=status, result=out)

def install(cls):
    original = cls.start_job
    stop = cls.stop

    def start(self, **kw):
        if not hasattr(self, '_temporal_request_lock'):
            self._temporal_request_lock = asyncio.Lock()
            self.client._temporal_request_lock = self._temporal_request_lock
            self.client.judge = types.MethodType(judge, self.client)
        self._suspect_gate_factory = lambda job: factory(self, job)
        return original(self, **kw)

    def factory(self, job):
        for old in ACTIVE.values():
            old['gate'].close()
            old['fetch'].close()
        gen = GEN.get(job, 0) + 1
        GEN[job] = gen
        session = os.environ['QUALIFICATION_HISTORY_SESSION']
        cal = json.loads(Path(os.environ['QUALIFICATION_REAL_WIDTH_CONTRACT']).read_text())
        loop = asyncio.get_running_loop()
        gate = SuspectWakeGate(job, gen, cal['closed_command_m'], cal['measurement_max_age_s'], time.time)
        camera = next((c for c in self.config.cameras if c.get('snapshot_url')))
        url = camera['snapshot_url'].rsplit('/', 1)[0] + '/history.jpg'
        fetch = FirstQualificationFetch(job, gen, session, url, self.client._http, self.client.image_fetch_timeout_s)
        det = SuspectedHoldLoss(job, gen, cal, time.time, lambda e: (log('SUSPECT_EVENT', {'job': job, 'generation': gen, 'measurement': e}), gate.offer(e)))
        det.qualification_history_request = fetch.on_qualification
        state = {'gate': gate, 'fetch': fetch, 'det': det, 'session': session}
        ACTIVE[job] = state
        if not hasattr(REG, '_temporal_original_hook'):
            REG._temporal_original_hook = REG._stream._on_sample
        previous = REG._temporal_original_hook

        def sample(s):
            if previous:
                previous(s)

            def feed():
                if ACTIVE.get(job) is state and (not gate.terminal):
                    det.on_sample(s)
            loop.call_soon_threadsafe(feed)
        REG.set_gripper_sample_hook(sample)
        return gate

    async def shutdown(self):
        for state in ACTIVE.values():
            state['gate'].close()
            state['fetch'].close()
        try:
            return await stop(self)
        finally:
            ACTIVE.clear()
    cls.start_job = start
    cls.stop = shutdown
