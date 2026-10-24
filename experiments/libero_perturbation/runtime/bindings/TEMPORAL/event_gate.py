import asyncio, math, copy

class SuspectWakeGate:

    def __init__(self, job, generation, closed_command, max_age, clock):
        self.job = job
        self.generation = generation
        self.closed = closed_command
        self.max_age = max_age
        self.clock = clock
        self.pending = None
        self.seen = set()
        self.serial = 0
        self.attempted = 0
        self.terminal = False
        self.wake = asyncio.Event()

    def offer(self, event):
        if self.terminal or event.get('job') != self.job or event.get('generation') != self.generation:
            return False
        ts = event.get('producer_ts')
        if not isinstance(ts, (int, float)) or not math.isfinite(ts) or (not 0 <= self.clock() - ts <= self.max_age):
            return False
        w = event.get('w_meas')
        if not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0:
            return False
        if event.get('w_cmd') != self.closed:
            return False
        if event.get('kind') not in ('suspected_empty_close', 'suspected_hold_loss'):
            return False
        identity = ts
        if identity in self.seen:
            return False
        self.seen.add(identity)
        self.serial += 1
        self.pending = {'kind': event['kind'], 'producer_ts': ts, 'w_cmd': event['w_cmd'], 'w_meas': event.get('w_meas'), 'prior_stable_width': copy.deepcopy(event.get('prior_stable_width')), 'serial': self.serial, 'status': 'SUSPECTED_NOT_MODEL_VERDICT'}
        self.wake.set()
        return True

    def snapshot(self):
        self.attempted = self.serial
        return copy.deepcopy(self.pending)

    def observed(self, decision):
        result = decision.result if isinstance(decision.result, dict) else {}
        obs = result.get('_observation') or {}
        ts = obs.get('producer_frame_ts')
        if self.pending and isinstance(ts, (int, float)) and (ts >= self.pending['producer_ts']):
            self.pending = None
            self.wake.clear()

    async def wait(self, seconds):
        if self.terminal:
            return
        if self.pending and self.serial > self.attempted:
            return
        self.wake.clear()
        try:
            await asyncio.wait_for(self.wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    def close(self):
        self.terminal = True
        self.pending = None
        self.wake.set()

def frame_after_event(context, metadata):
    event = context.get('suspected_measurement_event')
    if not event:
        return True
    ts = event['producer_ts']
    return bool(metadata) and all((isinstance(m.get('producer_frame_ts'), (int, float)) and m['producer_frame_ts'] >= ts for m in metadata.values()))
