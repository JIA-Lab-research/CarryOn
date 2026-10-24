import math
import copy
from carryon.judge.checkpoint import GripperTransitionMonitor
from gripper_width import calibration_valid
STABLE_S = 0.06
STABLE_BAND_M = 0.003

class SuspectedHoldLoss:

    def __init__(self, job, generation, calibration, clock, deliver):
        if not calibration_valid(calibration):
            raise ValueError('REAL_CALIBRATION_INVALID')
        self.job = job
        self.generation = generation
        self.cal = calibration
        self.clock = clock
        self.deliver = deliver
        self.last_ts = None
        self.last_w = None
        self.last_cmd = None
        self.diagnostics = []
        self.events = []
        self._reset('INITIAL')

    def _reset(self, reason):
        self.monitor = GripperTransitionMonitor(hold_min_m=self.cal['empty_width_m'] + self.cal['hold_margin_m'], hold_max_m=self.cal['max_open_width_m'] - self.cal['max_open_margin_m'], empty_max_m=self.cal['empty_width_m'] + self.cal['hold_margin_m'], stable_s=STABLE_S, stable_band_m=STABLE_BAND_M, pause_fn=None, hold_loss_fn=None, arm_fn=None, max_pauses=0, emptygrasp_count=0, clock=self.clock, log=lambda _: None)
        self.monitor.arm(self.job)
        self.last_ts = None
        self.last_w = None
        self.last_cmd = None
        self.hold_evidence = None
        self._qualification_samples = []
        self.diagnostics.append(reason)

    def on_sample(self, s):
        ts = getattr(s, 'ts', None)
        w = getattr(s, 'w_meas', None)
        cmd = getattr(s, 'w_cmd', None)
        valid = lambda x: isinstance(x, (float, int)) and (not isinstance(x, bool)) and math.isfinite(x)
        if not all((valid(x) for x in [ts, w, cmd])) or w < 0:
            self._reset('MISSING_INVALID')
            return
        now = self.clock()
        if not 0 <= now - ts <= self.cal['measurement_max_age_s']:
            self._reset('STALE')
            return
        if cmd != self.cal['closed_command_m']:
            self._reset('NOT_CONTINUOUS_CLOSE')
            return
        if self.last_ts is not None:
            if ts == self.last_ts:
                if w != self.last_w or cmd != self.last_cmd:
                    self._reset('CONFLICTING_DUPLICATE')
                return
            if ts < self.last_ts:
                self._reset('REVERSED_TIME')
                return
            if ts - self.last_ts > self.cal['measurement_max_age_s']:
                self._reset('HISTORY_GAP')
        self.last_ts = ts
        self.last_w = w
        self.last_cmd = cmd
        before = len(self.monitor._events)
        prior_state = self.monitor._state
        self._qualification_samples.append({'producer_ts': ts, 'w_meas': w, 'w_cmd': cmd})
        self.monitor.on_sample(ts, w, grasp_contact=None, release_commanded=False)
        start = self.monitor._win_start_ts
        if start is not None:
            self._qualification_samples = [x for x in self._qualification_samples if x['producer_ts'] >= start]
        if prior_state != 'holding' and self.monitor._state == 'holding' and (start is not None):
            self.hold_evidence = {'window_start_producer_ts': start, 'window_last_producer_ts': ts, 'window_min_width_m': self.monitor._win_min, 'window_max_width_m': self.monitor._win_max, 'stable_required_s': STABLE_S, 'stable_band_m': STABLE_BAND_M, 'qualification': 'NONEMPTY_WIDTH_ONLY_NOT_OBJECT_HOLD_TRUTH', 'qualification_samples': copy.deepcopy(self._qualification_samples), 'image_anchor': None, 'image_anchor_status': 'MISSING_NO_CAMERA_FRAME_CHANNEL_AT_SAMPLE_HOOK'}
        if prior_state != 'holding' and self.monitor._state == 'holding' and (self.hold_evidence is not None):
            request = getattr(self, 'qualification_history_request', None)
            if request is not None:
                request(self.job, self.generation, copy.deepcopy(self.hold_evidence))
        if start is None:
            self._qualification_samples = []
        for old in self.monitor._events[before:]:
            if old.kind == 'hold_loss' and w < self.monitor.empty_max_m:
                event = {'job': self.job, 'generation': self.generation, 'kind': 'suspected_hold_loss', 'producer_ts': ts, 'w_cmd': cmd, 'w_meas': w, 'status': 'SUSPECTED_NOT_MODEL_VERDICT', 'prior_stable_width': copy.deepcopy(self.hold_evidence)}
                self.events.append(event)
                self.deliver(event)

def install_sample_hook(registry, detector, loop):
    registry.set_gripper_sample_hook(lambda sample: loop.call_soon_threadsafe(detector.on_sample, sample))
