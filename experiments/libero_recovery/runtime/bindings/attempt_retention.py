"""Retain images and measured state for each recovery attempt."""
import copy, functools, hashlib, io, json, os, time, uuid
from pathlib import Path

def encoded(v):
    return (json.dumps(v, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def store(p, b):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())

def measured(st):
    if not isinstance(st, dict):
        return None
    return {k: copy.deepcopy(st[k]) for k in ('ts', 'w_meas', 'w_cmd', 'q', 'ee_pos', 'ee_quat', 'state_source') if k in st}

def install(guard, backend_class, root):
    """After width-only binding, before the same factory instance executes steps."""
    from selection_origin_binding import binding
    import gripper_width as N
    root = Path(root) / 'attempt_evidence'
    root.mkdir(parents=True, exist_ok=False)
    oldread = backend_class._read_state

    @functools.wraps(oldread)
    def read(self, *a, **kw):
        st = oldread(self, *a, **kw)
        self._retention_last_read = {'state': measured(st), 'read_finished_wall_ts': time.time()}
        return st
    backend_class._read_state = read
    cls = guard.STATE['class']
    old = cls._run_step
    candidate = guard.STATE['candidate']

    def archive(self, b, ctx, step, res, started, exception=None, capture=False):
        event = {'schema': 'same-attempt-retained-observation/v2', 'common_family_id': getattr(ctx, '_common_family_id', None), 'attempt_ordinal': getattr(ctx, '_common_attempt_ordinal', None), 'phase': step.op, 'started_wall_ts': started, 'step_returned_wall_ts': time.time(), 'capture_state': 'NOT_REQUESTED', 'executed_selection': binding(ctx), 'last_read': copy.deepcopy(getattr(b, '_retention_last_read', None)), 'step_exception': exception, 'data': N.clean(copy.deepcopy(getattr(res, 'data', {}) or {}))}
        try:
            if not N.valid_binding(event['executed_selection']):
                raise ValueError('missing actual executed selection')
            entry = getattr(ctx, '_retention_attempt_dir', None)
            if entry is None:
                entry = root / ('attempt_' + uuid.uuid4().hex)
                entry.mkdir()
                ctx._retention_attempt_dir = entry
                ctx._retention_binding = copy.deepcopy(event['executed_selection'])
                ctx._retention_sequence = 0
            if ctx._retention_binding != event['executed_selection']:
                raise ValueError('executed binding changed inside retained attempt')
            seq = ctx._retention_sequence
            ctx._retention_sequence += 1
            event['sequence'] = seq
            if step.op == 'lift':
                event['complete_lift_trace'] = N.clean(copy.deepcopy(b.last_lift_width_trace()))
            if capture:
                event['image_phase'] = 'after_descent_return_before_close' if step.op == 'goto_grasp_topdown_family' else 'after_close_before_lift' if step.op == 'close_gripper' else 'after_lift_and_holding_check_before_following_motion' if step.op == 'assert_holding' else 'after_lift_failed_return_before_handback'
                event['image_fetch_started_wall_ts'] = time.time()
                event['exposure_timestamp'] = None
                event['image_timestamp_meaning'] = 'host fetch interval only, not exposure time'
                event['camera_calls'] = 0
                try:
                    if not 0 < float(b.gello.timeout_s) <= 5.0:
                        raise ValueError('existing camera timeout outside retained contract')
                    fetch = b.gello.fetch_raw_rgb_jpeg
                    event['camera_calls'] = 1
                    data = fetch(b.cfg.grasp_camera)
                    if not isinstance(data, bytes) or not data:
                        raise ValueError('missing camera bytes')
                    from PIL import Image
                    with Image.open(io.BytesIO(data)) as im:
                        mime = Image.MIME.get(im.format)
                        im.verify()
                    if mime not in ('image/png', 'image/jpeg'):
                        raise ValueError('invalid camera image')
                    p = entry / (f'{seq:03}_{step.op}' + ('.png' if mime == 'image/png' else '.jpg'))
                    store(p, data)
                    event.update(capture_state='CAPTURED', image_path=str(p), image_sha256=hashlib.sha256(data).hexdigest(), mime=mime)
                except Exception as e:
                    event.update(capture_state='MISSING', capture_error=repr(e))
                event['image_fetch_finished_wall_ts'] = time.time()
            p = entry / f'{seq:03}_{step.op}.json'
            store(p, encoded(event))
            if res is not None:
                if res.data is None:
                    res.data = {}
                res.data['durable_observation'] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'capture_state': event['capture_state']}
        except Exception as e:
            errors = getattr(b, '_retention_errors', [])
            errors.append({'phase': step.op, 'error': repr(e)})
            b._retention_errors = errors
            if res is not None:
                if res.data is None:
                    res.data = {}
                res.data['durable_observation'] = {'status': 'MISSING', 'error': repr(e)}

    @functools.wraps(old)
    def run(self, b, ctx, step):
        if step.op not in ('goto_grasp_topdown_family', 'close_gripper', 'lift', 'assert_holding'):
            return old(self, b, ctx, step)
        started = time.time()
        res = None
        error = None
        try:
            res = old(self, b, ctx, step)
            return res
        except Exception as e:
            error = repr(e)
            raise
        finally:
            capture = step.op in ('goto_grasp_topdown_family', 'close_gripper')
            if step.op == 'lift':
                ctx._retention_pending_lift = bool(res is not None and res.ok)
                capture = not ctx._retention_pending_lift
            if step.op == 'assert_holding' and getattr(ctx, '_retention_pending_lift', False):
                capture = True
                ctx._retention_pending_lift = False
            archive(self, b, ctx, step, res, started, error, capture)
    cls._run_step = candidate.SequentialPolicy._run_step = run
    return {'class_identity_preserved': True, 'original_step_file': old.__code__.co_filename, 'new_step_file': run.__code__.co_filename, 'backend_read_before': oldread.__code__.co_filename, 'backend_read_after': read.__code__.co_filename, 'root': str(root)}
