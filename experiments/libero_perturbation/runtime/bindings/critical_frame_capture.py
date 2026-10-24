"""One existing-camera fetch immediately after physical descent; no motion changes."""
import copy
import functools
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

def sha(data):
    return hashlib.sha256(data).hexdigest()

def store(path, data):
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

def install(base, root):
    original = base._stream_joint_path
    assert not getattr(original, '__critical_frame_v1__', False)
    root = Path(root) / 'critical_frames'
    root.mkdir(parents=True, exist_ok=False)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if kwargs.get('motion_segment') != 'descent':
            return original(self, *args, **kwargs)
        result = None
        motion_error = None
        try:
            result = original(self, *args, **kwargs)
            return result
        except Exception as exc:
            motion_error = type(exc).__name__ + ': ' + str(exc)
            raise
        finally:
            event = {'status': 'MISSING', 'phase': 'descent_return_before_close_lift_or_retreat', 'started_wall_ts': time.time(), 'motion_ok': getattr(result, 'ok', None), 'motion_error': motion_error, 'camera_calls': 0, 'timestamp_meaning': 'host fetch interval, not hardware exposure timestamp'}
            try:
                event['binding'] = copy.deepcopy((getattr(self, 'last_goalset_execution_debug', {}) or {}).get('executed_selection'))
                if not event['binding']:
                    raise ValueError('physical candidate binding absent')
                event['camera'] = self.cfg.grasp_camera
                fetch = getattr(self.gello, 'fetch_raw_rgb_jpeg', None)
                if not callable(fetch):
                    fetch = self.gello.fetch_rgb_jpeg
                event['camera_calls'] = 1
                data = fetch(event['camera'])
                if not isinstance(data, bytes) or not data:
                    raise ValueError('camera returned no image bytes')
                from PIL import Image
                with Image.open(io.BytesIO(data)) as picture:
                    mime = Image.MIME.get(picture.format)
                    picture.verify()
                if mime not in ('image/jpeg', 'image/png'):
                    raise ValueError('camera returned unsupported image format')
                archive = Path(tempfile.mkdtemp(prefix='descent_', dir=root))
                path = archive / ('FRAME.jpg' if mime == 'image/jpeg' else 'FRAME.png')
                store(path, data)
                event.update(status='CAPTURED', source_path=str(path), sha256=sha(data), bytes=len(data), mime=mime)
            except Exception as exc:
                event.update(status='MISSING', error=type(exc).__name__ + ': ' + str(exc))
            event['finished_wall_ts'] = time.time()
            events = getattr(self, '_parameter_critical_frames', None)
            if events is None:
                events = []
                self._parameter_critical_frames = events
            events.append(event)
            try:
                archive = Path(tempfile.mkdtemp(prefix='receipt_', dir=root))
                store(archive / 'CAPTURE.json', (json.dumps(event, sort_keys=True, indent=2) + '\n').encode())
            except Exception as exc:
                event.update(status='MISSING', receipt_error=type(exc).__name__ + ': ' + str(exc))
    wrapped.__critical_frame_v1__ = True
    base._stream_joint_path = wrapped
    return (original, wrapped)

def for_attempt(backend, evidence):
    """Binding AND the actual motion interval; never reuse an earlier same-pose image."""
    descent = evidence.get('descent', {})
    start, end = (descent.get('started_wall_ts'), descent.get('finished_wall_ts'))
    if start is None or end is None:
        return (None, 'same-attempt descent interval absent')
    matches = [e for e in getattr(backend, '_parameter_critical_frames', []) if e.get('binding') == evidence['binding'] and start <= e['started_wall_ts'] <= e['finished_wall_ts'] <= end]
    if len(matches) != 1:
        return (None, 'same-attempt critical capture count=' + str(len(matches)))
    event = copy.deepcopy(matches[0])
    if event['status'] != 'CAPTURED':
        return (None, 'critical capture failed: ' + event.get('error', event.get('receipt_error', 'unknown')))
    data = Path(event['source_path']).read_bytes()
    if sha(data) != event['sha256']:
        raise ValueError('critical frame archive SHA mismatch')
    return ((data, event), None)
