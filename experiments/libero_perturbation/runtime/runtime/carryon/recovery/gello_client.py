"""HTTP client for camera observations, endpoint execution and control handoff."""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional
DEFAULT_TIMEOUT_S = 5.0

class GelloClientError(RuntimeError):
    """Raised on transport failure or a non-2xx gello response."""

@dataclass
class ExecuteResult:
    accepted: bool
    job_id: Optional[str] = None
    reason: Optional[str] = None
    raw: Optional[dict] = None

@dataclass
class CancelResult:
    ok: bool
    cancelled: bool
    return_home: bool = False
    standby: bool = False
    reason: Optional[str] = None
    raw: Optional[dict] = None

class GelloControlClient:
    """Thin HTTP client for one gello instance.

    ``base_url`` is the RemoteTaskControlServer root (e.g. ``http://host:18080``).
    ``camera_url`` is the CameraSnapshotHTTPServer root (e.g. ``http://host:18110``);
    defaults to ``base_url`` host on port 18110 if not given.
    """

    def __init__(self, base_url: str, *, camera_url: Optional[str]=None, timeout_s: float=DEFAULT_TIMEOUT_S, auth_token: str='') -> None:
        self.base_url = base_url.rstrip('/')
        self.camera_url = (camera_url or self._default_camera_url()).rstrip('/')
        self.timeout_s = timeout_s
        self.auth_token = auth_token

    def _default_camera_url(self) -> str:
        from urllib.parse import urlparse, urlunparse
        p = urlparse(self.base_url)
        host = p.hostname or '127.0.0.1'
        return urlunparse((p.scheme or 'http', f'{host}:18110', '', '', '', ''))

    def _request(self, method: str, url: str, *, body: Optional[dict]=None, expect_json: bool=True) -> Any:
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        if self.auth_token:
            headers['X-CarryOn-Token'] = self.auth_token
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode('utf-8'))
            except Exception:
                raise GelloClientError(f'{method} {url} -> HTTP {e.code}') from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise GelloClientError(f'{method} {url} failed: {e}') from e
        if not expect_json:
            return payload
        try:
            return json.loads(payload.decode('utf-8'))
        except Exception as e:
            raise GelloClientError(f'{method} {url}: invalid JSON response') from e

    def healthz(self) -> bool:
        try:
            resp = self._request('GET', f'{self.base_url}/healthz')
        except GelloClientError:
            return False
        return bool(isinstance(resp, dict) and resp.get('ok'))

    def execute(self, instruction: str, *, request_id: Optional[str]=None, skip_reset: bool=False, inputs: Optional[dict]=None, callback_url: str='', deadline_s: Optional[float]=None) -> ExecuteResult:
        """Submit a task. ``skip_reset`` is folded into ``inputs`` so gello's
        episode start resumes from the current (recovery-restored) pose."""
        merged_inputs = dict(inputs or {})
        if skip_reset:
            merged_inputs['skip_reset'] = True
        body: dict[str, Any] = {'instruction': instruction, 'inputs': merged_inputs}
        if request_id is not None:
            body['request_id'] = request_id
        if callback_url:
            body['callback_url'] = callback_url
        if deadline_s is not None:
            body['deadline_s'] = deadline_s
        resp = self._request('POST', f'{self.base_url}/execute', body=body)
        if not isinstance(resp, dict):
            raise GelloClientError('execute: unexpected response shape')
        return ExecuteResult(accepted=bool(resp.get('accepted')), job_id=resp.get('job_id'), reason=resp.get('reason'), raw=resp)

    def cancel(self, request_id: str, *, return_home: bool=False, standby: bool=True) -> CancelResult:
        """Cancel the active/pending task. For a recovery takeover use the
        defaults (``return_home=False, standby=True``): gello brakes, holds via
        the bridge, and stops sending commands so CarryOn can drive the arm."""
        body = {'return_home': bool(return_home), 'standby': bool(standby)}
        resp = self._request('POST', f'{self.base_url}/cancel/{request_id}', body=body)
        if not isinstance(resp, dict):
            raise GelloClientError('cancel: unexpected response shape')
        return CancelResult(ok=bool(resp.get('ok')), cancelled=bool(resp.get('cancelled')), return_home=bool(resp.get('return_home')), standby=bool(resp.get('standby')), reason=resp.get('reason'), raw=resp)

    def _terminal_handshake(self, endpoint: str, request_id: str, *, success_state: str, timeout_s: Optional[float]=None) -> bool:
        """Submit an idempotent control request and wait for its world-thread ack."""
        rid = str(request_id or '').strip()
        if not rid:
            return False
        try:
            accepted = self._request('POST', f'{self.base_url}/{endpoint}', body={'request_id': rid})
            if not isinstance(accepted, dict) or not accepted.get('accepted'):
                return False
            if accepted.get('state') == success_state:
                return True
            deadline = time.monotonic() + float(self.timeout_s if timeout_s is None else timeout_s)
            while time.monotonic() < deadline:
                status = self._request('POST', f'{self.base_url}/{endpoint}/status', body={'request_id': rid})
                if isinstance(status, dict):
                    state = str(status.get('state', ''))
                    if state == success_state:
                        return True
                    if state in {'failed', 'rejected', 'conflict', 'unknown'}:
                        return False
                time.sleep(0.05)
        except Exception:
            return False
        return False

    def complete_terminal(self, request_id: str='') -> bool:
        """Wait until the shim has settled and finalized the episode.

        The completed state acknowledges control only; benchmark success remains
        evaluator output and is never exposed to policy or candidate gating.
        """
        return self._terminal_handshake('complete_terminal', request_id, success_state='completed')

    def pregrasp_resume(self, request_id: str='') -> bool:
        """Wait until the world thread has actually resumed VLA control."""
        return self._terminal_handshake('pregrasp_resume', request_id, success_state='resumed')

    def fetch_rgb_jpeg(self, camera: str) -> bytes:
        return self._request('GET', f'{self.camera_url}/cameras/{camera}/latest.jpg', expect_json=False)

    def fetch_raw_rgb_jpeg(self, camera: str) -> bytes:
        return self._request('GET', f'{self.camera_url}/cameras/{camera}/latest_raw.jpg', expect_json=False)

    def fetch_depth_png(self, camera: str) -> bytes:
        return self._request('GET', f'{self.camera_url}/cameras/{camera}/latest_depth.png', expect_json=False)

    def fetch_intrinsics(self, camera: str) -> dict:
        resp = self._request('GET', f'{self.camera_url}/cameras/{camera}/intrinsics.json')
        if not isinstance(resp, dict):
            raise GelloClientError('intrinsics: unexpected response shape')
        return resp

    @staticmethod
    def decode_depth_png(png_bytes: bytes):
        """Decode a 16-bit depth PNG to a ``(H, W)`` uint16 numpy array.

        Lazily imports numpy + cv2 so the control plane stays dependency-light.
        """
        import cv2
        import numpy as np
        arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise GelloClientError('failed to decode depth PNG')
        return arr

    def fetch_depth_meters(self, camera: str):
        """Convenience: fetch depth + intrinsics and return a metric float32
        depth map (meters) using the camera's ``depth_scale``."""
        import numpy as np
        depth_raw = self.decode_depth_png(self.fetch_depth_png(camera))
        intr = self.fetch_intrinsics(camera)
        scale = float(intr.get('depth_scale', 0.001))
        return depth_raw.astype(np.float32) * scale
