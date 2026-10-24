"""Isolated pose-classification worker with explicitly configured transport."""
from runtime_paths import bound_path, resource
from pathlib import Path
import hashlib, json
from pose_transport import ArchivedPoseClient
from pose_resolution_adapter import make_resolver
from pose_grasp_family import install
from pose_memory import make_record
_BINDINGS = {}

def frozen_memory_loader(path, expected_sha):
    path = Path(path)

    def load():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError('pose snapshot drift')
        data = json.loads(raw)
        if data.get('schema') != 'isolated-pose-snapshot/v1':
            raise ValueError('wrong snapshot schema')
        if data.get('coordinate') != 'world_contact_tcp_z_m':
            raise ValueError('wrong z coordinate')
        rows = data['records']
        seen = set()
        for r in rows:
            if r.get('schema') != 'isolated-object-pose-p/v1':
                raise ValueError('legacy record not pose memory')
            k = (r['object_key'], r['pose'])
            if k in seen:
                raise ValueError('duplicate active object pose')
            seen.add(k)
            make_record(r['object_key'], {'pose': r['pose'], 'visual_basis': 'snapshot validation'}, r['corrected_z_m'], r['source_request_sha'], r['revision'])
            for field in ('source_request_sha', 'source_response_sha'):
                v = r.get(field, '')
                if len(v) != 64 or any((c not in '0123456789abcdef' for c in v)):
                    raise ValueError('missing source digest')
        return rows
    load()
    return load

def register(*, row_root, snapshot_path, snapshot_sha, send_once, max_pose_calls, provider=None, response_provider_aliases=None):
    """Caller must bind the configured transport explicitly; role is pose_classifier."""
    root = str(Path(row_root).resolve())
    if root in _BINDINGS:
        raise ValueError('row already registered')
    if not callable(send_once) or type(max_pose_calls) != int or max_pose_calls < 1:
        raise ValueError('invalid call budget')
    loader = frozen_memory_loader(snapshot_path, snapshot_sha)
    count = [0]

    def bounded_send(raw):
        if count[0] >= max_pose_calls:
            raise RuntimeError('pose call budget exhausted')
        count[0] += 1
        return send_once(raw)
    client = ArchivedPoseClient(Path(root) / 'pose_calls', bounded_send, provider=provider, response_provider_aliases=response_provider_aliases)
    _BINDINGS[root] = (make_resolver(client, loader), count)
    return {'role': 'pose_classifier', 'max_attempts': max_pose_calls, 'HTTP_retries': 0, 'snapshot_sha': snapshot_sha}

def install_family(backend_class, exact_executor, *, archive_root):
    root = str(Path(archive_root).resolve())
    if root not in _BINDINGS:
        raise RuntimeError('explicit pose transport/snapshot registration required')
    resolver, count = _BINDINGS[root]
    return install(backend_class, exact_executor, archive_root=archive_root, pose_resolver=resolver)
