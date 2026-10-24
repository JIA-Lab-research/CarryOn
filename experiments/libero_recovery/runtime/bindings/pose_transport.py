"""Send one pose-classification request through the configured transport."""
from runtime_paths import bound_path, resource
import base64, hashlib, json, time
from pathlib import Path
from pose_memory import prepare_request, parse_pose, PROMPT
ROLES = ('TARGET_REFERENCE', 'CURRENT_RGB', 'SELECTED_MASK_OVERLAY')

def wire_body(request, provider=None):
    rebuilt = prepare_request(request['binding']['object_key'], request['images'])
    if rebuilt['sha256'] != request['sha256']:
        raise ValueError('image/request binding drift')
    content = [{'type': 'text', 'text': 'Target object: ' + request['binding']['object_key']}]
    for role in ROLES:
        data = request['images'][role]
        if data.startswith(bytes.fromhex('89504e470d0a1a0a')):
            mime = 'image/png'
        elif data.startswith(bytes.fromhex('ffd8')):
            mime = 'image/jpeg'
        else:
            raise ValueError('unsupported image bytes')
        content.extend([{'type': 'text', 'text': role}, {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(data).decode()}}])
    body = {'model': 'qwen/qwen3-vl-235b-a22b-instruct', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 700, 'provider': provider if provider is not None else {'only': ['alibaba'], 'allow_fallbacks': False, 'data_collection': 'deny'}, 'messages': [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': content}]}
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()

class ArchivedPoseClient:

    def __init__(self, root, send_once, *, provider=None, response_provider_aliases=None):
        self.root = Path(root)
        self.send_once = send_once
        self.provider = provider
        self.response_provider_aliases = tuple(response_provider_aliases or ('Alibaba',))

    def __call__(self, request):
        raw = wire_body(request, self.provider)
        call_id = request.get('call_id')
        archive_key = (hashlib.sha256(str(call_id).encode()).hexdigest() + '_' if call_id is not None else '') + request['sha256']
        folder = self.root / archive_key
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'request.body.bin').write_bytes(raw)
        record = {'binding_sha256': request['sha256'], 'wire_sha256': hashlib.sha256(raw).hexdigest(), 'started': time.time(), 'max_attempts': 1, 'call_id': call_id}
        (folder / 'START.json').write_text(json.dumps(record, indent=2))
        try:
            status, response = self.send_once(raw)
            (folder / 'response.body.bin').write_bytes(response)
            record.update(http_status=status, response_sha256=hashlib.sha256(response).hexdigest())
            if status != 200:
                raise ValueError('HTTP response not successful')
            body = json.loads(response)
            record.update(provider=body.get('provider'), usage=body.get('usage'))
            if body.get('provider') not in self.response_provider_aliases:
                raise ValueError('provider mismatch')
            text = body['choices'][0]['message']['content']
            (folder / 'MODEL_TEXT.txt').write_text(text)
            parse_pose(text)
            record['status'] = 'RETURNED'
            return {'request_sha256': request['sha256'], 'text': text, 'wire_sha256': record['wire_sha256'], 'response_sha256': record['response_sha256'], 'usage': record['usage'], 'archive_path': str(folder)}
        except Exception as exc:
            record.update(status='STOP_ERROR_ZERO_RETRY', error_type=type(exc).__name__)
            raise
        finally:
            (folder / 'RESULT.json').write_text(json.dumps(record, indent=2))
