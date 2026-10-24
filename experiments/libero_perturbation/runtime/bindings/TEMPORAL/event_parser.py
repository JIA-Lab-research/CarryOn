import json
STATES = {'VISIBLE_GRASP_FAILURE', 'NORMAL_PROGRESS', 'INSUFFICIENT_EVIDENCE'}

def parse(text, frame_ids):
    x = json.loads(text)
    if not isinstance(x, dict):
        raise ValueError('object required')
    required = {'current_grasp_state', 'current_evidence', 'evidence_frame_ids', 'earlier_failure_visible', 'uncertainty'}
    if set(x) != required:
        raise ValueError('schema keys')
    if x['current_grasp_state'] not in STATES:
        raise ValueError('enum')
    if not isinstance(x['current_evidence'], str):
        raise ValueError('evidence type')
    ids = x['evidence_frame_ids']
    if not isinstance(ids, list) or any((not isinstance(i, str) or i not in frame_ids for i in ids)):
        raise ValueError('frame ids')
    if x['earlier_failure_visible'] is not None and type(x['earlier_failure_visible']) is not bool:
        raise ValueError('nullable bool')
    if x['uncertainty'] is not None and (not isinstance(x['uncertainty'], str)):
        raise ValueError('nullable string')
    return x

def control(x, event_present):
    request = x['current_grasp_state'] == 'VISIBLE_GRASP_FAILURE' or x['earlier_failure_visible'] is True
    return {'classification_requests_takeover': request, 'control_candidate': 'REQUEST_TAKEOVER_SUBJECT_TO_EXISTING_MIN6_AND_LIFECYCLE' if request else 'CONTINUE_OBSERVING', 'event_present': event_present, 'online_triggered': False, 'min6_evaluated': False, 'note': 'non-event controls are classifier checks, not event dispatches'}
