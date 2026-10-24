"""Contained paths and explicit resources for an isolated simulation worker."""
from pathlib import Path
import json
import os
ROOT = Path(__file__).resolve().parent


def bound_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Runtime path escapes its prepared directory')
    return path


def required_file(relative):
    path = bound_path(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def resource(name, required=True):
    values = json.loads(Path(os.environ['CARRYON_RECOVERY_RESOURCES']).read_text())
    value = values.get(name)
    if value is None:
        if required:
            raise ValueError('Missing recovery resource: ' + name)
        return ''
    return str(value)
