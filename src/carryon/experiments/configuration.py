"""Keep credentials out of experiment configurations and saved outputs."""
from __future__ import annotations
import re
from urllib.parse import urlsplit, parse_qsl

_SENSITIVE = re.compile(r"(^|_)(api_key|access_token|auth_token|bearer_token|password|secret|authorization|hf_token|openrouter_api_key)$", re.I)

def validate_public_config(value, location="config"):
    if isinstance(value, dict):
        for key, item in value.items():
            if _SENSITIVE.search(str(key)) and item not in (None, ""):
                raise ValueError(f"{location}.{key}: credentials must be inherited from the process environment; configure only the environment variable name")
            validate_public_config(item, location + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_public_config(item, f"{location}[{index}]")
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        url = urlsplit(value)
        if url.username is not None or url.password is not None or any(_SENSITIVE.search(key) or key.lower() in ("token", "key") for key, _ in parse_qsl(url.query)):
            raise ValueError(f"{location}: URLs must not contain credentials")
