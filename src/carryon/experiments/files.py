"""Content bindings and immutable output directories shared by experiment runners."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(value))
    return sha256(path)


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path must be a file below its root: {relative}")
    return path


def bound_file(binding, root=None):
    path = Path(binding["path"])
    if root is not None:
        path = inside(root, path)
    if not path.is_file():
        raise ValueError(f"Missing required file: {path}")
    return path


def bound_cached_file(binding,root):
    """Verify a named model-cache file, including shared content-addressed links.

    Only the resolved data bytes may be outside the logical cache root. Code
    resources continue to use bound_file's strict containment checks.
    """
    relative=Path(binding["path"])
    if relative.is_absolute() or relative.drive or ".." in relative.parts or not relative.parts:
        raise ValueError("Model cache entry must have a relative contained name")
    return bound_file({"path":str(Path(root)/relative)})


@contextmanager
def new_directory(destination, *, inputs=()):
    """Publish a complete directory; never overwrite or modify an input archive.

    A sibling reservation prevents two writers racing for the same destination.
    Incomplete output is retained for diagnosis when an operation fails.
    """
    destination = Path(destination).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if (destination == source or destination.is_relative_to(source)
                or source.is_relative_to(destination)):
            raise ValueError("Input and output directories must be disjoint")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_name(destination.name + ".creating")
    fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + "-",
                                    dir=destination.parent))
    try:
        yield staging
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        reservation.unlink()
