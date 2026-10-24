"""Download model files from the configured repository revision or URL."""
from __future__ import annotations
import os
from pathlib import Path, PurePosixPath
import re
import urllib.parse
import urllib.request
from .configuration import validate_public_config
from .files import read_json, bound_cached_file


def _relative(value):
    if not isinstance(value, str) or not value or chr(92) in value:
        raise ValueError("Model paths must be nonempty POSIX relative paths")
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or ':' in value or path.as_posix() != value:
        raise ValueError("Model path is not contained: " + value)
    return value


def read_spec(value):
    doc = dict(value) if isinstance(value, dict) else read_json(value)
    validate_public_config(doc)
    repo, revision = doc.get("repo_id"), doc.get("revision")
    if repo and (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                 or not re.fullmatch(r"[0-9a-f]{40}", str(revision))):
        raise ValueError("Hugging Face resources require a repository and full commit revision")
    if not isinstance(doc.get("files"), list) or not doc["files"]:
        raise ValueError("Model specification needs a nonempty file list")
    seen = set()
    for row in doc["files"]:
        relative = _relative(row["path"])
        if relative in seen:
            raise ValueError("Duplicate model path: " + relative)
        seen.add(relative)
        _relative(row.get("source_path", relative))
        if row.get("url"):
            url = urllib.parse.urlsplit(row["url"])
            if url.scheme != "https" or not url.hostname:
                raise ValueError("Model URLs require HTTPS")
        elif not repo:
            raise ValueError("Model file needs an HTTPS URL or a pinned Hugging Face repository")
    return doc


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def download(manifest, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(_Redirect())
    for row in manifest["files"]:
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        source = row.get("source_path", row["path"])
        url = row.get("url") or ("https://huggingface.co/" + manifest["repo_id"] + "/resolve/"
                                + manifest["revision"] + "/" + urllib.parse.quote(source, safe="/"))
        request = urllib.request.Request(url)
        if urllib.parse.urlsplit(url).hostname == "huggingface.co" and os.environ.get("HF_TOKEN"):
            request.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
        partial = target.with_name(target.name + ".part")
        with opener.open(request, timeout=120) as response, partial.open("xb") as output:
            expected = getattr(response, "headers", {}).get("Content-Length")
            received = 0
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(chunk)
                received += len(chunk)
            if expected is not None and received != int(expected):
                raise OSError("Incomplete model download: " + row["path"])
        partial.rename(target)
    return [row["path"] for row in manifest["files"]]


def copy_from_cache(manifest, source, destination):
    """Copy required model files for consumers that reject cache symlinks."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for row in manifest["files"]:
        original = bound_cached_file(row, source)
        target = destination / _relative(row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as input_file, target.open("xb") as output:
            for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
                output.write(chunk)
    return [row["path"] for row in manifest["files"]]
