"""Prepare experiment resources from explicitly configured locations."""
from __future__ import annotations
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

from .files import bound_file, bound_cached_file, inside, new_directory, read_json, write_json
from .configuration import validate_public_config
from . import model_resources
from .safety.materialize import materialize
from .safety.benchmark import prepare as prepare_benchmark

SUITES = ("libero_perturbation", "libero_safety", "libero_recovery", "robolab")


def read_config(path):
    import yaml
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            def replacement(match):
                name = match.group(1)
                if name not in os.environ:
                    raise ValueError("Required environment variable is unset: " + name)
                return os.environ[name]
            return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replacement, value)
        return value
    if not isinstance(document, dict):
        raise ValueError("Experiment configuration must be an object")
    validate_public_config(document)
    document = expand(document)
    validate_public_config(document)
    if document.get("schema") != "carryon-experiment/v1":
        raise ValueError("Expected config schema carryon-experiment/v1")
    if not document.get("suites") or not set(document["suites"]) <= set(SUITES):
        raise ValueError("Config must select one or more supported suites")
    return document, path.parent


def inspect_resources(config, base):
    missing, resolved = [], {}
    for name, spec in config.get("resources", {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid resource identifier: " + name)
        spec = dict(spec)
        if spec.get("model_spec"):
            model = model_resources.read_spec(spec.pop("model_spec"))
            if "url" in spec:
                raise ValueError(name + ": model_spec cannot be combined with another download URL")
            spec.update(kind="model", model=model, files=[r["path"] for r in model["files"]])
            if "path" not in spec:
                resolved[name] = spec
                continue
        if ("path" in spec) == ("url" in spec):
            raise ValueError(name + " needs exactly one of path or url")
        if "url" in spec:
            if urllib.parse.urlsplit(spec["url"]).scheme != "https":
                raise ValueError(name + " needs an HTTPS URL")
            resolved[name] = dict(spec)
            continue
        source = (base / spec["path"]).expanduser().resolve()
        if not source.exists():
            missing.append(name + ": " + str(source))
        elif source.is_dir():
            if not spec.get("files"):
                missing.append(name + ": a directory resource needs its required file list")
            else:
                for relative in spec["files"]:
                    try:
                        (bound_cached_file if spec.get("kind")=="model" else bound_file)({"path": relative}, source)
                    except (OSError, ValueError) as exc:
                        missing.append(name + ": " + str(exc))
        resolved[name] = {**spec, "path": str(source)}
    return resolved, missing


def prepare(config_path, output, release_root):
    config, base = read_config(config_path)
    resources, missing = inspect_resources(config, base)
    if missing:
        raise ValueError("Missing or invalid resources:\n  " + "\n  ".join(missing))
    release_root, output = Path(release_root).resolve(), Path(output).resolve()
    with new_directory(output, inputs=[release_root]) as root:
        bindings = {}
        for name, spec in resources.items():
            if spec.get("model") and ("path" not in spec or spec.get("copy_to_cache")):
                relative = Path("models") / name
                files = (model_resources.copy_from_cache(spec["model"], spec["path"], root / relative)
                         if "path" in spec else model_resources.download(spec["model"], root / relative))
                bindings[name] = {"path":str(output / relative), "kind":"model", "files":files,
                                  "repo_id":spec["model"].get("repo_id"), "revision":spec["model"].get("revision")}
                continue
            if "url" not in spec:
                bindings[name] = spec
                continue
            target = root / "downloads" / name
            target.parent.mkdir(exist_ok=True)
            with urllib.request.urlopen(spec["url"], timeout=120) as source, target.open("xb") as dest:
                shutil.copyfileobj(source, dest)
            bindings[name] = {"path": str(output / target.relative_to(root))}
        if set(config["suites"]) & {"libero_perturbation","libero_recovery"}:
            if "libero_assets" not in resources:
                raise ValueError("Recovery requires the pinned libero_assets resource")
            spec = resources["libero_assets"]
            assets = root / "downloads/libero_assets" if "url" in spec else spec["path"]
            prepare_benchmark(release_root / "third_party/libero", assets,
                              release_root / "experiments" / ("libero_perturbation" if "libero_perturbation" in config["suites"] else "libero_recovery") / "resources/libero.json", root / "libero_benchmark",
                              published_root=output / "libero_benchmark")
        if "libero_recovery" in config["suites"]:
            from .recovery.resources import prepare_recorded
            for key in ("recorded_scenes","recorded_protocol"):
                if key not in resources:raise ValueError("Recorded Recovery requires "+key)
            scenes=resources["recorded_scenes"]
            assets=root/"downloads/recorded_scenes" if "url" in scenes else scenes["path"]
            protocol=resources["recorded_protocol"]
            path=root/"downloads/recorded_protocol" if "url" in protocol else Path(protocol["path"])
            prepare_recorded(assets,release_root/"experiments/libero_recovery/resources/recorded_scenes.json",
                {"path":str(path)},root/"recorded_scenes")
        if "libero_safety" in config["suites"]:
            materialize(release_root / "experiments/libero_safety/runtime_sources", root / "safety")
            if "safety_assets" not in resources:
                raise ValueError("libero_safety requires the pinned safety_assets resource")
            spec = resources["safety_assets"]
            assets = root / "downloads/safety_assets" if "url" in spec else spec["path"]
            prepare_benchmark(release_root / "third_party/libero_safety", assets,
                              release_root / "experiments/libero_safety/resources/benchmark.json", root / "safety_benchmark",
                              published_root=output / "safety_benchmark")
        write_json(root / "RESOURCES.json", {"schema": "carryon-prepared-resources/v1",
            "config": config, "resources": bindings,
            "libero_benchmark": "libero_benchmark" if set(config["suites"]) & {"libero_perturbation","libero_recovery"} else None, "safety_benchmark": "safety_benchmark" if "libero_safety" in config["suites"] else None, "safety_runtime": "safety" if "libero_safety" in config["suites"] else None})
    return read_json(output / "RESOURCES.json")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release-root", required=True)
    args = parser.parse_args(argv)
    result = prepare(args.config, args.output, args.release_root)
    print("Prepared", len(result["resources"]), "external resources")


if __name__ == "__main__":
    main()
