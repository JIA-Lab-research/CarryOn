"""Prepare the benchmark and its required object assets."""
from __future__ import annotations
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
import shutil
import stat
import zipfile
import tarfile
from ..files import bound_file, inside, read_json, write_json

@contextmanager
def asset_reader(source, required):
    source = Path(source).resolve()
    if source.is_dir():
        def read(relative):
            return inside(source, relative).open("rb")
        yield read
        return
    if tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            members=archive.getmembers();counts=Counter(m.name for m in members)
            first=next(iter(required))
            candidates=[m.name[:-len(first)] for m in members if m.name==first or m.name.endswith("/"+first)]
            if len(candidates)!=1:raise ValueError("Cannot identify one unambiguous assets root in TAR")
            prefix=candidates[0];mapped={}
            for relative in required:
                key=prefix+relative
                if counts[key]!=1:raise ValueError("Missing or duplicate asset in TAR: "+relative)
                item=archive.getmember(key)
                if not item.isfile():raise ValueError("Asset must be a regular file: "+relative)
                mapped[relative]=item
            yield lambda relative:archive.extractfile(mapped[relative])
        return
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        counts = Counter(names)
        # Locate one archive root by an exact required filename; never use
        # basename matching, and reject ambiguous or duplicate entries.
        first = next(iter(required))
        candidates = [x[:-len(first)] for x in names if x == first or x.endswith("/" + first)]
        if len(candidates) != 1:
            raise ValueError("Cannot identify one unambiguous assets root in ZIP")
        prefix = candidates[0]
        mapped = {}
        for relative in required:
            key = prefix + relative
            if counts[key] != 1:
                raise ValueError("Missing or duplicate asset in ZIP: " + relative)
            item = archive.getinfo(key)
            if stat.S_ISLNK(item.external_attr >> 16) or item.is_dir():
                raise ValueError("Asset must be a regular file: " + relative)
            mapped[relative] = item
        yield lambda relative: archive.open(mapped[relative], "r")

def prepare(template, assets, manifest_path, output, published_root=None):
    template, output = Path(template).resolve(), Path(output).resolve()
    published_root = Path(published_root).resolve() if published_root else output
    if output.exists():
        raise FileExistsError(output)
    index = read_json(template / "SOURCE_INDEX.json")
    for record in index["files"]:
        bound_file(record, template)
    records = read_json(manifest_path)["files"]
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate asset manifest paths")
    overlays = {r["path"][len("libero/libero/assets/"):] for r in index["files"]
                if r["path"].startswith("libero/libero/assets/")}
    required = {r["path"]: r for r in records if r["path"] not in overlays}
    output.mkdir(parents=True)
    for record in index["files"]:
        target = inside(output, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template / record["path"], target)
    asset_root = output / "libero/libero/assets"
    with asset_reader(assets, required) as read:
        for relative, record in required.items():
            target = inside(asset_root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with read(relative) as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    for record in records:
        bound_file(record, asset_root)
    import yaml
    root = published_root / "libero/libero"
    config = dict(benchmark_root=str(root), bddl_files=str(root / "bddl_files"),
                  init_states=str(root / "init_files"), datasets=str(published_root / "datasets"),
                  assets=str(root / "assets"))
    cfg = output / "config/config.yaml"
    cfg.parent.mkdir()
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    receipt = {"schema": "carryon-benchmark-prepared/v1", "verified_assets": len(records),
               "files": {p.relative_to(output).as_posix(): None for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "BENCHMARK_MANIFEST.json", receipt)
    return receipt
