"""Prepare recorded-scene inputs for execution."""
from pathlib import Path
import shutil
from ..files import read_json,write_json,inside,bound_file
from ..safety.benchmark import asset_reader


def prepare_recorded(assets,manifest,protocol,output):
    output=Path(output)
    rows=read_json(manifest)["files"]
    required={row["path"]:row for row in rows}
    if len(required)!=len(rows):raise ValueError("Duplicate recorded scene asset")
    bound_file(protocol)
    output.mkdir(parents=True,exist_ok=False)
    with asset_reader(assets,required) as read:
        for relative,row in required.items():
            target=inside(output/"scenes",relative);target.parent.mkdir(parents=True,exist_ok=True)
            with read(relative) as src,target.open("xb") as dst:shutil.copyfileobj(src,dst)
    shutil.copyfile(protocol["path"],output/"protocol.py")
    receipt={"files":{str(p.relative_to(output)):None for p in sorted(output.rglob("*")) if p.is_file()},
        "scene_files":len(rows)}
    write_json(output/"RECORDED_MANIFEST.json",receipt)
    return receipt
