"""Load the pinned VLAct framework without importing unrelated optional models."""
import ast
import importlib
import importlib.util
from pathlib import Path
import sys


def load_vlact_framework(upstream):
    package="starVLA.model.framework"
    if package in sys.modules:
        raise RuntimeError("VLAct framework must load in a fresh policy process")
    upstream=Path(upstream).resolve()
    parent=importlib.import_module("starVLA.model")
    tools=importlib.import_module("starVLA.model.tools")
    path=upstream/"starVLA/model/framework/__init__.py"
    source=path.read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_framework")
    # Keep the upstream factory verbatim. Its package-wide eager discovery also
    # imports optional NeuroVLA dependencies, which the selected policy never uses.
    spec=importlib.util.spec_from_file_location(package,path,submodule_search_locations=[str(path.parent)])
    module=importlib.util.module_from_spec(spec)
    module.FRAMEWORK_REGISTRY=tools.FRAMEWORK_REGISTRY
    exec(compile(ast.get_source_segment(source,node),str(path),"exec"),module.__dict__)
    sys.modules[package]=module;setattr(parent,"framework",module)
    importlib.import_module(package+".QwenPI_v4")
    return module
