import inspect
import os
import pathlib
from typing import Dict, List, Optional, Set, Type

from bohrapi.core import ArtifactType, HeuristicObj

from bohrruntime.core import is_heuristic_file
from bohrruntime.util.paths import AbsolutePath, RelativePath, relative_to_safe


def load_all_heuristics(
    artifact_type: Type,
    heuristics_root: AbsolutePath,
    limited_to_modules: Optional[Set[str]] = None,
) -> Dict[str, List[HeuristicObj]]:
    modules: Dict[str, List[HeuristicObj]] = {}
    for path in get_heuristic_files(heuristics_root):
        path: RelativePath = relative_to_safe(path, heuristics_root.parent)
        heuristic_module_path = ".".join(str(path).replace("/", ".").split(".")[:-1])
        if limited_to_modules is None or heuristic_module_path in limited_to_modules:
            hs = load_heuristics_from_file(heuristic_module_path, artifact_type)
            if len(hs) > 0:
                modules[heuristic_module_path] = hs
    return modules


def get_heuristic_files(
    path: AbsolutePath, top_artifact: Optional[ArtifactType] = None
) -> List[AbsolutePath]:
    heuristic_files = []
    if path.is_file() and is_heuristic_file(path):
        heuristic_files.append(path)
    else:
        for root, dirs, files in os.walk(path):
            for file in files:
                path = AbsolutePath(pathlib.Path(os.path.join(root, file)))
                if is_heuristic_file(path):
                    heuristic_files.append(path)
    if top_artifact is not None:
        res = []
        for file in heuristic_files:
            if load_heuristics_from_file(file, top_artifact):
                res.append(file)
        heuristic_files = res
    if len(heuristic_files) == 0:
        raise RuntimeError(f"No heuristic groups are found at path: {path}.")
    return sorted(heuristic_files)


def load_heuristics_from_file(
    heuristic_file: AbsolutePath, artifact_type: Optional[Type] = None
) -> List[HeuristicObj]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("heuristic.module", heuristic_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def is_heuristic_of_needed_type(obj, artifact_type):
        return isinstance(obj, HeuristicObj) and (
            obj.artifact_type_applied_to == artifact_type or artifact_type is None
        )

    heuristics: List[HeuristicObj] = []
    heuristics.extend(
        [
            obj
            for name, obj in inspect.getmembers(module)
            if is_heuristic_of_needed_type(obj, artifact_type)
        ]
    )
    for name, obj in inspect.getmembers(module):
        if (
            isinstance(obj, list)
            and len(obj) > 0
            and (
                is_heuristic_of_needed_type(obj[0], artifact_type)
                or artifact_type is None
            )
        ):
            heuristics.extend(obj)
    check_names_unique(heuristics)
    return heuristics


def load_heuristic_by_name(
    name: str, artifact_type: Type, heuristics_path: AbsolutePath
) -> HeuristicObj:
    for hs in load_all_heuristics(artifact_type, heuristics_path).values():
        for h in hs:
            if h.func.__name__ == name:
                return h
    raise ValueError(f"Heuristic {name} does not exist")


def check_names_unique(heuristics: List[HeuristicObj]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)
