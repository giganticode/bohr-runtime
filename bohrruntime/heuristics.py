import inspect
import os
import pathlib
from typing import Dict, List, Optional, Set, Type

from bohrapi.core import ArtifactType, HeuristicObj
from snorkel.labeling import LabelingFunction

from bohrruntime.core import to_labeling_functions
from bohrruntime.util.paths import AbsolutePath, RelativePath, relative_to_safe


def load_all_heuristics(
    artifact_type: Type,
    heuristics_root: AbsolutePath,
    limited_to_modules: Optional[Set[str]] = None,
) -> Dict[RelativePath, List[HeuristicObj]]:
    modules: Dict[RelativePath, List[HeuristicObj]] = {}
    for path in get_heuristic_files(heuristics_root):
        rel_path: RelativePath = relative_to_safe(path, heuristics_root.parent)
        heuristic_module_path = ".".join(
            str(rel_path).replace("/", ".").split(".")[:-1]
        )
        if limited_to_modules is None or heuristic_module_path in limited_to_modules:
            hs = load_heuristics_from_file(path, artifact_type)
            if len(hs) > 0:
                modules[rel_path] = hs
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
    name: str,
    artifact_type: Type,
    heuristics_path: AbsolutePath,
    return_path: bool = False,
) -> HeuristicObj:
    for path, hs in load_all_heuristics(artifact_type, heuristics_path).items():
        for h in hs:
            if h.func.__name__ == name:
                return h if not return_path else (h, path)
    raise ValueError(f"Heuristic {name} does not exist")


def check_names_unique(heuristics: List[HeuristicObj]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)


def is_heuristic_file(file: AbsolutePath) -> bool:
    """
    >>> from pathlib import Path
    >>> is_heuristic_file(Path('/home/user/heuristics/mult.py'))
    True
    >>> is_heuristic_file(Path('/home/user/heuristics/_mult.py'))
    False
    >>> is_heuristic_file(Path('/home/user/heuristics/__pycache__/mult.py'))
    False
    """
    return (
        not str(file.name).startswith("_")
        and not str(file.parent).endswith("__pycache__")
        and str(file.name).endswith(".py")
    )


def get_labeling_functions_from_path(
    heuristic_file: AbsolutePath, category_mapping_cache
) -> List[LabelingFunction]:
    heuristics = load_heuristics_from_file(heuristic_file)
    labeling_functions = to_labeling_functions(heuristics, category_mapping_cache)
    return labeling_functions