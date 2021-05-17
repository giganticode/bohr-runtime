import functools
import importlib
import inspect
import os
from pathlib import Path
from typing import Callable, List, Optional, Set, Type

from bohr.datamodel.artifact import Artifact, ArtifactType
from bohr.labeling.labelset import Label
from bohr.util.paths import AbsolutePath, RelativePath, relative_to_safe


class HeuristicObj:
    def __init__(
        self, func: Callable, artifact_type_applied_to: ArtifactType, resources=None
    ):
        self.artifact_type_applied_to = artifact_type_applied_to
        self.resources = resources
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, artifact: Artifact, *args, **kwargs) -> Label:
        return self.func(artifact, *args, **kwargs)


def get_heuristic_module_list(
    artifact_type: Type,
    heuristics_path: AbsolutePath,
    limited_to_modules: Optional[Set[str]] = None,
) -> List[str]:
    modules: List[str] = []
    for root, dirs, files in os.walk(heuristics_path):
        for file in files:
            if (
                file.startswith("_")
                or root.endswith("__pycache__")
                or not file.endswith(".py")
            ):
                continue
            path: RelativePath = relative_to_safe(
                Path(root) / file, heuristics_path.parent
            )
            heuristic_module_path = ".".join(
                str(path).replace("/", ".").split(".")[:-1]
            )
            if (
                limited_to_modules is None
                or heuristic_module_path in limited_to_modules
            ):
                hs = load_heuristics_from_module(artifact_type, heuristic_module_path)
                if len(hs) > 0:
                    modules.append(heuristic_module_path)
    return sorted(modules)


def load_heuristics_from_module(
    artifact_type: Type, full_module_path: str
) -> List[HeuristicObj]:
    def is_heuristic_of_needed_type(obj):
        return (
            isinstance(obj, HeuristicObj)
            and obj.artifact_type_applied_to == artifact_type
        )

    heuristics: List[HeuristicObj] = []
    module = importlib.import_module(full_module_path)
    heuristics.extend(
        [
            obj
            for name, obj in inspect.getmembers(module)
            if is_heuristic_of_needed_type(obj)
        ]
    )
    for name, obj in inspect.getmembers(module):
        if (
            isinstance(obj, list)
            and len(obj) > 0
            and is_heuristic_of_needed_type(obj[0])
        ):
            heuristics.extend(obj)
    check_names_unique(heuristics)
    return heuristics


def check_names_unique(heuristics: List[HeuristicObj]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)
