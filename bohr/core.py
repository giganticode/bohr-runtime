import functools
import importlib
import inspect
import logging
import os
from subprocess import CalledProcessError
from typing import Callable, List, Optional, Set, Type

from bohr.artifacts.core import Artifact
from bohr.config import Config
from bohr.datamodel import ArtifactMapper
from bohr.labels.cache import CategoryMappingCache
from bohr.labels.labelset import Label, Labels
from bohr.snorkel_util import SnorkelLabelingFunction, to_snorkel_label

KEYWORD_GROUP_SEPARATOR = "|"


logger = logging.getLogger(__name__)


class _Heuristic:
    def __init__(
        self, func: Callable, artifact_type_applied_to: Type[Artifact], resources=None
    ):
        self.artifact_type_applied_to = artifact_type_applied_to
        self.resources = resources
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, artifact: Artifact, *args, **kwargs) -> Label:
        return self.func(artifact, *args, **kwargs)


HeuristicFunction = Callable[..., Optional[Labels]]


class Heuristic:
    def __init__(self, artifact_type_applied_to: Type[Artifact]):
        self.artifact_type_applied_to = artifact_type_applied_to

    def get_artifact_safe_func(self, f: HeuristicFunction) -> HeuristicFunction:
        def func(artifact, *args, **kwargs):
            if not isinstance(artifact, self.artifact_type_applied_to):
                raise ValueError("Not right artifact")
            try:
                return f(artifact, *args, **kwargs)
            except (
                ValueError,
                KeyError,
                AttributeError,
                IndexError,
                TypeError,
                CalledProcessError,
            ):
                logger.exception(
                    "Exception thrown while applying heuristic, "
                    "skipping the heuristic for this datapoint ..."
                )
                return None

        return functools.wraps(f)(func)

    def __call__(self, f: Callable[..., Label]) -> _Heuristic:
        safe_func = self.get_artifact_safe_func(f)
        return _Heuristic(safe_func, self.artifact_type_applied_to)


def load_heuristics_from_module(
    artifact_type: Type, module_name: str, heuristic_package: str
) -> List[_Heuristic]:
    def is_heuristic_of_needed_type(obj):
        return (
            isinstance(obj, _Heuristic)
            and obj.artifact_type_applied_to == artifact_type
        )

    heuristics: List[_Heuristic] = []
    module = importlib.import_module(f"{heuristic_package}.{module_name}")
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
    return heuristics


def check_names_unique(heuristics: List[_Heuristic]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)


def load_heuristics(
    artifact_type: Type, config: Config, limited_to_modules: Optional[Set[str]] = None
) -> List[_Heuristic]:
    heuristics: List[_Heuristic] = []
    for heuristic_file in next(os.walk(config.paths.heuristics))[2]:
        heuristic_module_name = ".".join(heuristic_file.split(".")[:-1])
        if limited_to_modules is None or heuristic_module_name in limited_to_modules:
            heuristics.extend(
                load_heuristics_from_module(
                    artifact_type, heuristic_module_name, config.paths.heuristics_dir
                )
            )
    check_names_unique(heuristics)
    return heuristics


def apply_heuristic_and_convert_to_snorkel_label(
    heuristic: _Heuristic, cache: CategoryMappingCache, *args, **kwargs
) -> int:
    return to_snorkel_label(heuristic(*args, **kwargs), cache)


def to_labeling_functions(
    heuristics: List[_Heuristic], mapper: ArtifactMapper, labels: List[str]
) -> List[SnorkelLabelingFunction]:
    category_mapping_cache = CategoryMappingCache(labels, maxsize=10000)
    labeling_functions = list(
        map(
            lambda h: SnorkelLabelingFunction(
                name=h.__name__,
                f=lambda *args, **kwargs: apply_heuristic_and_convert_to_snorkel_label(
                    h, category_mapping_cache, *args, **kwargs
                ),
                mapper=mapper,
                resources=h.resources,
            ),
            heuristics,
        )
    )
    return labeling_functions
