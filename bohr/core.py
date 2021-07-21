import functools
import logging
from subprocess import CalledProcessError
from typing import Any, Callable, List, Mapping, Optional, Type

from snorkel.labeling import LabelingFunction
from snorkel.map import BaseMapper
from snorkel.preprocess import BasePreprocessor

from bohr.datamodel.artifact import Artifact
from bohr.datamodel.artifactmapper import ArtifactMapper
from bohr.datamodel.heuristic import HeuristicObj
from bohr.labeling.cache import CategoryMappingCache
from bohr.labeling.labelset import Label, Labels, LabelSet

logger = logging.getLogger(__name__)

HeuristicFunction = Callable[..., Optional[Labels]]


def apply_heuristic_and_convert_to_snorkel_label(
    heuristic: HeuristicObj, cache: CategoryMappingCache, *args, **kwargs
) -> int:
    return to_snorkel_label(heuristic(*args, **kwargs), cache)


class SnorkelLabelingFunction(LabelingFunction):
    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        mapper: BaseMapper,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
    ) -> None:
        if pre is None:
            pre = []
        pre.insert(0, mapper)
        super().__init__(name, f, resources, pre=pre)


def to_labeling_functions(
    heuristics: List[HeuristicObj], mapper: ArtifactMapper, labels: List[str]
) -> List[SnorkelLabelingFunction]:
    labeling_functions = list(
        map(
            lambda h: to_labeling_function(h, mapper, labels),
            heuristics,
        )
    )
    return labeling_functions


def to_labeling_function(
    h: HeuristicObj, mapper: ArtifactMapper, labels: List[str]
) -> SnorkelLabelingFunction:
    category_mapping_cache = CategoryMappingCache(labels, maxsize=10000)
    return SnorkelLabelingFunction(
        name=h.__name__,
        f=lambda *args, **kwargs: apply_heuristic_and_convert_to_snorkel_label(
            h, category_mapping_cache, *args, **kwargs
        ),
        mapper=mapper,
        resources=h.resources,
    )


def to_snorkel_label(labels, category_mapping_cache_map: CategoryMappingCache) -> int:
    if labels is None:
        return -1
    label_set = labels if isinstance(labels, LabelSet) else LabelSet.of(labels)
    snorkel_label = category_mapping_cache_map[label_set]
    return snorkel_label


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

    def __call__(self, f: Callable[..., Label]) -> HeuristicObj:
        safe_func = self.get_artifact_safe_func(f)
        return HeuristicObj(safe_func, self.artifact_type_applied_to)
