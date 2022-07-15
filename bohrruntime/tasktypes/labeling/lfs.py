import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Mapping, Optional, Union

import pandas as pd
from bohrapi.core import Heuristic, HeuristicObj
from bohrlabels.core import Label, LabelSet, OneOrManyLabels
from bohrlabels.labels import CommitLabel, MatchLabel
from snorkel.labeling import LabelingFunction, LFApplier
from snorkel.map import BaseMapper
from snorkel.preprocess import BasePreprocessor

from bohrruntime.datamodel.dataset import DatapointList
from bohrruntime.datamodel.model import HeuristicOutputs


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
    heuristics: List[HeuristicObj],
) -> List[SnorkelLabelingFunction]:
    labeling_functions = list(
        map(
            lambda h: to_labeling_function(h),
            heuristics,
        )
    )
    return labeling_functions


class HeuristicApplier(ABC):
    @abstractmethod
    def apply(
        self, heuristics: List[HeuristicObj], artifacts: DatapointList
    ) -> HeuristicOutputs:
        pass


class SnorkelHeuristicApplier(HeuristicApplier):
    def apply(
        self, heuristics: List[HeuristicObj], artifacts: DatapointList
    ) -> HeuristicOutputs:
        labeling_functions = to_labeling_functions(heuristics)
        applier = LFApplier(lfs=labeling_functions)
        applied_lf_matrix = applier.apply(artifacts)
        df = pd.DataFrame(
            applied_lf_matrix, columns=[lf.name for lf in labeling_functions]
        )
        return HeuristicOutputs(df)


logger = logging.getLogger(__name__)
HeuristicFunction = Callable[..., Optional[OneOrManyLabels]]


def apply_heuristic_and_convert_to_snorkel_label(
    heuristic: HeuristicObj, *args, **kwargs
) -> int:
    return to_snorkel_label(heuristic(*args, **kwargs))


def to_labeling_function(h: HeuristicObj) -> SnorkelLabelingFunction:
    """
    >>> from bohrapi.core import Heuristic, Artifact
    >>> from enum import auto
    >>> class TestArtifact(Artifact): pass
    >>> class TestLabel(Label): Test = auto()

    >>> @Heuristic(TestArtifact)
    ... def heuristic(artifact: TestArtifact) -> Optional[OneOrManyLabels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic)
    >>> a = TestArtifact({'value': 0})
    >>> lf(a)
    1

    >>> @Heuristic(TestArtifact)
    ... def heuristic2(artifact) -> Optional[OneOrManyLabels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic2)
    >>> lf(3)
    Traceback (most recent call last):
    ...
    TypeError: Heuristic heuristic2 can only be applied to TestArtifact object, not int

    >>> @Heuristic(TestArtifact)
    ... def heuristic3(artifact) -> Optional[OneOrManyLabels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic3)
    >>> lf((3,8))
    Traceback (most recent call last):
    ...
    TypeError: Expected artifact of type TestArtifact, got tuple

    >>> @Heuristic(TestArtifact, TestArtifact)
    ... def heuristic4(artifact) -> Optional[OneOrManyLabels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic4)
    >>> lf(3)
    Traceback (most recent call last):
    ...
    TypeError: Heuristic heuristic4 accepts only tuple of two artifacts

    >>> @Heuristic(TestArtifact, TestArtifact)
    ... def heuristic5(artifact) -> Optional[OneOrManyLabels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic5)
    >>> lf((3,5))
    Traceback (most recent call last):
    ...
    TypeError: Heuristic heuristic5 can only be applied to TestArtifact and TestArtifact

    >>> @Heuristic(TestArtifact, TestArtifact)
    ... def heuristic6(artifact: TestArtifact) -> Optional[OneOrManyLabels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic6)
    >>> a = TestArtifact({'value': 0})
    >>> lf((a, a))
    1
    """
    return SnorkelLabelingFunction(
        name=h.__name__,
        f=lambda *args, **kwargs: apply_heuristic_and_convert_to_snorkel_label(
            h, *args, **kwargs
        ),
        mapper=lambda x: x,
        resources=h.resources,
    )


def to_snorkel_label(labels: Optional[Union[Label, LabelSet]]) -> int:
    if labels is None:
        return -1
    labels = labels if isinstance(labels, LabelSet) else LabelSet.of(labels)
    if len(labels.labels) > 1 or next(iter(labels.labels)).hierarchy_class not in [
        CommitLabel,
        MatchLabel,
    ]:  # TODO
        raise NotImplementedError(f"Cannot handle {labels} yet.")
    snorkel_label = next(iter(labels.labels)).label
    return snorkel_label
