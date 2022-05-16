import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from bohrlabels.core import Label, Labels, LabelSet
from jsonlines import jsonlines
from pymongo import MongoClient
from snorkel.labeling import LabelingFunction
from snorkel.preprocess import BasePreprocessor
from tqdm import tqdm

from bohrruntime.dataset import Dataset
from bohrruntime.fs import find_project_root

logger = logging.getLogger(__name__)


HeuristicFunction = Callable[..., Optional[Labels]]


from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar

from bohrapi.core import ArtifactType, HeuristicObj
from snorkel.map import BaseMapper
from snorkel.types import DataPoint

BOHR_ORGANIZATION = "giganticode"
BOHR_REPO_NAME = "bohr"
BOHR_REMOTE_URL = f"git@github.com:{BOHR_ORGANIZATION}/{BOHR_REPO_NAME}"


class ArtifactMapper(BaseMapper, ABC):
    def __init__(self, artifact_type: ArtifactType):
        super().__init__(self._get_name(artifact_type), [], memoize=True)
        self.artifact_type = artifact_type

    def _get_name(self, artifact_type: Optional[ArtifactType] = None) -> str:
        return f"{artifact_type.__name__}Mapper"

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        if isinstance(x, tuple):
            raise AssertionError()
        return self.artifact_type(x)


ArtifactMapperSubclass = TypeVar("ArtifactMapperSubclass", bound="ArtifactMapper")
MapperType = Type[ArtifactMapperSubclass]


def apply_heuristic_and_convert_to_snorkel_label(
    heuristic: HeuristicObj, *args, **kwargs
) -> int:
    return to_snorkel_label(heuristic(*args, **kwargs))


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


def to_labeling_function(h: HeuristicObj) -> SnorkelLabelingFunction:
    """
    >>> from bohrapi.core import Heuristic, Artifact
    >>> from enum import auto
    >>> class TestArtifact(Artifact): pass
    >>> class TestLabel(Label): Test = auto()

    >>> @Heuristic(TestArtifact)
    ... def heuristic(artifact: TestArtifact) -> Optional[Labels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic)
    >>> a = TestArtifact({'value': 0})
    >>> lf(a)
    1

    >>> @Heuristic(TestArtifact)
    ... def heuristic2(artifact) -> Optional[Labels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic2)
    >>> lf(3)
    Traceback (most recent call last):
    ...
    TypeError: Heuristic heuristic2 can only be applied to TestArtifact object, not int

    >>> @Heuristic(TestArtifact)
    ... def heuristic3(artifact) -> Optional[Labels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic3)
    >>> lf((3,8))
    Traceback (most recent call last):
    ...
    TypeError: Expected artifact of type TestArtifact, got tuple

    >>> @Heuristic(TestArtifact, TestArtifact)
    ... def heuristic4(artifact) -> Optional[Labels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic4)
    >>> lf(3)
    Traceback (most recent call last):
    ...
    TypeError: Heuristic heuristic4 accepts only tuple of two artifacts

    >>> @Heuristic(TestArtifact, TestArtifact)
    ... def heuristic5(artifact) -> Optional[Labels]:
    ...     return TestLabel.Test
    >>> lf = to_labeling_function(heuristic5)
    >>> lf((3,5))
    Traceback (most recent call last):
    ...
    TypeError: Heuristic heuristic5 can only be applied to TestArtifact and TestArtifact

    >>> @Heuristic(TestArtifact, TestArtifact)
    ... def heuristic6(artifact: TestArtifact) -> Optional[Labels]:
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


def to_snorkel_label(labels) -> int:
    if labels is None:
        return -1
    if isinstance(labels, LabelSet):
        raise AssertionError()
    snorkel_label = labels.value
    return snorkel_label


class ArtifactMock:
    """
    >>> mock = ArtifactMock()
    >>> _ = mock['a']['b']
    >>> _ = mock['c']
    >>> mock
    {'a': {'b': 1}, 'c': 1}

    """

    def __init__(self):
        self.projection: Union[Dict, int] = 1

    def __getattr__(self, attr):
        # only called when self.attr doesn't exist
        raise KeyError(
            f"Tried to access property {attr} of artifact. Artifacts should be accessed as dictionaries."
        )

    def __getitem__(self, item):
        if self.projection == 1:
            self.projection = {}
        if item not in self.projection:
            self.projection[item] = ArtifactMock()
        return self.projection[item]

    def __repr__(self):
        return f"{self.projection}"


db = MongoClient("mongodb://read-only-user:123@10.10.20.160:27017")["commit_explorer"]


def query_dataset(
    name: str,
    match: Dict,
    projection: Optional[Dict] = None,
    n_datapoints: Optional[int] = None,
    lookup: Optional[Dict] = None,
    save_to: Optional[str] = None,
):
    if save_to is None:
        raise ValueError("save_to has to be specified")
    query = [{"$match": match}]
    if lookup is not None:
        query.append({"$lookup": lookup})
    if projection is not None:
        query.append({"$project": projection})
    query.append({"$sort": {"_id": 1}})
    if n_datapoints is not None:
        query.append({"$limit": n_datapoints})

    artifacts = db.commits.aggregate(query, allowDiskUse=True)
    path = f"{save_to}/{name}.jsonl"
    with jsonlines.open(path, "w") as writer:
        for artifact in tqdm(artifacts):
            writer.write(artifact)
    with open(f"{path}.metadata.json", "w") as f:
        json.dump(
            {
                "match": match,
                "projection": projection,
                "lookup": lookup,
                "limit": n_datapoints,
            },
            f,
        )


def load_dataset_from_explorer(dataset: Dataset) -> None:
    print(f"Loading dataset: {dataset.id}")
    # save_to = str(get_path_to_file(dataset, projection))
    save_to = find_project_root() / "cached-datasets"
    if not save_to.exists():
        save_to.mkdir()
    if dataset.query is None:
        query_dataset(
            dataset.id,
            {dataset.id: {"$exists": True}},
            dataset.projection,
            dataset.n_datapoints,
            lookup={
                "from": "issues",
                "localField": "links.bohr.issues",
                "foreignField": "_id",
                "as": "issues",
            },
            save_to=str(save_to),
        )
    else:
        query_dataset(
            dataset.id,
            dataset.query,
            dataset.projection,
            dataset.n_datapoints,
            lookup={
                "from": "issues",
                "localField": "links.bohr.issues",
                "foreignField": "_id",
                "as": "issues",
            },
            save_to=str(save_to),
        )

    print(f"Dataset loaded: {dataset.id}, and save to {save_to}")


def get_projection(heuristics: List[HeuristicObj]) -> Dict:
    mock = ArtifactMock()
    for heuristic in heuristics:
        heuristic.non_safe_func(mock, **heuristic.resources)
    return mock.projection


class Model(ABC):
    @abstractmethod
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class BohrLabelModel(Model):
    def __init__(self, label_model, label_matrix, tie_break_policy):
        self.label_model = label_model
        self.label_matrix = label_matrix
        self.tie_break_policy = tie_break_policy

    def predict(self) -> Tuple[List[int], List[float]]:
        Y_pred, Y_prob = self.label_model.predict(
            self.label_matrix, return_probs=True, tie_break_policy=self.tie_break_policy
        )
        return Y_pred, Y_prob


class RandomModel(Model):
    def __init__(self, n: int):
        self.n = n

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        Y_pred = [a for _ in range(self.n // 2) for a in [0, 1]]
        if self.n % 2 == 1:
            Y_pred.append(0)
        Y_prob = [[1.0, 0.0] for _ in range(self.n)]
        return np.array(Y_pred), np.array(Y_prob)


class ZeroModel(Model):
    def __init__(self, n: int):
        self.n = n

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        Y_pred = [0 for _ in range(self.n)]
        Y_prob = [[1.0, 0.0] for _ in range(self.n)]
        return np.array(Y_pred), np.array(Y_prob)
