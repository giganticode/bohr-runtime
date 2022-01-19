import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
from bohrlabels.core import Label, Labels, LabelSet
from jsonlines import jsonlines
from pymongo import MongoClient
from snorkel.labeling import LabelingFunction
from snorkel.preprocess import BasePreprocessor

from bohrruntime import version
from bohrruntime.fs import find_project_root
from bohrruntime.labeling.cache import CategoryMappingCache
from bohrruntime.util.paths import AbsolutePath

logger = logging.getLogger(__name__)


HeuristicFunction = Callable[..., Optional[Labels]]


from abc import ABC
from typing import Optional, Type, TypeVar

from bohrapi.core import ArtifactType, Dataset, HeuristicObj, Task, Workspace
from snorkel.map import BaseMapper
from snorkel.types import DataPoint


class ArtifactMapper(BaseMapper, ABC):
    def __init__(self, artifact_type: ArtifactType):
        super().__init__(self._get_name(artifact_type), [], memoize=True)
        self.artifact_type = artifact_type

    def _get_name(self, artifact_type: Optional[ArtifactType] = None) -> str:
        return f"{artifact_type.__name__}Mapper"

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        return self.artifact_type(x)


ArtifactMapperSubclass = TypeVar("ArtifactMapperSubclass", bound="ArtifactMapper")
MapperType = Type[ArtifactMapperSubclass]


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
    heuristics: List[HeuristicObj], category_mapping_cache
) -> List[SnorkelLabelingFunction]:
    labeling_functions = list(
        map(
            lambda h: to_labeling_function(h, category_mapping_cache),
            heuristics,
        )
    )
    return labeling_functions


def to_labeling_function(
    h: HeuristicObj, category_mapping_cache
) -> SnorkelLabelingFunction:
    return SnorkelLabelingFunction(
        name=h.__name__,
        f=lambda *args, **kwargs: apply_heuristic_and_convert_to_snorkel_label(
            h, category_mapping_cache, *args, **kwargs
        ),
        mapper=lambda x: x,
        resources=h.resources,
    )


def to_snorkel_label(labels, category_mapping_cache_map: CategoryMappingCache) -> int:
    if labels is None:
        return -1
    label_set = labels if isinstance(labels, LabelSet) else LabelSet.of(labels)
    snorkel_label = category_mapping_cache_map[label_set]
    return snorkel_label


def load_workspace(project_root: Optional[AbsolutePath] = None) -> Workspace:
    project_root = project_root or find_project_root()
    file = project_root / "bohr.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("heuristic.module", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in inspect.getmembers(module):
        if isinstance(obj, Workspace):
            workspace = obj

            version_installed = version()
            if str(workspace.bohr_runtime_version) != version_installed:
                raise EnvironmentError(
                    f"Version of bohr framework from config: {workspace.bohr_runtime_version}. "
                    f"Version of bohr installed: {version_installed}"
                )
            return workspace
    raise ValueError(f"Object of type {Workspace.__name__} not found in bohr.py")


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

    commits = db.commits.aggregate(query)
    path = f"{save_to}/{name}.jsonl"
    with jsonlines.open(path, "w") as writer:
        writer.write_all(commits)
    with open(f"{path}.metadata.json", "w") as f:
        json.dump({"match": match, "projection": projection, "lookup": lookup}, f)


def query_dataset_with_json(name: str, json: Dict, save_to: Optional[str]):
    return query_dataset(
        name, json["match"], json["projection"], json["lookup"], save_to
    )


def load_dataset_from_explorer(
    dataset: Dataset, projection: Optional[Dict] = None
) -> None:
    print(f"Loading dataset: {dataset.id}")
    # save_to = str(get_path_to_file(dataset, projection))
    save_to = find_project_root() / "cached-datasets"
    if not save_to.exists():
        save_to.mkdir()
    if dataset.query is None:
        query_dataset(
            dataset.id,
            {dataset.id: {"$exists": True}},
            None,
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
            None,
            lookup={
                "from": "issues",
                "localField": "links.bohr.issues",
                "foreignField": "_id",
                "as": "issues",
            },
            save_to=str(save_to),
        )

    print(f"Dataset loaded: {dataset.id}, and save to {save_to}")


def get_path_to_file(
    dataset: Dataset, projection: Optional[Dict] = None
) -> AbsolutePath:
    return find_project_root() / "cached-datasets" / f"{dataset.id}.jsonl"


def load_dataset(
    dataset: Dataset,
    n_datapoints: Optional[int] = None,
    projection: Optional[Dict] = None,
) -> List[Dict]:
    path = get_path_to_file(dataset, projection)
    if not path.exists():
        raise RuntimeError(
            f"Dataset {dataset.id} should have been loaded by a dvc stage first!"
        )
    artifact_list = []
    with jsonlines.open(path, "r") as reader:
        for artifact in reader:
            artifact_list.append(dataset.top_artifact(artifact))
            if len(artifact_list) == n_datapoints:
                break
    return artifact_list


def get_projection(heuristics: List[HeuristicObj]) -> Dict:
    mock = ArtifactMock()
    for heuristic in heuristics:
        heuristic.non_safe_func(mock, **heuristic.resources)
    return mock.projection


def load_ground_truth_labels(
    task: Task, dataset: Dataset, pre_loaded_artifacts: Optional[pd.DataFrame] = None
) -> Optional[Sequence[Label]]:
    if pre_loaded_artifacts is None:
        pre_loaded_artifacts = load_dataset(dataset)
    if dataset in task.test_datasets and task.test_datasets[dataset] is not None:
        label_from_datapoint_function = task.test_datasets[dataset]
        label_series = [
            label_from_datapoint_function(artifact) for artifact in pre_loaded_artifacts
        ]
    else:
        label_series = None
    return label_series
