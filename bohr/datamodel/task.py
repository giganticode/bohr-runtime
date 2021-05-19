from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Type

import jsons

from bohr.collection.artifacts import artifact_map
from bohr.datamodel.dataset import Dataset, get_all_linked_datasets
from bohr.datamodel.heuristic import get_heuristic_module_list
from bohr.util.paths import AbsolutePath, RelativePath, load_class_by_full_path


@dataclass(frozen=True)
class Task:
    name: str
    author: str
    description: Optional[str]
    top_artifact: Type
    labels: List[str]
    _train_datasets: Dict[str, Dataset]
    _test_datasets: Dict[str, Dataset]
    label_column_name: str
    heuristic_groups: List[str]

    @property
    def train_datasets(self) -> Dict[str, Dataset]:
        return self._train_datasets

    @property
    def test_datasets(self) -> Dict[str, Dataset]:
        return self._test_datasets

    def add_dataset(self, dataset: Dataset, is_test: bool) -> None:
        dct = self._train_datasets if not is_test else self._test_datasets
        self.check_artifact_right(dataset)
        dct[dataset.name] = dataset

    def serealize(self, **kwargs) -> Dict[str, Any]:
        return {
            "description": self.description,
            "top_artifact": ".".join(
                [self.top_artifact.__module__, self.top_artifact.__name__]
            ),
            "label_categories": self.labels,
            "test_datasets": sorted(self.test_datasets.keys()),
            "train_datasets": sorted(self.train_datasets.keys()),
            "label_column_name": self.label_column_name,
        }

    @property
    def datasets(self) -> Dict[str, Dataset]:
        return {**self.train_datasets, **self.test_datasets}

    @property
    def all_affected_datasets(self) -> Dict[str, Dataset]:
        return get_all_linked_datasets(self.datasets)

    def _datapaths(self, datasets: Iterable[Dataset]) -> List[RelativePath]:
        return [dataset.path_preprocessed for dataset in datasets]

    @property
    def datapaths(self) -> List[RelativePath]:
        return self.train_datapaths + self.test_datapaths

    @property
    def train_datapaths(self) -> List[RelativePath]:
        return self._datapaths(self.train_datasets.values())

    @property
    def test_datapaths(self) -> List[RelativePath]:
        return self._datapaths(self.test_datasets.values())

    def check_artifact_right(self, dataset: Dataset) -> None:
        if dataset.artifact_type != self.top_artifact:
            raise ValueError(
                f"Dataset {dataset.name} is a dataset of {dataset.artifact_type}, "
                f"but this task works on {self.top_artifact}"
            )

    def __post_init__(self):
        for dataset in self.datasets.values():
            self.check_artifact_right(dataset)


def deserialize_task(
    dct: Dict[str, Any],
    cls,
    task_name: str,
    heuristic_path: AbsolutePath,
    datasets: Dict[str, Dataset],
    **kwargs,
) -> "Task":
    # """
    # >>> jsons.loads('{"top_artifact": "artifacts.commit.Commit", "test_dataset_names": [], "train_dataset_names": []}', Task, project_root='/', task_name="x")
    # """
    test_datasets = {
        dataset_name: datasets[dataset_name] for dataset_name in dct["test_datasets"]
    }
    train_datasets = {
        dataset_name: datasets[dataset_name] for dataset_name in dct["train_datasets"]
    }
    try:
        artifact = artifact_map[dct["top_artifact"]]
    except KeyError:
        artifact = load_class_by_full_path(dct["top_artifact"])
    heuristic_groups = get_heuristic_module_list(artifact, heuristic_path)
    return Task(
        task_name,
        dct["author"] if "author" in dct else None,
        dct["description"] if "description" in dct else "",
        artifact,
        dct["label_categories"],
        _train_datasets=train_datasets,
        _test_datasets=test_datasets,
        label_column_name=dct["label_column_name"],
        heuristic_groups=heuristic_groups,
    )


jsons.set_deserializer(deserialize_task, Task)
jsons.set_serializer(Task.serealize, Task)
