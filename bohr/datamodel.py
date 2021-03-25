import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Type

from dask.dataframe import DataFrame
from snorkel.map import BaseMapper

from bohr.artifacts.core import Artifact
from bohr.labels.labelset import Label, Labels

logger = logging.getLogger(__name__)


class ArtifactMapper(BaseMapper, ABC):
    @abstractmethod
    def get_artifact(self) -> Type:
        pass


@dataclass
class DatasetLoader(ABC):
    path_preprocessed: Path
    mapper: ArtifactMapper
    main_file: Optional[Path] = None

    def get_artifact(self) -> Type:
        return self.get_mapper().get_artifact()

    @abstractmethod
    def load(self) -> DataFrame:
        pass

    def get_mapper(self) -> ArtifactMapper:
        return self.mapper


@dataclass
class Dataset(ABC):
    name: str
    path_preprocessed: Path
    path_dist: Path
    dataloader: DatasetLoader
    test_set: bool
    preprocessor: str

    def get_artifact(self):
        return self.dataloader.get_artifact()

    def load(self):
        return self.dataloader.load()


@dataclass(frozen=True)
class Task:
    name: str
    top_artifact: Type
    labels: List[str]
    train_datasets: Dict[str, Dataset]
    test_datasets: Dict[str, Dataset]
    label_column_name: str
    heuristic_groups: List[str]

    @property
    def datasets(self) -> Dict[str, Dataset]:
        return {**self.train_datasets, **self.test_datasets}

    def _datapaths(self, datasets: Iterable[Dataset]) -> List[Path]:
        return [dataset.path_preprocessed for dataset in datasets]

    @property
    def datapaths(self) -> List[Path]:
        return self.train_datapaths + self.test_datapaths

    @property
    def train_datapaths(self) -> List[Path]:
        return self._datapaths(self.train_datasets.values())

    @property
    def test_datapaths(self) -> List[Path]:
        return self._datapaths(self.test_datasets.values())

    def __post_init__(self):
        for dataset in self.datasets.values():
            if dataset.get_artifact() != self.top_artifact:
                raise ValueError(
                    f"Dataset {dataset} is a dataset of {dataset.get_artifact()}, "
                    f"but this task works on {self.top_artifact}"
                )


class Heuristic:
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
