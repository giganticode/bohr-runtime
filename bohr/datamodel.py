import functools
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Type, Union

import pandas as pd
from dask.dataframe import DataFrame
from snorkel.map import BaseMapper
from snorkel.types import DataPoint

from bohr.artifacts.core import Artifact
from bohr.labels.labelset import Label, Labels
from bohr.nlp_utils import camel_case_to_snake_case

logger = logging.getLogger(__name__)


ArtifactDependencies = Dict[str, Union[Artifact, List[Artifact]]]


class ArtifactMapper(BaseMapper, ABC):
    def __init__(self, artifact_type: Type, keys: List[str]):
        name = f"{artifact_type.__name__}Mapper"
        super().__init__(name, [], memoize=False)
        self.artifact_type = artifact_type
        self.keys = keys
        self._linkers = None

    @property
    def linkers(self) -> List["DatasetLinker"]:
        if self._linkers is None:
            raise AssertionError("Linkers have not been initialized yet.")
        return self._linkers

    @linkers.setter
    def linkers(self, linkers: List["DatasetLinker"]):
        self._linkers = linkers

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        return self.cached_map(x)

    def cached_map(self, x: DataPoint) -> Optional[Artifact]:
        key = tuple([getattr(x, key) for key in self.keys])
        cache = type(self).cache
        if key in cache:
            return cache[key]

        dependencies = ArtifactMapper.load_dependent_artifacts(self.linkers, key)
        artifact = self.map(x, dependencies)
        cache[key] = artifact

        return artifact

    def get_artifact(self) -> Type:
        return self.artifact_type

    @abstractmethod
    def map(
        self, x: DataPoint, dependencies: ArtifactDependencies
    ) -> Optional[DataPoint]:
        pass

    @staticmethod
    def load_dependent_artifacts(
        dataset_linkers: List["DatasetLinker"], index
    ) -> Dict[str, Union[Artifact, List[Artifact]]]:
        result: Dict[str, Union[Artifact, List[Artifact]]] = {}
        for dataset_linker in dataset_linkers:
            name = (
                camel_case_to_snake_case(
                    dataset_linker.to.dataloader.get_artifact().__name__
                )
                + "s"
            )
            df = dataset_linker.get_resources_from_file(index)

            result[name] = [
                dataset_linker.to.dataloader.get_mapper().cached_map(issue)
                for issue in df.itertuples(index=False)
            ]
        return result


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
    description: Optional[str]
    path_preprocessed: Path
    path_dist: Path
    dataloader: DatasetLoader
    test_set: bool
    preprocessor: str

    def get_artifact(self):
        return self.dataloader.get_artifact()

    def load(self):
        return self.dataloader.load()

    def get_linked_datasets(self) -> List["Dataset"]:
        return list(map(lambda l: l.to, self.dataloader.get_mapper().linkers))


class DatasetLinker(ABC):
    def __init__(self, from_: Dataset, to: Dataset, link_file: Path):
        self.from_ = from_
        self.to = to
        self.link_file = link_file

    @abstractmethod
    def get_resources(self) -> pd.DataFrame:
        pass

    def get_resources_from_file(self, index) -> pd.DataFrame:
        try:
            return self.get_resources.loc[[index]]
        except KeyError:
            return pd.DataFrame()


@dataclass(frozen=True)
class Task:
    name: str
    description: Optional[str]
    top_artifact: Type
    labels: List[str]
    train_datasets: Dict[str, Dataset]
    test_datasets: Dict[str, Dataset]
    label_column_name: str
    heuristic_groups: List[str]

    @property
    def datasets(self) -> Dict[str, Dataset]:
        return {**self.train_datasets, **self.test_datasets}

    @property
    def all_affected_datasets(self) -> Dict[str, Dataset]:
        total = self.datasets
        for dataset_name, dataset in self.datasets.items():
            total.update({d.name: d for d in dataset.get_linked_datasets()})
        return total

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
