import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
from dask.dataframe import DataFrame
from pandas.core.indexing import IndexingError
from snorkel.map import BaseMapper
from snorkel.types import DataPoint

from bohr.artifacts.core import Artifact
from bohr.labels.labelset import Label, Labels
from bohr.nlp_utils import camel_case_to_snake_case

logger = logging.getLogger(__name__)


ArtifactDependencies = Dict[str, Union[Artifact, List[Artifact]]]
ArtifactSubclass = TypeVar("ArtifactSubclass", bound=Artifact)
ArtifactType = Type[ArtifactSubclass]
ArtifactMapperSubclass = TypeVar("ArtifactMapperSubclass", bound="ArtifactMapper")
MapperType = Type[ArtifactMapperSubclass]


class ArtifactMapper(BaseMapper, ABC):
    def __init__(
        self,
        artifact_type: ArtifactType,
        primary_key: Union[str, Tuple[str, ...]] = "id",
        foreign_key: Union[str, Tuple[str, ...], None] = None,
    ):
        super().__init__(self.get_name(artifact_type), [], memoize=False)
        self.artifact_type = artifact_type
        self.primary_key = primary_key
        self.foreign_key = foreign_key
        self._linkers = None

    def get_name(self, artifact_type: Optional[ArtifactType] = None) -> str:
        return f"{artifact_type.__name__}Mapper"

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
        # TODO use snorkels cache?
        try:
            key = x.name
            cache = type(self).cache
            if key in cache:
                return cache[key]

            dependencies = ArtifactMapper.load_dependent_artifacts(self.linkers, key)
            artifact = self.map(x, dependencies)
            cache[key] = artifact

            return artifact
        except AttributeError as ex:
            raise AttributeError(f"Datapoint:\n {x}, \n\nprimary_key: {x.name}") from ex

    @abstractmethod
    def map(
        self, x: DataPoint, dependencies: ArtifactDependencies
    ) -> Optional[DataPoint]:
        pass

    @staticmethod
    def load_dependent_artifacts(
        dataset_linkers: List["DatasetLinker"], index: Tuple[str]
    ) -> Dict[str, Union[Artifact, List[Artifact]]]:
        result: Dict[str, Union[Artifact, List[Artifact]]] = {}
        for dataset_linker in dataset_linkers:
            name = (
                camel_case_to_snake_case(dataset_linker.to.artifact_type.__name__) + "s"
            )
            df = dataset_linker.get_dependency_by_key(index)

            lst = [
                dataset_linker.to.dataloader.mapper.cached_map(issue[1])
                for issue in df.iterrows()
            ]

            result[name] = lst
        return result


class DummyMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(None)

    def map(
        self, x: DataPoint, dependencies: ArtifactDependencies
    ) -> Optional[DataPoint]:
        raise NotImplementedError()

    def get_name(self, artifact_type: Optional[ArtifactType] = None) -> str:
        return "DummyMapper"


@dataclass
class DatasetLoader(ABC):
    path_preprocessed: Path
    mapper: ArtifactMapperSubclass = DummyMapper()

    @property
    def artifact_type(self) -> ArtifactType:
        return self.mapper.artifact_type

    @abstractmethod
    def load(self) -> DataFrame:
        pass


@dataclass
class Dataset(ABC):
    name: str
    description: Optional[str]
    path_preprocessed: Path
    path_dist: Path
    dataloader: DatasetLoader
    test_set: bool
    preprocessor: str

    def load(self):
        return self.dataloader.load()

    def get_linked_datasets(self) -> List["Dataset"]:
        return list(map(lambda l: l.to, self.mapper.linkers))

    @property
    def mapper(self) -> ArtifactMapperSubclass:
        return self.dataloader.mapper

    @property
    def primary_key(self) -> List[str]:
        return self.mapper.primary_key

    @property
    def foreign_key(self):
        return self.mapper.foreign_key

    @property
    def artifact_type(self) -> ArtifactType:
        return self.dataloader.artifact_type

    def __str__(self):
        return f"name: {self.name}, path: {self.path_preprocessed}"


class DatasetLinker(ABC):
    def __init__(self, from_: Dataset, to: Dataset, link: Optional[Dataset] = None):
        self.from_ = from_
        self.link = link
        self.to = to
        self.dependent_dataframe = None

    def finish_setup(self):
        self.dependent_dataframe = self.calc_dependent_dataframe()

    def __str__(self):
        return f"{self.from_} -> {self.to}, linker: {self.link}"

    def calc_dependent_dataframe(self):
        artifact_type_name = self.to.artifact_type.__name__
        logger.debug(f"Reading {artifact_type_name}s ... ")
        to_df = self.to.load()
        logger.debug(
            f"Index: {list(to_df.index.names)}, "
            f"columns: {list(to_df.columns)}, "
            f"n_rows: {len(to_df.index)}"
        )
        if self.link is None and self.to.foreign_key is None:
            raise ValueError(
                f"Linker: {self}.\n"
                f"Either linking dataset has to be defined "
                f"or destination dataset has to have foreign key defined, "
                f"however its foreign key is {self.to.foreign_key}"
            )
        if self.link is not None:
            if self.link.foreign_key is None:
                raise ValueError(
                    f"Foreign key is not specified for linker dataset: {self.link}"
                )
            logger.debug(f"Reading linker dataset ... ")
            link_df = self.link.load()
            logger.debug(
                f"Index: {list(link_df.index.names)}, "
                f"columns: {list(link_df.columns)}, "
                f"n_rows: {len(link_df.index)}"
            )
            logger.debug(f"Merging on {self.to.primary_key}")
            link_df = link_df.reset_index()
            res = pd.merge(link_df, to_df, on=self.to.primary_key)
            res.set_index(self.link.foreign_key, inplace=True)
            logger.debug(
                f"Merged dataset -> Index: {list(res.index.names)}, "
                f"columns: {list(res.columns)}, "
                f"n_rows: {len(res.index)}"
            )
            return res
        else:
            return to_df

    def get_dependency_by_key(self, key: Tuple[str]) -> pd.DataFrame:
        try:
            return self.dependent_dataframe.loc[[key]]
        except KeyError:
            return pd.DataFrame()
        except IndexingError as ex:
            raise AssertionError(
                f"Dataframe:\n{self.dependent_dataframe}\n\n"
                f"Key: {key}\nIndex: {self.dependent_dataframe.index}"
            ) from ex


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
            if dataset.artifact_type != self.top_artifact:
                raise ValueError(
                    f"Dataset {dataset} is a dataset of {dataset.artifact_type}, "
                    f"but this task works on {self.top_artifact}"
                )


class Heuristic:
    def __init__(
        self, func: Callable, artifact_type_applied_to: ArtifactType, resources=None
    ):
        self.artifact_type_applied_to = artifact_type_applied_to
        self.resources = resources
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, artifact: Artifact, *args, **kwargs) -> Label:
        return self.func(artifact, *args, **kwargs)


HeuristicFunction = Callable[..., Optional[Labels]]
