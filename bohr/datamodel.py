import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NewType,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from dask.dataframe import DataFrame
from snorkel.map import BaseMapper
from snorkel.types import DataPoint

from bohr.artifacts.core import Artifact, ArtifactProxy
from bohr.labels.labelset import Label, Labels
from bohr.nlp_utils import camel_case_to_snake_case

logger = logging.getLogger(__name__)


ArtifactDependencies = Dict[str, Union[Artifact, List[Artifact]]]
ArtifactSubclass = TypeVar("ArtifactSubclass", bound=Artifact)
ArtifactType = Type[ArtifactSubclass]
ArtifactMapperSubclass = TypeVar("ArtifactMapperSubclass", bound="ArtifactMapper")
MapperType = Type[ArtifactMapperSubclass]
RelativePath = NewType("RelativePath", Path)
AbsolutePath = NewType("AbsolutePath", Path)

RelativeOrAbsolute = TypeVar("RelativeOrAbsolute", RelativePath, AbsolutePath)


def relative_to_safe(
    path: RelativeOrAbsolute, base_path: RelativeOrAbsolute
) -> RelativePath:
    return path.relative_to(base_path)


def concat_paths_safe(p1: RelativeOrAbsolute, p2: RelativePath) -> RelativeOrAbsolute:
    return p1 / p2


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

    @functools.cached_property
    def proxies(self) -> Dict[str, ArtifactProxy]:
        proxies = {}
        for linker in self.linkers:
            name = camel_case_to_snake_case(linker.to.artifact_type.__name__) + "s"
            proxies[name] = ArtifactProxy(
                linker.to, linker.from_.primary_key, linker.link
            )
        return proxies

    def cached_map(self, x: DataPoint) -> Optional[Artifact]:
        # TODO use snorkels cache?
        try:
            key = x.name
            cache = type(self).cache
            if key in cache:
                return cache[key]

            artifact = self.map(x)
            artifact.proxies = self.proxies
            artifact.keys = key
            cache[key] = artifact

            return artifact
        except AttributeError as ex:
            raise AttributeError(f"Datapoint:\n {x}, \n\nprimary_key: {x.name}") from ex

    @abstractmethod
    def map(self, x: DataPoint) -> Optional[DataPoint]:
        pass


class DummyMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(None)

    def map(self, x: DataPoint) -> Optional[DataPoint]:
        raise NotImplementedError()

    def get_name(self, artifact_type: Optional[ArtifactType] = None) -> str:
        return "DummyMapper"


@dataclass
class DatasetLoader(ABC):
    path_preprocessed: RelativePath
    mapper: ArtifactMapperSubclass = DummyMapper()

    @property
    def artifact_type(self) -> ArtifactType:
        return self.mapper.artifact_type

    @abstractmethod
    def load(self) -> DataFrame:
        pass

    @abstractmethod
    def get_extra_params(self) -> Dict[str, Any]:
        pass


@dataclass
class Dataset(ABC):
    name: str
    author: str
    description: Optional[str]
    path_preprocessed: RelativePath
    path_dist: RelativePath
    dataloader: DatasetLoader
    test_set: bool
    preprocessor: str

    def serealize(self, **kwargs) -> Dict[str, Any]:
        dct = {
            "author": self.author,
            "description": self.description,
            "path": self.path_dist.name,
            "path_preprocessed": str(
                relative_to_safe(self.dataloader.path_preprocessed, kwargs["data_dir"])
            ),
            "test_set": self.test_set,
            "preprocessor": self.preprocessor,
            "loader": "csv",
        }
        if type(self.mapper).__name__ != "DummyMapper":
            dct["mapper"] = ".".join(
                [type(self.mapper).__module__, type(self.mapper).__name__]
            )
        return {**dct, **self.dataloader.get_extra_params()}

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

    def __str__(self):
        return f"{self.from_} -> {self.to}, linker: {self.link}"

    def serealize(self, **kwargs) -> Dict[str, Any]:
        dct = {"from": self.from_.name, "to": self.to.name}
        if self.link:
            dct["link"] = self.link.name
        return dct


@dataclass(frozen=True)
class Task:
    name: str
    author: str
    description: Optional[str]
    top_artifact: Type
    labels: List[str]
    train_datasets: Dict[str, Dataset]
    test_datasets: Dict[str, Dataset]
    label_column_name: str
    heuristic_groups: List[str]

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
        total = self.datasets
        for dataset_name, dataset in self.datasets.items():
            total.update({d.name: d for d in dataset.get_linked_datasets()})
        return total

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
