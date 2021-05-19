import functools
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from snorkel.map import BaseMapper
from snorkel.types import DataPoint

from bohr.datamodel.artifact import Artifact, ArtifactProxy, ArtifactType
from bohr.util.nlp import camel_case_to_snake_case


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


ArtifactMapperSubclass = TypeVar("ArtifactMapperSubclass", bound="ArtifactMapper")
MapperType = Type[ArtifactMapperSubclass]
