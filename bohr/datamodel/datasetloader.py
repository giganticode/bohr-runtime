from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dask.dataframe import DataFrame

from bohr.datamodel.artifact import ArtifactType
from bohr.datamodel.artifactmapper import ArtifactMapperSubclass, DummyMapper
from bohr.util.paths import RelativePath


@dataclass
class DatasetLoader(ABC):
    path_preprocessed: RelativePath
    mapper: ArtifactMapperSubclass = DummyMapper()

    @property
    def artifact_type(self) -> ArtifactType:
        return self.mapper.artifact_type

    @abstractmethod
    def load(self, n_datapoints: Optional[int] = None) -> DataFrame:
        pass

    @abstractmethod
    def get_extra_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def is_column_present(self, column: str) -> bool:
        pass
