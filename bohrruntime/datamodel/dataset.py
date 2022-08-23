from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from bohrapi.core import Artifact, ArtifactType
from fs.base import FS
from jsonlines import jsonlines

DatapointList = Union[List[Artifact], List[Tuple[Artifact, Artifact]]]


@dataclass(frozen=True)
class Dataset(ABC):
    id: str
    heuristic_input_artifact_type: ArtifactType
    query: Optional[Dict] = field(compare=False, default=None)
    projection: Optional[Dict] = field(compare=False, default=None)
    n_datapoints: Optional[int] = None
    path: Optional[str] = None

    def __lt__(self, other):
        if not isinstance(other, Dataset):
            raise ValueError(
                f"Cannot compare {Dataset.__name__} with {type(other).__name__}"
            )

        return self.id < other.id

    def get_n_datapoints(self, cached_datasets_fs: FS) -> int:
        return len(self.load_artifacts(cached_datasets_fs, self.n_datapoints))

    def load_artifacts(
        self, cached_datasets_fs: FS, n_datapoints: Optional[int] = None
    ) -> DatapointList:
        path = f"{self.id}.jsonl"
        if not cached_datasets_fs.exists(path):
            raise AssertionError(
                f"Dataset {self.id} not found in the filesystem. \n"
                f"It should have been loaded previously as a part of a pipeline stage."
            )
        artifact_list = []
        with cached_datasets_fs.open(path) as f:
            reader = jsonlines.Reader(f)
            for artifact in reader:
                artifact_list.append(self.heuristic_input_artifact_type(artifact))
                if len(artifact_list) == n_datapoints:
                    break
            reader.close()
        return artifact_list
