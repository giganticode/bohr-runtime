from abc import ABC
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from bohrapi.core import Artifact, ArtifactType
from bohrlabels.core import Label
from jsonlines import jsonlines

from bohrruntime.fs import find_project_root
from bohrruntime.util.paths import AbsolutePath


@dataclass(frozen=True)
class Dataset(ABC):
    id: str
    top_artifact: ArtifactType
    query: Optional[Dict] = field(compare=False, default=None)
    projection: Optional[Dict] = field(compare=False, default=None)
    n_datapoints: Optional[int] = None

    def __lt__(self, other):
        if not isinstance(other, Dataset):
            raise ValueError(
                f"Cannot compare {Dataset.__name__} with {type(other).__name__}"
            )

        return self.id < other.id

    def get_path_to_file(self) -> AbsolutePath:
        return find_project_root() / "cached-datasets" / f"{self.id}.jsonl"

    def get_n_datapoints(self) -> int:
        return len(self.load_artifacts(self.n_datapoints))

    def load_artifacts(
        self, n_datapoints: Optional[int] = None
    ) -> Union[List[Artifact], List[Tuple[Artifact, Artifact]]]:
        path = self.get_path_to_file()
        if not path.exists():
            raise RuntimeError(
                f"Dataset {self.id} should have been loaded by a dvc stage first!"
            )
        artifact_list = []
        with jsonlines.open(path, "r") as reader:
            for artifact in reader:
                artifact_list.append(self.top_artifact(artifact))
                if len(artifact_list) == n_datapoints:
                    break
        return artifact_list

    def load_ground_truth_labels(
        self, label_from_datapoint_function: Callable
    ) -> Optional[Sequence[Label]]:
        artifacts = self.load_artifacts()
        label_series = [
            label_from_datapoint_function(artifact) for artifact in artifacts
        ]
        return label_series
