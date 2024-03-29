from typing import List, Optional, Tuple, Union

from bohrapi.artifacts import Commit
from bohrapi.core import Artifact, ArtifactType, HeuristicObj
from bohrlabels.labels import CommitLabel
from fs.memoryfs import MemoryFS

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment
from bohrruntime.datamodel.task import Task
from bohrruntime.heuristics import HeuristicLoader, HeuristicURI
from bohrruntime.storageengine import BohrPathStructure, StorageEngine
from bohrruntime.tasktypes.labeling.core import LabelingTask

"""
Stub datamodel objects for tetsing
"""


class StubArtifact(Artifact):
    def __init__(self):
        super(StubArtifact, self).__init__({"field": "stub-data"})


def get_stub_artifact() -> StubArtifact:
    return StubArtifact()


def get_stub_dataset(name: str = "stub-dataset") -> Dataset:
    return Dataset(name, StubArtifact)


def get_stub_task(test_dataset: Optional[Dataset] = None) -> Task:
    test_datasets = {test_dataset or get_stub_dataset(): None}
    return LabelingTask(
        "stub-task",
        "stub-author",
        "stub-description",
        type(get_stub_artifact()),
        test_datasets,
        (CommitLabel.NonBugFix, CommitLabel.BugFix),
    )


def get_stub_experiment(
    task: Optional[Task] = None, no_training_dataset: bool = False
) -> Experiment:
    training_dataset = (
        None if no_training_dataset else get_stub_dataset("stub-test-dataset")
    )
    return Experiment("stub-exp", task or get_stub_task(), training_dataset)


class StubHeuristicLoader(HeuristicLoader):
    def load_heuristics_by_uri(
        self, heuristic_uri: HeuristicURI
    ) -> List[Tuple[str, Union[HeuristicObj, List[HeuristicObj]]]]:
        return [
            (
                "heuristic1",
                HeuristicObj(lambda x: x, lambda x: x, artifact_type_applied_to=Commit),
            )
        ]

    def get_heuristic_uris(
        self,
        heuristic_uri: HeuristicURI = None,
        input_artifact_type: Optional[ArtifactType] = None,
        error_if_none_found: bool = True,
    ) -> List[HeuristicURI]:
        return [
            HeuristicURI.from_path_and_fs("/heuristic1", self.heuristic_fs),
            HeuristicURI.from_path_and_fs("/heuristic2", self.heuristic_fs),
        ]


def get_stub_storage_engine() -> StorageEngine:
    fs = MemoryFS()
    return StorageEngine(fs, BohrPathStructure(), StubHeuristicLoader(fs))
