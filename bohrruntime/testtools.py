from typing import List, Optional

from bohrapi.core import Artifact, ArtifactType
from bohrlabels.labels import CommitLabel
from fs.memoryfs import MemoryFS

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment
from bohrruntime.datamodel.task import Task
from bohrruntime.heuristicuri import HeuristicURI
from bohrruntime.storageengine import BohrPathStructure, HeuristicLoader, StorageEngine

# class StubTask(Task):
#     def load_ground_truth_labels(self, func):
#         return None
#
#     def get_preparator(self) -> DatasetPreparator:
#         return None
#
#     def get_model_trainer(self) -> ModelTrainer:
#         return None
#
#     def calculate_heuristic_output_metrics(self, heuristic_outputs: HeuristicOutputs, label_series: np.ndarray = None) -> Dict[str, Any]:
#         return {}
#
#     def is_grouping_task(self) -> bool:
#         return False
from bohrruntime.tasktypes.labeling.core import LabelingTask


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


class VirtualHeuristicLoader(HeuristicLoader):
    def get_heuristic_uris(
        self,
        heuristic_uri: HeuristicURI = None,
        input_artifact_type: Optional[ArtifactType] = None,
        error_if_none_found: bool = True,
    ) -> List[HeuristicURI]:
        fs = MemoryFS()
        return [
            HeuristicURI.from_path_and_fs("/heuristic1", fs),
            HeuristicURI.from_path_and_fs("/heuristic2", fs),
        ]


class VirtualStorageEngine(StorageEngine):
    def __init__(self):
        super(VirtualStorageEngine, self).__init__(MemoryFS(), BohrPathStructure())

    def get_heuristic_loader(self) -> HeuristicLoader:
        return VirtualHeuristicLoader(MemoryFS())
