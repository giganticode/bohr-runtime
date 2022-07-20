"""
>>> from bohrlabels.core import LabelSet
>>> from bohrlabels.labels import CommitLabel
>>> from bohrapi.artifacts import Commit
>>> from bohrruntime.tasktypes.labeling.core import LabelingTask
>>> from bohrruntime.datamodel.experiment import Experiment
>>> test_train_dataset = Dataset(id='test_train_dataset', heuristic_input_artifact_type=Commit)
>>> test_test_dataset = Dataset(id='test_test_dataset', heuristic_input_artifact_type=Commit)
>>> task = LabelingTask(name='test_task', author='test_author', description='test_description', heuristic_input_artifact_type=Commit, labels=(LabelSet.of(CommitLabel.BugFix), LabelSet.of(CommitLabel.NonBugFix)), test_datasets={test_test_dataset: None})
>>> task.get_dataset_by_id('test_test_dataset')
Dataset(id='test_test_dataset', heuristic_input_artifact_type=<class 'bohrapi.artifacts.commit.Commit'>, query=None, projection=None, n_datapoints=None, path=None)
>>> task.get_dataset_by_id('unknown_dataset')
Traceback (most recent call last):
...
ValueError: Dataset unknown_dataset is not found in task test_task
>>> task_with_one_label = LabelingTask(name='test_task', author='test_author', description='test_description', heuristic_input_artifact_type=Commit, labels=(CommitLabel.NonBugFix), test_datasets={test_test_dataset: None})
Traceback (most recent call last):
...
TypeError: object of type 'CommitLabel' has no len()
>>> zero_test_datasets_task = LabelingTask(name='test_task', author='test_author', description='test_description', heuristic_input_artifact_type=Commit, labels=(LabelSet.of(CommitLabel.NonBugFix), LabelSet.of(CommitLabel.NonBugFix)), test_datasets={})
Traceback (most recent call last):
...
ValueError: At least 1 test dataset has to be specified
>>> exp0 = Experiment(name='test', task=task, train_dataset=test_train_dataset)
>>> exp0.heuristic_groups is None
True
>>> exp0.revision is None
True
>>> exp = Experiment(name='test', task=task, train_dataset=test_train_dataset, heuristics_classifier='')
>>> exp.heuristic_groups is None
True
>>> exp.revision is None
True
>>> exp2 = Experiment(name='test', task=task, train_dataset=test_train_dataset, heuristics_classifier='@afaf233')
>>> exp2.heuristic_groups is None
True
>>> exp2.revision
'afaf233'
>>> exp3 = Experiment(name='test', task=task, train_dataset=test_train_dataset, heuristics_classifier='group1:group2@afaf233')
>>> exp3.heuristic_groups
['group1', 'group2']
>>> exp3.revision
'afaf233'
>>> exp4 = Experiment(name='test', task=task, train_dataset=test_train_dataset, heuristics_classifier=':@afaf233')
>>> exp4.heuristic_groups
Traceback (most recent call last):
...
ValueError: Invalid heuristic classifier syntax: :@afaf233. Correct syntax is: [[path1]:path2][@revision_sha]
>>> exp5 = Experiment(name='test', task=task, train_dataset=test_train_dataset, heuristics_classifier=':group2@afaf233')
>>> exp5.heuristic_groups
Traceback (most recent call last):
...
ValueError: Invalid heuristic classifier syntax: :group2@afaf233. Correct syntax is: [[path1]:path2][@revision_sha]
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from bohrapi.core import ArtifactType, DataPointToLabelFunction
from fs.base import FS

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.model import (
    CombinedHeuristicOutputs,
    GroundTruthLabels,
    HeuristicOutputs,
    ModelTrainer,
)

# TODO heuristicURI vs heuristicClassifier
from bohrruntime.tasktypes.labeling.lfs import HeuristicApplier, SnorkelHeuristicApplier


class PreparedDataset(ABC):
    def save(self, subfs: FS) -> None:
        pass


class DatasetPreparator(ABC):
    @abstractmethod
    def prepare(
        self,
        dataset: Dataset,
        task: "Task",
        combined_heuristic_outputs: CombinedHeuristicOutputs,
        subfs: FS,
    ) -> PreparedDataset:
        pass


@dataclass(frozen=True)
class Task(ABC):
    name: str
    author: str
    description: Optional[str]
    heuristic_input_artifact_type: ArtifactType
    test_datasets: Dict[Dataset, DataPointToLabelFunction]

    def get_heuristic_applier(self) -> HeuristicApplier:
        return SnorkelHeuristicApplier()  # TODO to be replaced by in-house applier

    @abstractmethod
    def get_preparator(self) -> DatasetPreparator:
        pass

    @abstractmethod
    def get_model_trainer(self, fs: FS) -> ModelTrainer:
        pass

    @abstractmethod
    def calculate_heuristic_output_metrics(
        self,
        heuristic_outputs: HeuristicOutputs,
        label_series: GroundTruthLabels = None,
    ) -> Dict[str, Any]:
        pass

    def __hash__(self):
        return hash(self.name)

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.get_test_datasets():
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f"Dataset {dataset_id} is not found in task {self.name}")

    def get_test_datasets(self) -> List[Dataset]:
        return list(self.test_datasets.keys())

    @abstractmethod
    def load_ground_truth_labels(
        self, dataset: Dataset, cached_datasets_fs: FS
    ) -> Optional[GroundTruthLabels]:
        pass

    @abstractmethod
    def combine_heuristics(
        self, matrix_list: List[HeuristicOutputs]
    ) -> HeuristicOutputs:
        pass
