"""
>>> class TestArtifact(Artifact): pass
>>> test_train_dataset = Dataset(id='test_train_dataset', top_artifact=TestArtifact)
>>> test_test_dataset = Dataset(id='test_test_dataset', top_artifact=TestArtifact)
>>> task = Task(name='test_task', author='test_author', description='test_description', top_artifact=TestArtifact, labels=[CommitLabel.BugFix, CommitLabel.NonBugFix], test_datasets={test_test_dataset: None})
>>> task.get_dataset_by_id('test_test_dataset')
Dataset(id='test_test_dataset', top_artifact=<class 'core.TestArtifact'>, query=None, projection=None, n_datapoints=None)
>>> task.get_dataset_by_id('unknown_dataset')
Traceback (most recent call last):
...
ValueError: Dataset unknown_dataset is not found in task test_task
>>> task_with_one_label = Task(name='test_task', author='test_author', description='test_description', top_artifact=TestArtifact, labels=[CommitLabel.NonBugFix], test_datasets={test_test_dataset: None})
Traceback (most recent call last):
...
ValueError: At least 2 labels have to be specified
>>> zero_test_datasets_task = Task(name='test_task', author='test_author', description='test_description', top_artifact=TestArtifact, labels=[CommitLabel.NonBugFix, CommitLabel.NonBugFix], test_datasets={})
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
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from bohrapi.core import Artifact, ArtifactType, DataPointToLabelFunction
from bohrlabels.core import Label, LabelSubclass, NumericLabel, to_numeric_label
from bohrlabels.labels import MatchLabel
from frozendict import frozendict

from bohrruntime.dataset import Dataset


@dataclass(frozen=True)
class Task:
    name: str
    author: str
    description: Optional[str]
    top_artifact: ArtifactType
    labels: List[Union[Label, NumericLabel]] = field()
    test_datasets: Dict[Dataset, DataPointToLabelFunction]
    hierarchy: Optional[Type[LabelSubclass]] = None

    def __hash__(self):
        return hash(self.name)

    def is_matching_task(self) -> bool:
        return set(map(lambda x: x.to_numeric_label(), self.labels)) == {
            MatchLabel.Match.to_numeric_label(),
            MatchLabel.NoMatch.to_numeric_label(),
        }

    @staticmethod
    def infer_hierarchy(
        labels: List[Union[Label, NumericLabel]],
        hierarchy: Optional[Type[LabelSubclass]],
    ) -> Type[LabelSubclass]:
        """
        >>> from bohrlabels.labels import CommitLabel, SStuB
        >>> Task.infer_hierarchy([CommitLabel.Refactoring, CommitLabel.Feature], None)
        <enum 'CommitLabel'>
        >>> Task.infer_hierarchy([CommitLabel.Refactoring, CommitLabel.Feature], CommitLabel)
        <enum 'CommitLabel'>
        >>> Task.infer_hierarchy([CommitLabel.Refactoring, CommitLabel.Feature], SStuB)
        Traceback (most recent call last):
        ...
        ValueError: Passed hierarchy is: <enum 'SStuB'>, and one of the categories is <enum 'CommitLabel'>
        >>> Task.infer_hierarchy([SStuB.WrongFunction, CommitLabel.Feature], None)
        Traceback (most recent call last):
        ...
        ValueError: Cannot specify categories from different hierarchies: <enum 'CommitLabel'> and <enum 'SStuB'>
        """
        inferred_hierarchy = hierarchy
        for label in labels:
            if isinstance(label, Label):
                label = label.to_numeric_label()

            if isinstance(label, NumericLabel):
                if inferred_hierarchy is None:
                    inferred_hierarchy = label.hierarchy
                elif label.hierarchy != inferred_hierarchy:
                    if hierarchy is None:
                        raise ValueError(
                            f"Cannot specify categories from different hierarchies: {label.hierarchy} and {inferred_hierarchy}"
                        )
                    else:
                        raise ValueError(
                            f"Passed hierarchy is: {inferred_hierarchy}, and one of the categories is {label.hierarchy}"
                        )
            elif isinstance(label, int):
                pass
            else:
                raise AssertionError()
        if inferred_hierarchy is None:
            raise ValueError(
                "Cannot infer which hierarchy to use. Please pass `hierarchy` argument"
            )

        return inferred_hierarchy

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError(f"At least 2 labels have to be specified")
        hierarchy = self.infer_hierarchy(self.labels, self.hierarchy)
        numeric_labels = [to_numeric_label(label, hierarchy) for label in self.labels]
        object.__setattr__(self, "labels", numeric_labels)
        object.__setattr__(self, "hierarchy", hierarchy)
        if len(self.test_datasets) == 0:
            raise ValueError(f"At least 1 test dataset has to be specified")

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.get_test_datasets():
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f"Dataset {dataset_id} is not found in task {self.name}")

    def get_test_datasets(self) -> List[Dataset]:
        return list(self.test_datasets.keys())


@dataclass(frozen=True)
class Experiment:
    name: str
    task: Task
    train_dataset: Dataset
    class_balance: Optional[Tuple[float, ...]] = None
    heuristics_classifier: Optional[str] = None
    extra_test_datasets: Dict[Dataset, Callable] = field(default_factory=frozendict)

    def __post_init__(self):
        dataset_name_set = set()
        for dataset in self.datasets:
            count_before = len(dataset_name_set)
            dataset_name_set.add(dataset.id)
            count_after = len(dataset_name_set)
            if count_after == count_before:
                raise ValueError(f"Dataset {dataset.id} is present more than once.")
        if (n_labels := len(self.task.labels)) != (
            n_classes := len(self.class_balance)
        ):
            raise ValueError(
                f"Invalid class imbalance, there are {n_labels} target labels,\n"
                f"there should be the same number of class balance values {n_classes} ({self.class_balance})"
            )

    @property
    def heuristic_groups(self) -> Optional[List[str]]:
        return self._parse_heuristic_classifier()[0]

    @property
    def revision(self) -> Optional[str]:
        return self._parse_heuristic_classifier()[1]

    def _parse_heuristic_classifier(self) -> Tuple[Optional[List[str]], Optional[str]]:
        if self.heuristics_classifier is None:
            return None, None
        CLASSIFIER_REGEX = re.compile(
            "(?P<groups>(([^:@]+:)*[^:@]+)?)(@(?P<revision>[0-9a-fA-F]{7}|[0-9a-fA-F]{40}))?"
        )
        m = CLASSIFIER_REGEX.fullmatch(self.heuristics_classifier)
        if not m:
            raise ValueError(
                f"Invalid heuristic classifier syntax: {self.heuristics_classifier}. Correct syntax is: [[path1]:path2][@revision_sha]"
            )
        lst = m.group("groups")
        return None if lst == "" else lst.split(":"), m.group("revision")

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f"Unknown dataset: {dataset_id}")

    @property
    def datasets(self) -> List[Dataset]:
        return (
            self.task.get_test_datasets()
            + [self.train_dataset]
            + list(self.extra_test_datasets.keys())
        )
