import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bohrapi.core import Artifact
from bohrlabels.core import LabelSet
from bohrlabels.labels import CommitLabel
from cachetools import LRUCache
from fs.base import FS
from pandas import Series
from snorkel.labeling.model import MajorityLabelVoter

from bohrruntime.datamodel.dataset import DatapointList, Dataset
from bohrruntime.datamodel.model import (
    CombinedHeuristicOutputs,
    GroundTruthLabels,
    HeuristicOutputs,
    ModelTrainer,
)
from bohrruntime.datamodel.task import DatasetPreparator, PreparedDataset, Task
from bohrruntime.tasktypes.labeling.lfs import HeuristicApplier, SnorkelHeuristicApplier
from bohrruntime.tasktypes.labeling.model import LabelModelTrainer

logger = logging.getLogger(__name__)


class CategoryMappingCache(LRUCache):
    """
    >>> from bohrlabels.labels import CommitLabel
    >>> logger.setLevel("CRITICAL")

    >>> cache = CategoryMappingCache([LabelSet.of(CommitLabel.NonBugFix), LabelSet.of(CommitLabel.BugFix)], 10)
    >>> cache[LabelSet.of(CommitLabel.NonBugFix)]
    0
    >>> cache[LabelSet.of(CommitLabel.BugFix)]
    1
    >>> cache[LabelSet.of(CommitLabel.MinorBugFix)]
    1
    >>> cache[LabelSet.of(CommitLabel.CommitLabel)]
    -1
    """

    def __init__(self, label_categories: List[LabelSet], maxsize: int):
        super().__init__(maxsize)
        self.label_categories = label_categories
        self.map = {label_set: i for i, label_set in enumerate(self.label_categories)}

    def __missing__(self, labels: LabelSet) -> int:
        selected_label = labels.belongs_to(self.label_categories)
        if selected_label in self.map:
            snorkel_label = self.map[selected_label]
            self[labels] = snorkel_label
            logger.info(
                f"Converted {'|'.join(labels.to_set_of_labels())} label into {snorkel_label}"
            )
            return snorkel_label
        else:
            logger.info(
                f"Label {'|'.join(labels.to_set_of_labels())} cannot be unambiguously converted to any label, abstaining.."
            )
            self[labels] = -1
            return -1


def check_duplicate_heuristics(all_heuristics_matrix: pd.DataFrame) -> None:
    if sum(all_heuristics_matrix.columns.duplicated()) != 0:
        s = set()
        for c in all_heuristics_matrix.columns:
            if c in s:
                raise ValueError(f"Duplicate heuristics are present: {c}")
            s.add(c)
        raise AssertionError()


@dataclass(frozen=True)
class LabelingTask(Task):
    labels: Tuple[LabelSet]
    class_balance: Tuple[float, ...] = None

    def get_category_mapping_cache(self) -> CategoryMappingCache:
        return CategoryMappingCache(list(self.labels), maxsize=10000)

    def get_model_trainer(self, fs: FS) -> ModelTrainer:
        class_balance = self.class_balance or (
            [1.0 / len(self.labels) for _ in range(len(self.labels))]
        )
        return LabelModelTrainer(fs, class_balance)

    def get_preparator(self) -> DatasetPreparator:
        return DatasetLabeler()

    def calculate_heuristic_output_metrics(
        self,
        heuristic_outputs: HeuristicOutputs,
        label_series: GroundTruthLabels = None,
    ) -> Dict[str, Any]:
        if heuristic_outputs is None:
            return {}
        label_matrix = heuristic_outputs.label_matrix.to_numpy()
        metrics = {}
        coverage = sum((label_matrix != -1).any(axis=1)) / len(label_matrix)
        if label_series is not None:
            label_matrix = self.get_task_specific_labels(label_matrix)
            majority_accuracy = majority_acc(label_matrix, label_series.labels)
            metrics["majority_accuracy_not_abstained"] = majority_accuracy
        metrics["n_labeling_functions"] = label_matrix.shape[1]
        metrics[f"coverage"] = coverage
        return metrics

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError(f"At least 2 labels have to be specified")

        if len(self.test_datasets) == 0:
            raise ValueError(f"At least 1 test dataset has to be specified")

        if self.class_balance:
            if (n_labels := len(self.labels)) != (n_classes := len(self.class_balance)):
                raise ValueError(
                    f"Invalid class imbalance, there are {n_labels} target labels,\n"
                    f"there should be the same number of class balance values {n_classes} ({self.class_balance})"
                )

    def load_ground_truth_labels(
        self, dataset: Dataset, cached_datasets_fs: FS
    ) -> Optional[GroundTruthLabels]:
        if (
            dataset not in self.test_datasets
            or (label_from_datapoint_function := self.test_datasets[dataset]) is None
        ):
            return None
        category_mapping_cache = self.get_category_mapping_cache()
        artifacts = dataset.load_artifacts(cached_datasets_fs)
        label_series = [
            label_from_datapoint_function(artifact) for artifact in artifacts
        ]
        label_series = np.array(
            list(
                map(
                    lambda x: category_mapping_cache[
                        LabelSet.from_bitmap(x, CommitLabel)
                    ],
                    label_series,
                )
            )
        )
        return GroundTruthLabels(label_series)

    def combine_heuristics(
        self, matrix_list: List[HeuristicOutputs]
    ) -> HeuristicOutputs:
        all_heuristics_matrix = pd.concat(
            map(lambda l: l.label_matrix, matrix_list), axis=1
        )
        all_heuristics_matrix = self.get_task_specific_labels(all_heuristics_matrix)
        check_duplicate_heuristics(all_heuristics_matrix)
        return HeuristicOutputs(all_heuristics_matrix)

    def get_task_specific_labels(self, all_heuristics_matrix):
        category_mapping_cache = self.get_category_mapping_cache()
        from_snorkel_label = (
            lambda v: v
            if v == -1
            else category_mapping_cache[LabelSet.from_bitmap(v, CommitLabel)]
        )
        if isinstance(all_heuristics_matrix, np.ndarray):  # TODO make it better
            all_heuristics_matrix = pd.DataFrame(all_heuristics_matrix)
            all_heuristics_matrix = all_heuristics_matrix.applymap(from_snorkel_label)
            return all_heuristics_matrix.to_numpy()
        else:
            return all_heuristics_matrix.applymap(from_snorkel_label)


class LabeledDataset(PreparedDataset):
    def __init__(self, df_labeled):
        self.df_labeled = df_labeled

    def save(self, subfs: FS):
        """
        >>> from fs.memoryfs import MemoryFS
        >>> from fs.errors import NoURL
        >>> fs = MemoryFS()
        >>> try:
        ...     LabeledDataset(pd.DataFrame()).save(fs)
        ... except NoURL: pass
        >>> fs.exists('labeled.csv')
        True
        """
        target_file = "labeled.csv"
        with subfs.open(target_file, "w") as f:
            self.df_labeled.to_csv(f, index=False)
        print(f"Labeled dataset has been written to {subfs.geturl(target_file)}.")


class DatasetLabeler(DatasetPreparator):
    def prepare(
        self,
        dataset: Dataset,
        task: Task,
        combined_heuristic_outputs: CombinedHeuristicOutputs,
        subfs: FS,
    ) -> PreparedDataset:
        label_from_datapoint_func = None
        if dataset in task.test_datasets and task.test_datasets[dataset] is not None:
            label_from_datapoint_func = task.test_datasets[dataset]
        artifact_list: DatapointList = dataset.load_artifacts(subfs)
        df = create_df_from_dataset(
            artifact_list,
            dataset.heuristic_input_artifact_type.important_fields_map(),
            label_from_datapoint_func,
        )

        df_labeled = df.assign(predicted=Series(combined_heuristic_outputs.labels))
        df_labeled[f"prob_{'|'.join(task.labels[1].to_set_of_labels())}"] = Series(
            combined_heuristic_outputs.probs[:, 1]
        )

        # debug_labeling_functions(path_to_weights, label_matrix, df_labeled)
        return LabeledDataset(df_labeled)


def create_df_from_dataset(
    artifacts: List[Artifact], important_fields_map, label_from_datapoint_func: Callable
) -> pd.DataFrame:
    keys = []
    columns = []
    for k, v in important_fields_map.items():
        keys.append(k)
        columns.append(v)
    if label_from_datapoint_func is not None:
        df = pd.DataFrame(
            [
                [c.raw_data[k] for k in keys] + [label_from_datapoint_func(c)]
                for c in artifacts
            ],
            columns=columns + ["label"],
        )
    else:
        df = pd.DataFrame(
            [[c.raw_data[k] for k in keys] for c in artifacts],
            columns=columns,
        )
    return df


def majority_acc(line: np.ndarray, label_series: np.ndarray) -> float:
    majority_model = MajorityLabelVoter()
    maj_model_train_acc = majority_model.score(
        L=line, Y=label_series, tie_break_policy="abstain"
    )["accuracy"]
    return maj_model_train_acc
