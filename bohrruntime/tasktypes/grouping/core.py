import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bohrlabels.core import LabelSet
from bohrlabels.labels import MatchLabel
from fs.base import FS
from pandas import DataFrame

from bohrruntime.datamodel.model import (
    CombinedHeuristicOutputs,
    GroundTruthLabels,
    HeuristicOutputs,
    Model,
    ModelTrainer,
)
from bohrruntime.datamodel.task import DatasetPreparator, PreparedDataset, Task
from bohrruntime.tasktypes.grouping.dataset import (
    GroupingDataset,
    distances_into_clusters,
    into_clusters2,
)
from bohrruntime.tasktypes.labeling.core import (
    CategoryMappingCache,
    DatasetLabeler,
    check_duplicate_heuristics,
    majority_acc,
)
from bohrruntime.tasktypes.labeling.lfs import HeuristicApplier, SnorkelHeuristicApplier
from bohrruntime.tasktypes.labeling.model import LabelModelTrainer


class MatchingModel(Model):
    def predict(self, heuristic_outputs: HeuristicOutputs) -> CombinedHeuristicOutputs:
        pass

    def save(self) -> None:
        pass

    def load(self):
        pass


def label_matching(
    matching_dataset: GroupingDataset,
    combined_heuristic_outputs: CombinedHeuristicOutputs,
    subfs: FS,
):
    artifacts_expended_view = matching_dataset.load_expanded_view(subfs)
    distance_matrix = prob_list_to_matrix(
        [
            1.0 if l == -1 else p
            for l, p in zip(
                combined_heuristic_outputs.labels,
                combined_heuristic_outputs.probs[:, 0],
            )
        ]
    )
    similarity_matrix = prob_list_to_matrix(
        [
            0.0 if l == -1 else p
            for l, p in zip(
                combined_heuristic_outputs.labels,
                combined_heuristic_outputs.probs[:, 1],
            )
        ]
    )
    similarity_df = to_df(
        [a.single_identity for a in artifacts_expended_view], similarity_matrix
    )
    cluster_numbers = distances_into_clusters(distance_matrix)
    labeled_dataset = into_clusters2(artifacts_expended_view, cluster_numbers)
    return labeled_dataset, similarity_df


def prob_list_to_matrix(param: List[float]):
    """
    >>> prob_list_to_matrix([1] * 9)
    Traceback (most recent call last):
    ...
    ValueError: Invalid lengths of list: 9
    >>> prob_list_to_matrix([0.12, 0.13, 0.14, 0.21, 0.23, 0.24, 0.31, 0.32, 0.34, 0.41, 0.42, 0.43])
    [[0.0, 0.12, 0.13, 0.14], [0.21, 0.0, 0.23, 0.24], [0.31, 0.32, 0.0, 0.34], [0.41, 0.42, 0.43, 0.0]]
    """
    matrix_len = int((1 + (1 + 4 * len(param)) ** (0.5)) // 2)
    if matrix_len * (matrix_len - 1) != len(param):
        raise ValueError(f"Invalid lengths of list: {len(param)}")
    res = []
    for i in range(matrix_len):
        r = []
        for j in range(matrix_len - 1):
            if j == i:
                r.append(0.0)
            r.append(param[(matrix_len - 1) * i + j])
        res.append(r)
    res[-1].append(0.0)
    return res


def to_df(artifacts_expended_view: List, distance_matrix: List[List]) -> pd.DataFrame:
    """
    >>> to_df(["a", "b", "c"], [[0.0, 0.6, 0.7], [0.6, 0.0, 0.8], [0.7, 0.8, 0.0]])
         a    b    c
    a  0.0  0.6  0.7
    b  0.6  0.0  0.8
    c  0.7  0.8  0.0
    """
    return DataFrame(
        distance_matrix, index=artifacts_expended_view, columns=artifacts_expended_view
    )


@dataclass(frozen=True)
class GroupingTask(Task):
    def combine_heuristics(
        self, matrix_list: List[HeuristicOutputs]
    ) -> HeuristicOutputs:
        # TODO copy-paste from labeling
        all_heuristics_matrix = pd.concat(
            map(lambda l: l.label_matrix, matrix_list), axis=1
        )
        all_heuristics_matrix = self.get_task_specific_labels(all_heuristics_matrix)
        check_duplicate_heuristics(all_heuristics_matrix)
        return HeuristicOutputs(all_heuristics_matrix)

    def get_preparator(self) -> DatasetPreparator:
        return DatasetMatcher()

    def get_model_trainer(self, fs: FS) -> ModelTrainer:
        class_balance = [matches := (3 - 1) / (18.0 - 1), 1 - matches]  # TODO
        return LabelModelTrainer(fs, class_balance)

    def calculate_heuristic_output_metrics(
        self,
        heuristic_outputs: HeuristicOutputs,
        label_series: GroundTruthLabels = None,
    ) -> Dict[str, Any]:
        # TODO this is copy-paste from labels
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

    def get_category_mapping_cache(self) -> CategoryMappingCache:
        return CategoryMappingCache(
            [LabelSet.of(MatchLabel.NoMatch), LabelSet.of(MatchLabel.Match)],
            maxsize=10000,
        )

    def get_task_specific_labels(self, all_heuristics_matrix):
        # TODO this is copy-paste from labeling
        category_mapping_cache = self.get_category_mapping_cache()
        from_snorkel_label = (
            lambda v: v
            if v == -1
            else category_mapping_cache[LabelSet.from_bitmap(v, MatchLabel)]
        )
        if isinstance(all_heuristics_matrix, np.ndarray):  # TODO make it better
            all_heuristics_matrix = pd.DataFrame(all_heuristics_matrix)
            all_heuristics_matrix = all_heuristics_matrix.applymap(from_snorkel_label)
            return all_heuristics_matrix.to_numpy()
        else:
            return all_heuristics_matrix.applymap(from_snorkel_label)

    def load_ground_truth_labels(
        self, dataset: GroupingDataset, cached_datasets_fs: FS
    ) -> Optional[GroundTruthLabels]:
        if (
            dataset not in self.test_datasets
            or (label_from_datapoint_function := self.test_datasets[dataset]) is None
        ):
            return None
        category_mapping_cache = self.get_category_mapping_cache()
        label_series = dataset.load_ground_truth_labels(
            label_from_datapoint_function, cached_datasets_fs
        )
        label_series = np.array(
            list(
                map(
                    lambda x: category_mapping_cache[
                        LabelSet.from_bitmap(x, MatchLabel)
                    ],
                    label_series.labels,
                )
            )
        )
        return GroundTruthLabels(label_series)


class MatchedDataset(PreparedDataset):
    def __init__(self, labeled_dataset, similiarity_df: pd.DataFrame):
        self.labeled_dataset = labeled_dataset
        self.similiarity_df = similiarity_df

    def save(self, subfs: FS) -> None:
        target_file = "labeled.csv"
        with subfs.open(target_file, "w") as f:
            json.dump(self.labeled_dataset, f)
        self.similiarity_df.to_csv("weighted_similarities.csv")


class DatasetMatcher(DatasetPreparator):
    def prepare(
        self,
        dataset: GroupingDataset,
        task: Task,
        combined_heuristic_outputs: CombinedHeuristicOutputs,
        subfs: FS,
    ) -> PreparedDataset:
        labeled_dataset, similarity_df = label_matching(
            dataset, combined_heuristic_outputs, subfs
        )
        return MatchedDataset(labeled_dataset, similarity_df)
