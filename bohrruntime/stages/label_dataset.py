import json
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from snorkel.labeling.model import LabelModel

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.dataset import Dataset
from bohrruntime.debugging import _normalize_weights
from bohrruntime.matching.dataset import (
    MatchingDataset,
    distances_into_clusters,
    into_clusters,
    into_clusters2,
)
from bohrruntime.task import Experiment


def create_df_from_dataset(
    artifacts: List, important_fields_map, label_from_datapoint_func: Callable
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


def prob_list_to_matrix(param: List[float]):
    """
    >>> prob_list_to_matrix([1] * 9)
    Traceback (most recent call last):
    ...
    ValueError: Invalid lengths of list: 9
    >>> prob_list_to_matrix([0.12, 0.13, 0.14, 0.21, 0.23, 0.24, 0.31, 0.32, 0.34, 0.41, 0.42, 0.43])
    [[0.12, 0.13, 0.14], [0.21, 0.23, 0.24], [0.31, 0.32, 0.34], [0.41, 0.42, 0.43]]
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


def label_matching(matching_dataset: MatchingDataset, labels, probs):
    artifacts_expended_view = matching_dataset.load_expanded_view()
    distance_matrix = prob_list_to_matrix(
        [1.0 if l == -1 else p for l, p in zip(labels, probs[:, 0])]
    )
    similarity_matrix = prob_list_to_matrix(
        [0.0 if l == -1 else p for l, p in zip(labels, probs[:, 1])]
    )
    similarity_df = to_df(
        [a.single_identity for a in artifacts_expended_view], similarity_matrix
    )
    cluster_numbers = distances_into_clusters(distance_matrix)
    labeled_dataset = into_clusters2(artifacts_expended_view, cluster_numbers)
    return labeled_dataset, similarity_df


def label_dataset(
    exp: Experiment,
    dataset: Dataset,
    fs: BohrFileSystem,
    debug: bool = False,
):
    task_dir = fs.exp_dir(exp)
    dataset_dir = fs.exp_dataset_dir(exp, dataset).to_absolute_path()
    path_to_weights = task_dir.to_absolute_path() / f"label_model_weights.csv"
    label_matrix = pd.read_pickle(dataset_dir / f"heuristic_matrix.pkl")
    target_file = dataset_dir / "labeled.csv"

    label_model = LabelModel()
    label_model.load(str(task_dir.to_absolute_path() / "label_model.pkl"))
    # label_model = MajorityLabelVoter()
    artifact_list = dataset.load_artifacts()
    labels, probs = label_model.predict(L=label_matrix.to_numpy(), return_probs=True)
    probs = np.around(probs, decimals=2)
    if exp.task.is_matching_task():
        labeled_dataset, similarity_df = label_matching(dataset, labels, probs)
        with open(target_file, "w") as f:
            json.dump(labeled_dataset, f)
        similarity_df.to_csv(dataset_dir / "weighted_similarities.csv")
        return
    else:
        label_from_datapoint_func = None
        if (
            dataset in exp.task.test_datasets
            and exp.task.test_datasets[dataset] is not None
        ):
            label_from_datapoint_func = exp.task.test_datasets[dataset]
        df = create_df_from_dataset(
            artifact_list,
            dataset.top_artifact.important_fields_map(),
            label_from_datapoint_func,
        )

        df_labeled = df.assign(predicted=Series(labels))
        df_labeled[
            f"prob_{'|'.join(exp.task.labels[1].to_commit_labels_set())}"
        ] = Series(probs[:, 1])

        if debug:
            debug_(path_to_weights, label_matrix, df_labeled)

        df_labeled.to_csv(target_file, index=False)
    print(f"Labeled dataset has been written to {target_file}.")


def debug_(path_to_weights: Path, label_matrix, df_labeled):
    weights = pd.read_csv(path_to_weights, index_col="heuristic_name")

    for (
        heuristic_name,
        applied_heuristic_series,
    ) in label_matrix.iteritems():
        weights_for_heuristic = np.around(weights.loc[heuristic_name, :], decimals=3)
        formatted_weights = f'({weights_for_heuristic["00"]}/{weights_for_heuristic["01"]})__({weights_for_heuristic["10"]}/{weights_for_heuristic["11"]})'
        column_name = f"{heuristic_name}__{formatted_weights}"
        df_labeled[column_name] = applied_heuristic_series

        cond_weights = weights.apply(
            lambda row: row["01"]
            if row[column_name] == 0
            else (row["11"] if row[column_name] == 0 else 1.0),
            axis=1,
        )
        weights2[[out]] = weights[[out]]
        weights2[[zero, one]] = pd.DataFrame(
            _normalize_weights(weights2[[zero, one]].to_numpy()),
            index=weights.index,
        )
