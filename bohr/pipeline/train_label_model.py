import math
from math import log
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from dask.dataframe import DataFrame
from snorkel.labeling.model import LabelModel

from bohr.config import Config, load_config
from bohr.datamodel import DatasetLoader
from bohr.pipeline.core import label, train_lmodel


def get_test_set_accuracy(
    label_model: LabelModel,
    test_set_name: str,
    df: DataFrame,
    save_to: Path,
    label_column_name: str,
) -> float:
    lines = np.load(
        str(save_to / f"heuristic_matrix_{test_set_name}.pkl"), allow_pickle=True
    )
    return label_model.score(
        L=lines, Y=df[label_column_name], tie_break_policy="random"
    )["accuracy"]


def extract_subset(matrix: np.ndarray, df: DataFrame) -> Tuple[np.ndarray, DataFrame]:
    if len(matrix) != len(df):
        raise AssertionError
    n_datapoints_to_extract = int(log(len(matrix)) * 3)
    indices_to_extract = [
        i for i in range(len(matrix)) if i % n_datapoints_to_extract == 0
    ]
    new_df = df.iloc[indices_to_extract].copy().reset_index(drop=True)
    # TODO for bugginess task only select columns: [["owner", "repository", "sha", "message", "bug"]]
    # TODO in the fututre the list of columns shoudl depend on the task
    return matrix[indices_to_extract], new_df


def label_subset(
    label_model: LabelModel,
    test_set_name: str,
    df: DataFrame,
    task_generated_path: Path,
):
    applied_heuristics_matrix = np.load(
        str(task_generated_path / f"heuristic_matrix_{test_set_name}.pkl"),
        allow_pickle=True,
    )

    applied_heuristics_matrix_part, df_part = extract_subset(
        applied_heuristics_matrix, df
    )
    df_labeled = label(label_model, applied_heuristics_matrix_part, df_part)
    target_file = task_generated_path / f"labeled_test_subset_{test_set_name}.csv"
    df_labeled.to_csv(target_file, index=False)
    print(f"Labeled dataset has been written to {target_file}.")


def train_label_model(task_name: str, config: Config) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    task_dir_generated = config.paths.generated / task_name
    if not task_dir_generated.exists():
        task_dir_generated.mkdir()

    lines_train = np.load(
        str(task_dir_generated / "heuristic_matrix_train.pkl"), allow_pickle=True
    )
    label_model = train_lmodel(lines_train)
    label_model.save(str(task_dir_generated / "label_model.pkl"))
    label_model.eval()

    task = config.tasks[task_name]
    for test_set_name, test_set in task.test_datasets.items():
        df = test_set.load(config.project_root)
        stats[f"label_model_acc_{test_set_name}"] = get_test_set_accuracy(
            label_model,
            test_set_name,
            df,
            save_to=task_dir_generated,
            label_column_name=task.label_column_name,
        )

        label_subset(label_model, test_set_name, df, task_dir_generated)

    return stats


if __name__ == "__main__":
    train_label_model("bugginess", load_config())
