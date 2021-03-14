from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from snorkel.labeling import LabelingFunction, PandasLFApplier

from bohr.config import Config, load_config, load_heuristics_from_module
from bohr.core import to_labeling_functions
from bohr.datamodel import Task
from bohr.pipeline.data_analysis import calculate_metrics


def apply_lfs_to_dataset(
    lfs: List[LabelingFunction],
    artifact_df: pd.DataFrame,
    save_to: Path,
) -> np.ndarray:
    applier = PandasLFApplier(lfs=lfs)
    applied_lf_matrix = applier.apply(df=artifact_df)
    df = pd.DataFrame(applied_lf_matrix, columns=[lf.name for lf in lfs])
    df.to_pickle(str(save_to))
    return applied_lf_matrix


def create_dirs_if_necessary(
    task: Task, config: Config, heuristic_group: str
) -> Tuple[Path, Path]:
    task_dir_generated = config.paths.generated / task.name / heuristic_group
    task_dir_metrics = config.paths.metrics / task.name / heuristic_group
    for dir in [task_dir_generated, task_dir_metrics]:
        if not dir.exists():
            dir.mkdir(parents=True)
    return task_dir_generated, task_dir_metrics


def apply_heuristics(
    task_name: str, config: Config, heuristic_group: str, dataset: str
) -> None:
    task = config.tasks[task_name]

    task_dir_generated, task_dir_metrics = create_dirs_if_necessary(
        task, config, heuristic_group=heuristic_group
    )
    heuristics = load_heuristics_from_module(task.top_artifact, heuristic_group)
    if not heuristics:
        raise ValueError(f"Heuristics not found for artifact: {task.top_artifact}")

    dataset_loader = task.datasets[dataset]
    save_to_matrix = task_dir_generated / f"heuristic_matrix_{dataset}.pkl"
    save_to_metrics = task_dir_metrics / f"heuristic_metrics_{dataset}.json"
    labeling_functions = to_labeling_functions(
        heuristics, dataset_loader.get_mapper(), task.labels
    )
    artifact_df = dataset_loader.load(config.project_root)
    apply_lf_matrix = apply_lfs_to_dataset(
        labeling_functions, artifact_df=artifact_df, save_to=save_to_matrix
    )
    label_series = (
        artifact_df[task.label_column_name]
        if task.label_column_name in artifact_df.columns
        else None
    )
    calculate_metrics(
        apply_lf_matrix, labeling_functions, label_series, save_to=save_to_metrics
    )


if __name__ == "__main__":
    config = load_config()

    apply_heuristics(
        "bugginess",
        config,
        "heuristics.bugginess.main_heurstics",
        "dataloaders.1151-commits",
    )
