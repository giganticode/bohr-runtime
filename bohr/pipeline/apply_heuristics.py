from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from snorkel.labeling import LabelingFunction, PandasLFApplier

from bohr.config.pathconfig import PathConfig
from bohr.core import to_labeling_function, to_labeling_functions
from bohr.datamodel.bohrrepo import load_bohr_repo
from bohr.datamodel.dataset import Dataset
from bohr.datamodel.heuristic import load_heuristic_by_name, load_heuristics_from_module
from bohr.datamodel.task import Task
from bohr.pipeline.data_analysis import calculate_metrics


def apply_lfs_to_dataset(
    lfs: List[LabelingFunction], artifact_df: pd.DataFrame
) -> np.ndarray:
    applier = PandasLFApplier(lfs=lfs)
    applied_lf_matrix = applier.apply(df=artifact_df)
    return applied_lf_matrix


def create_dirs_if_necessary(
    task: Task, path_config: PathConfig, heuristic_group: str
) -> Tuple[Path, Path]:
    task_dir_generated = path_config.generated / task.name / heuristic_group
    task_dir_metrics = path_config.metrics / task.name / heuristic_group
    for dir in [task_dir_generated, task_dir_metrics]:
        if not dir.exists():
            dir.mkdir(parents=True)
    return task_dir_generated, task_dir_metrics


def get_labeling_functions_from_heuristic_group(
    task: Task, heuristic_group: str, dataset: Dataset
) -> List[LabelingFunction]:
    heuristics = load_heuristics_from_module(task.top_artifact, heuristic_group)
    if not heuristics:
        raise ValueError(f"Heuristics not found for artifact: {task.top_artifact}")
    labeling_functions = to_labeling_functions(heuristics, dataset.mapper, task.labels)
    return labeling_functions


def apply_heuristic_group_to_dataset(
    task: Task, heuristic_group: str, dataset: Dataset
) -> np.ndarray:
    labeling_functions = get_labeling_functions_from_heuristic_group(
        task, heuristic_group, dataset
    )
    artifact_df = dataset.load()
    apply_lf_matrix = apply_lfs_to_dataset(labeling_functions, artifact_df=artifact_df)
    return apply_lf_matrix


def apply_heuristic_to_dataset(
    task: Task,
    heuristic_name: str,
    dataset: Dataset,
    n_datapoints: Optional[int] = None,
    path_config: Optional[PathConfig] = None,
) -> np.ndarray:
    path_config = path_config or PathConfig.load()
    heuristic = load_heuristic_by_name(
        heuristic_name, task.top_artifact, path_config.heuristics
    )
    labeling_function = to_labeling_function(heuristic, dataset.mapper, task.labels)
    apply_lf_matrix = apply_lfs_to_dataset(
        [labeling_function], artifact_df=dataset.load(n_datapoints)
    )
    return apply_lf_matrix


def apply_heuristics_and_save_metrics(
    task: Task,
    heuristic_group: str,
    dataset: Dataset,
    path_config: Optional[PathConfig] = None,
) -> None:

    path_config = path_config or PathConfig.load()
    task_dir_generated, task_dir_metrics = create_dirs_if_necessary(
        task, path_config, heuristic_group=heuristic_group
    )

    save_to_matrix = task_dir_generated / f"heuristic_matrix_{dataset.name}.pkl"
    save_to_metrics = task_dir_metrics / f"heuristic_metrics_{dataset.name}.json"

    labeling_functions = get_labeling_functions_from_heuristic_group(
        task, heuristic_group, dataset
    )
    artifact_df = dataset.load()
    applied_lf_matrix = apply_lfs_to_dataset(
        labeling_functions, artifact_df=artifact_df
    )

    df = pd.DataFrame(applied_lf_matrix, columns=[lf.name for lf in labeling_functions])
    df.to_pickle(str(save_to_matrix))
    label_series = (
        artifact_df[task.label_column_name]
        if task.label_column_name in artifact_df.columns
        else None
    )
    calculate_metrics(
        applied_lf_matrix, labeling_functions, label_series, save_to=save_to_metrics
    )
