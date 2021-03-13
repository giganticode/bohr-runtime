import json
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpyencoder import NumpyEncoder
from pandas import Series
from snorkel.labeling import LabelingFunction, LFAnalysis, PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter

from bohr.config import Config, load_config, load_heuristics_from_module
from bohr.core import to_labeling_functions
from bohr.datamodel import DatasetLoader, Heuristic, Task


def majority_acc(line: np.ndarray, label_series: Series) -> float:
    majority_model = MajorityLabelVoter()
    maj_model_train_acc = majority_model.score(
        L=line, Y=label_series.values, tie_break_policy="random"
    )["accuracy"]
    return maj_model_train_acc


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


def save_analysis(
    applied_lf_matrix: np.ndarray,
    lfs: List[LabelingFunction],
    save_csv_to: Path,
    save_json_to: Path,
    label_series: Optional[Series] = None,
) -> None:
    lf_analysis_summary = LFAnalysis(applied_lf_matrix, lfs).lf_summary(
        Y=label_series.values if label_series is not None else None
    )
    lf_analysis_summary.to_csv(save_csv_to)
    analysis_dict = lf_analysis_summary.to_dict()
    del analysis_dict["j"]
    with open(save_json_to, "w") as f:
        json.dump(analysis_dict, f, indent=4, sort_keys=True, cls=NumpyEncoder)


def calculate_metrics(
    applied_lf_matrix: np.ndarray,
    lfs: List[LabelingFunction],
    dataset_name: str,
    label_series: Optional[Series] = None,
) -> Dict[str, Any]:
    metrics = {}
    coverage = sum((applied_lf_matrix != -1).any(axis=1)) / len(applied_lf_matrix)
    if label_series is not None:
        majority_accuracy = majority_acc(applied_lf_matrix, label_series)
        metrics[f"majority_accuracy_{dataset_name}"] = majority_accuracy
    else:
        metrics["n_labeling_functions"]: len(lfs)
    metrics[f"coverage_{dataset_name}"] = coverage
    return metrics


def create_dirs_if_necessary(
    task: Task, config: Config, heuristic_module_path: str
) -> Tuple[Path, Path]:
    task_dir_generated = config.paths.generated / task.name / heuristic_module_path
    task_dir_metrics = config.paths.metrics / task.name / heuristic_module_path
    for dir in [task_dir_generated, task_dir_metrics]:
        if not dir.exists():
            dir.mkdir(parents=True)
    return task_dir_generated, task_dir_metrics


def apply_heuristics_to_dataset(
    heuristics: List[Heuristic],
    dataset_loader: DatasetLoader,
    labels: List[str],
    save_to: Path,
    config: Config,
) -> Tuple[np.ndarray, List[LabelingFunction]]:
    labeling_functions = to_labeling_functions(
        heuristics, dataset_loader.get_mapper(), labels
    )
    artifact_df = dataset_loader.load(config.project_root)

    applied_lf_matrix = apply_lfs_to_dataset(
        labeling_functions, artifact_df=artifact_df, save_to=save_to
    )
    return applied_lf_matrix, labeling_functions


def analysis_and_metrics(
    applied_lf_matrix: np.ndarray,
    labeling_functions: List[LabelingFunction],
    dataset_loader,
    dataset_loader_name: str,
    task: Task,
    config: Config,
    heuristic_module_path: Optional[str] = None,
) -> Dict[str, Any]:
    heuristic_module_path = heuristic_module_path or ""
    save_csv_to = (
        config.paths.generated
        / task.name
        / heuristic_module_path
        / f"analysis_{dataset_loader_name}.csv"
    )
    save_json_to = (
        config.paths.metrics
        / task.name
        / heuristic_module_path
        / f"analysis_{dataset_loader_name}.json"
    )
    artifact_df = dataset_loader.load(config.project_root)
    label_series = (
        artifact_df[task.label_column_name]
        if task.label_column_name in artifact_df.columns
        else None
    )
    save_analysis(
        applied_lf_matrix, labeling_functions, save_csv_to, save_json_to, label_series
    )
    stats = calculate_metrics(
        applied_lf_matrix, labeling_functions, dataset_loader_name, label_series
    )
    return stats


def apply_heuristics(
    task_name: str, config: Config, heuristic_module_path: str
) -> None:
    task = config.tasks[task_name]

    task_dir_generated, task_dir_metrics = create_dirs_if_necessary(
        task, config, heuristic_module_path=heuristic_module_path
    )
    heuristics = load_heuristics_from_module(task.top_artifact, heuristic_module_path)
    if not heuristics:
        raise ValueError(f"Heuristics not found for artifact: {task.top_artifact}")

    all_stats: Dict[str, Any] = {}
    for dataset_loader_name, dataset_loader in {
        **task.test_datasets,
        **task.train_datasets,
    }.items():
        save_to = task_dir_generated / f"heuristic_matrix_{dataset_loader_name}.pkl"
        applied_matrix, lfs = apply_heuristics_to_dataset(
            heuristics, dataset_loader, task.labels, save_to, config
        )
        stats = analysis_and_metrics(
            applied_matrix,
            lfs,
            dataset_loader,
            dataset_loader_name,
            task,
            config,
            heuristic_module_path,
        )
        all_stats.update(**stats)

    with open(task_dir_metrics / "heuristic_metrics.json", "w") as f:
        json.dump(all_stats, f)

    pprint(all_stats)


def combine_applied_heuristics(task_name: str, config: Config) -> None:
    task = config.tasks[task_name]
    task_dir_generated = config.paths.generated / task_name
    all_stats: Dict[str, Any] = {}
    for dataset_loader_name, dataset_loader in {
        **task.test_datasets,
        **task.train_datasets,
    }.items():
        all_heuristics_file = (
            task_dir_generated / f"heuristic_matrix_{dataset_loader_name}.pkl"
        )
        matrix_list = []
        all_heuristics = []
        for heuristic_module_path in task.heuristic_groups:
            partial_heuristics_file = (
                task_dir_generated
                / heuristic_module_path
                / f"heuristic_matrix_{dataset_loader_name}.pkl"
            )
            matrix = pd.read_pickle(str(partial_heuristics_file))
            matrix_list.append(matrix)
            hs = load_heuristics_from_module(task.top_artifact, heuristic_module_path)
            all_heuristics.extend(hs)
        labeling_functions = to_labeling_functions(
            all_heuristics, dataset_loader.get_mapper(), task.labels
        )

        all_heuristics_matrix = pd.concat(matrix_list, axis=1)
        if sum(all_heuristics_matrix.columns.duplicated()) != 0:
            raise ValueError(
                f"Duplicate heursitics are present: {all_heuristics_matrix.columns}"
            )
        all_heuristics_matrix.to_pickle(str(all_heuristics_file))
        stats = analysis_and_metrics(
            all_heuristics_matrix.to_numpy(),
            labeling_functions,
            dataset_loader,
            dataset_loader_name,
            task,
            config,
        )
        all_stats.update(**stats)

    with open(config.paths.metrics / task.name / "heuristic_metrics.json", "w") as f:
        json.dump(all_stats, f)

    pprint(all_stats)


if __name__ == "__main__":
    config = load_config()

    apply_heuristics("bugginess", config, "heuristics.h.bugginess")
