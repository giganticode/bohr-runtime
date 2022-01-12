from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Task
from bohrlabels.core import Label
from bohrlabels.labels import CommitLabel
from snorkel.labeling import LabelingFunction
from snorkel.labeling.apply.core import LFApplier

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import load_dataset, to_labeling_function, to_labeling_functions
from bohrruntime.heuristics import load_heuristic_by_name, load_heuristics_from_file
from bohrruntime.labeling.cache import CategoryMappingCache
from bohrruntime.util.paths import AbsolutePath


def load_ground_truth_labels(
    task: Task, dataset: Dataset, pre_loaded_artifacts: Optional[pd.DataFrame] = None
) -> Optional[Sequence[Label]]:
    if pre_loaded_artifacts is None:
        pre_loaded_artifacts = load_dataset(dataset)
    if dataset in task.test_datasets and task.test_datasets[dataset] is not None:
        label_from_datapoint_function = task.test_datasets[dataset]
        label_series = [
            label_from_datapoint_function(artifact) for artifact in pre_loaded_artifacts
        ]
    else:
        label_series = None
    return label_series


def apply_lfs_to_dataset(
    lfs: List[LabelingFunction], artifacts: List[Dict]
) -> np.ndarray:
    applier = LFApplier(lfs=lfs)
    applied_lf_matrix = applier.apply(artifacts)
    return applied_lf_matrix


def get_labeling_functions_from_path(
    heuristic_file: AbsolutePath, category_mapping_cache
) -> List[LabelingFunction]:
    heuristics = load_heuristics_from_file(heuristic_file)
    labeling_functions = to_labeling_functions(heuristics, category_mapping_cache)
    return labeling_functions


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
    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), task.labels)), maxsize=10000
    )
    labeling_function = to_labeling_function(
        heuristic, dataset.top_artifact, category_mapping_cache
    )
    apply_lf_matrix = apply_lfs_to_dataset(
        [labeling_function], artifacts=load_dataset(dataset, n_datapoints=n_datapoints)
    )
    return apply_lf_matrix


def apply_heuristics_to_dataset(
    heuristic_group: str,
    dataset: Dataset,
    path_config: Optional[PathConfig] = None,
) -> None:
    path_config = path_config or PathConfig.load()

    save_to_matrix = path_config.heuristic_matrix_file(dataset, heuristic_group)

    heuristic_file = path_config.heuristics / heuristic_group
    # heuristics = load_heuristics_from_path(heuristic_file, task.top_artifact)
    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), [CommitLabel.NonBugFix, CommitLabel.BugFix])),
        maxsize=10000,
    )  # FIXME this should not be bugginess task-specific
    labeling_functions = get_labeling_functions_from_path(
        heuristic_file, category_mapping_cache
    )
    if not labeling_functions:
        raise AssertionError(f"No labeling functions for in {heuristic_file}")
    # projection = get_projection(heuristics)
    artifact_df = load_dataset(dataset, projection={})
    applied_lf_matrix = apply_lfs_to_dataset(labeling_functions, artifacts=artifact_df)

    df = pd.DataFrame(applied_lf_matrix, columns=[lf.name for lf in labeling_functions])
    df.to_pickle(str(save_to_matrix))
