from typing import Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Task
from bohrlabels.core import LabelSet

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import load_dataset
from bohrruntime.labeling.cache import CategoryMappingCache
from bohrruntime.pipeline.apply_heuristics import (
    get_labeling_functions_from_path,
    load_ground_truth_labels,
)
from bohrruntime.pipeline.data_analysis import calculate_metrics


def calculate_metrics_for_heuristic(task: Task, heuristic_group: str, dataset: Dataset, path_config: Optional[PathConfig] = None) -> None:
    # projection = get_projection(heuristics)
    artifact_df = load_dataset(dataset, projection={})
    label_series = load_ground_truth_labels(task, dataset, pre_loaded_artifacts=artifact_df)
    category_mapping_cache = CategoryMappingCache(list(map(lambda x: str(x), task.labels)), maxsize=10000)
    heuristic_file = path_config.heuristics / heuristic_group
    labeling_functions = get_labeling_functions_from_path(
        heuristic_file, category_mapping_cache
    )
    if not(labeling_functions):
        raise AssertionError(f'No labeling functions in {heuristic_file}')
    if label_series is not None:
        label_series = np.array(list(map(lambda x: category_mapping_cache[LabelSet.of(x)], label_series))) #TODO code duplication?
    save_to_metrics = path_config.single_heuristic_metrics(task, dataset, heuristic_group)
    save_to_matrix = path_config.heuristic_matrix_file(dataset, heuristic_group)
    applied_lf_matrix = pd.read_pickle(save_to_matrix).to_numpy()
    calculate_metrics(
        applied_lf_matrix, labeling_functions, label_series, save_to=save_to_metrics
    )