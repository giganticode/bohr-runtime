from typing import List

import numpy as np
import pandas as pd
from bohrapi.core import Task
from bohrlabels.core import LabelSet
from tqdm import tqdm

from bohrruntime.bohrfs import BohrFileSystem, BohrFsPath
from bohrruntime.core import load_dataset, load_ground_truth_labels
from bohrruntime.data_analysis import calculate_lf_metrics
from bohrruntime.heuristics import get_heuristic_files, get_labeling_functions_from_path
from bohrruntime.labeling.cache import CategoryMappingCache


def calculate_metrics_for_heuristic(task: Task, fs: BohrFileSystem) -> None:
    category_mapping_cache = CategoryMappingCache(list(map(lambda x: str(x), task.labels)), maxsize=10000)
    for dataset in tqdm(task.test_datasets, desc="Calculating metrics for dataset: "):
        # projection = get_projection(heuristics)
        artifact_df = load_dataset(dataset, projection={})
        label_series = load_ground_truth_labels(task, dataset, pre_loaded_artifacts=artifact_df)
        if label_series is not None:
            label_series = np.array(list(map(lambda x: category_mapping_cache[LabelSet.of(x)], label_series)))
        heuristic_groups: List[BohrFsPath] = get_heuristic_files(
            fs.heuristics, task.top_artifact
        )
        for heuristic_group in heuristic_groups:
            labeling_functions = get_labeling_functions_from_path(
                heuristic_group, category_mapping_cache
            )
            if not(labeling_functions):
                raise AssertionError(f'No labeling functions in {heuristic_group.to_absolute_path()}')

            save_to_metrics = fs.single_heuristic_metrics(task, dataset, str(heuristic_group))
            save_to_matrix = fs.heuristic_matrix_file(dataset, str(heuristic_group)).to_absolute_path()
            label_matrix = pd.read_pickle(save_to_matrix).to_numpy()
            calculate_lf_metrics(
                label_matrix, label_series, save_to=save_to_metrics
            )