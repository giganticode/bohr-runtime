from typing import List

import numpy as np
import pandas as pd
from bohrlabels.core import to_numeric_label
from tqdm import tqdm

from bohrruntime.bohrfs import BohrFileSystem, BohrFsPath
from bohrruntime.data_analysis import calculate_lf_metrics
from bohrruntime.heuristics import get_heuristic_files, get_labeling_functions_from_path
from bohrruntime.labeling.cache import CategoryMappingCache, map_numeric_label_value
from bohrruntime.task import Task


def calculate_metrics_for_heuristic(task: Task, fs: BohrFileSystem) -> None:
    category_mapping_cache = CategoryMappingCache(task.labels, maxsize=10000)
    for dataset, datapoint_to_label_func in tqdm(task.test_datasets.items(), desc="Calculating metrics for dataset: "):
        if datapoint_to_label_func is not None:
            label_series = dataset.load_ground_truth_labels(datapoint_to_label_func)
            label_series = np.array(
                list(
                    map(lambda x: category_mapping_cache[to_numeric_label(x, task.hierarchy)], label_series)
                )
            )
        else:
            label_series = None
        heuristic_groups: List[BohrFsPath] = get_heuristic_files(
            fs.heuristics, task.top_artifact
        )
        for heuristic_group in heuristic_groups:
            labeling_functions = get_labeling_functions_from_path(heuristic_group)
            if not(labeling_functions):
                raise AssertionError(f'No labeling functions in {heuristic_group.to_absolute_path()}')

            save_to_metrics = fs.single_heuristic_metrics(task, dataset, str(heuristic_group))
            save_to_matrix = fs.heuristic_matrix_file(dataset, str(heuristic_group)).to_absolute_path()
            label_matrix = pd.read_pickle(save_to_matrix)
            label_matrix = label_matrix.applymap(lambda v: map_numeric_label_value(v, category_mapping_cache, task))
            label_matrix = label_matrix.to_numpy()
            calculate_lf_metrics(
                label_matrix, label_series, save_to=save_to_metrics
            )