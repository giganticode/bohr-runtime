from typing import Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Experiment
from bohrlabels.core import LabelSet

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import load_ground_truth_labels, to_labeling_functions
from bohrruntime.data_analysis import calculate_metrics, run_analysis
from bohrruntime.heuristics import get_heuristic_files, load_heuristics_from_file
from bohrruntime.labeling.cache import CategoryMappingCache
from bohrruntime.util.paths import relative_to_safe


def combine_applied_heuristics(
    exp: Experiment, dataset: Dataset, path_config: Optional[PathConfig] = None
) -> None:

    path_config = path_config or PathConfig.load()
    dataset_dir = path_config.exp_dataset_dir(exp, dataset)
    all_heuristics_file = dataset_dir / f"heuristic_matrix.pkl"
    matrix_list = []
    all_heuristics = []
    for heuristic_group in (
        exp.heuristic_groups if exp.heuristic_groups is not None else ["."]
    ):
        for heuristic_module_path in get_heuristic_files(
            path_config.heuristics / heuristic_group, exp.task.top_artifact
        ):
            partial_heuristics_file = path_config.heuristic_matrix_file(
                dataset,
                str(relative_to_safe(heuristic_module_path, path_config.heuristics)),
            )
            matrix = pd.read_pickle(str(partial_heuristics_file))
            matrix_list.append(matrix)
            heuristics = load_heuristics_from_file(
                heuristic_module_path, exp.task.top_artifact
            )
            all_heuristics.extend(heuristics)

    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), exp.task.labels)), maxsize=10000
    )
    labeling_functions = to_labeling_functions(all_heuristics, category_mapping_cache)
    all_heuristics_matrix = pd.concat(matrix_list, axis=1)
    if sum(all_heuristics_matrix.columns.duplicated()) != 0:
        s = set()
        for c in all_heuristics_matrix.columns:
            if c in s:
                raise ValueError(f"Duplicate heuristics are present: {c}")
            s.add(c)
        raise AssertionError()
    all_heuristics_matrix.to_pickle(str(all_heuristics_file))
    label_series = load_ground_truth_labels(exp.task, dataset)
    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), exp.task.labels)), maxsize=10000
    )
    if label_series is not None:
        label_series = np.array(
            list(map(lambda x: category_mapping_cache[LabelSet.of(x)], label_series))
        )
        label_series = np.array(label_series)
    save_csv_to = dataset_dir / f"analysis.csv"
    save_json_to = dataset_dir / f"analysis.json"
    save_metrics_to = dataset_dir / f"heuristic_metrics.json"

    run_analysis(
        all_heuristics_matrix.to_numpy(),
        labeling_functions,
        save_csv_to,
        save_json_to,
        label_series,
    )

    stats = calculate_metrics(
        all_heuristics_matrix.to_numpy(),
        labeling_functions,
        label_series,
        save_to=save_metrics_to,
    )
