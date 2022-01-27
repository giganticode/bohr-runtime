from typing import Optional

import pandas as pd
from bohrapi.core import Dataset, Experiment

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.heuristics import get_heuristic_files, load_heuristics_from_file
from bohrruntime.util.paths import relative_to_safe


def check_duplicate_heuristics(all_heuristics_matrix: pd.DataFrame):
    if sum(all_heuristics_matrix.columns.duplicated()) != 0:
        s = set()
        for c in all_heuristics_matrix.columns:
            if c in s:
                raise ValueError(f"Duplicate heuristics are present: {c}")
            s.add(c)
        raise AssertionError()


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

    all_heuristics_matrix = pd.concat(matrix_list, axis=1)
    check_duplicate_heuristics(all_heuristics_matrix)
    all_heuristics_matrix.to_pickle(str(all_heuristics_file))
