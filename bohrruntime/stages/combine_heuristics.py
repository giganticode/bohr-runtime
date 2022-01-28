from typing import Optional

import pandas as pd
from bohrapi.core import Dataset, Experiment

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.heuristics import get_heuristic_files, load_heuristics_from_file


def check_duplicate_heuristics(all_heuristics_matrix: pd.DataFrame):
    if sum(all_heuristics_matrix.columns.duplicated()) != 0:
        s = set()
        for c in all_heuristics_matrix.columns:
            if c in s:
                raise ValueError(f"Duplicate heuristics are present: {c}")
            s.add(c)
        raise AssertionError()


def combine_applied_heuristics(
    exp: Experiment, dataset: Dataset, fs: Optional[BohrFileSystem] = None
) -> None:

    fs = fs or BohrFileSystem.init()
    dataset_dir = fs.exp_dataset_dir(exp, dataset).to_absolute_path()
    all_heuristics_file = dataset_dir / f"heuristic_matrix.pkl"
    matrix_list = []
    all_heuristics = []
    for heuristic_group in (
        exp.heuristic_groups if exp.heuristic_groups is not None else ["."]
    ):
        for heuristic_module_path in get_heuristic_files(
            fs.heuristics / heuristic_group,
            exp.task.top_artifact,
            with_anchor=fs.heuristics.to_absolute_path(),
        ):
            partial_heuristics_file = fs.heuristic_matrix_file(
                dataset,
                str(heuristic_module_path),
            ).to_absolute_path()
            matrix = pd.read_pickle(str(partial_heuristics_file))
            matrix_list.append(matrix)
            heuristics = load_heuristics_from_file(
                heuristic_module_path, exp.task.top_artifact
            )
            all_heuristics.extend(heuristics)

    all_heuristics_matrix = pd.concat(matrix_list, axis=1)
    check_duplicate_heuristics(all_heuristics_matrix)
    all_heuristics_matrix.to_pickle(str(all_heuristics_file))
