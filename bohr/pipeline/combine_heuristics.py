from pprint import pprint
from typing import Optional

import pandas as pd

from bohr.config.pathconfig import PathConfig
from bohr.core import to_labeling_functions
from bohr.datamodel.heuristic import load_heuristics_from_module
from bohr.datamodel.task import Task
from bohr.pipeline.data_analysis import calculate_metrics, run_analysis


def combine_applied_heuristics(
    task: Task, path_config: Optional[PathConfig] = None
) -> None:

    path_config = path_config or PathConfig.load()
    task_dir_generated = path_config.generated / task.name
    for dataset_name, dataset in task.datasets.items():
        all_heuristics_file = (
            task_dir_generated / f"heuristic_matrix_{dataset_name}.pkl"
        )
        matrix_list = []
        all_heuristics = []
        for heuristic_module_path in task.heuristic_groups:
            partial_heuristics_file = (
                task_dir_generated
                / heuristic_module_path
                / f"heuristic_matrix_{dataset_name}.pkl"
            )
            matrix = pd.read_pickle(str(partial_heuristics_file))
            matrix_list.append(matrix)
            heuristics = load_heuristics_from_module(
                task.top_artifact, heuristic_module_path
            )
            all_heuristics.extend(heuristics)
        labeling_functions = to_labeling_functions(
            all_heuristics, dataset.mapper, task.labels
        )
        all_heuristics_matrix = pd.concat(matrix_list, axis=1)
        if sum(all_heuristics_matrix.columns.duplicated()) != 0:
            s = set()
            for c in all_heuristics_matrix.columns:
                if c in s:
                    raise ValueError(f"Duplicate heuristics are present: {c}")
                s.add(c)
            raise AssertionError()
        all_heuristics_matrix.to_pickle(str(all_heuristics_file))
        artifact_df = dataset.load()
        label_series = (
            artifact_df[task.label_column_name]
            if task.label_column_name in artifact_df.columns
            else None
        )
        save_csv_to = path_config.generated / task.name / f"analysis_{dataset_name}.csv"
        save_json_to = path_config.metrics / task.name / f"analysis_{dataset_name}.json"
        save_metrics_to = (
            path_config.metrics / task.name / f"heuristic_metrics_{dataset_name}.json"
        )

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

        pprint(stats)
