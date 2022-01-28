from typing import List, Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Experiment
from pandas import Series
from snorkel.labeling.model import LabelModel

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import load_dataset


def label_dataset(
    exp: Experiment,
    dataset: Dataset,
    path_config: Optional[BohrFileSystem] = None,
    debug: bool = False,
):
    path_config = path_config or BohrFileSystem.init()
    task_dir = path_config.exp_dir(exp)
    dataset_dir = path_config.exp_dataset_dir(exp, dataset).to_absolute_path()

    label_matrix = pd.read_pickle(dataset_dir / f"heuristic_matrix.pkl")

    label_model = LabelModel()
    label_model.load(str(task_dir.to_absolute_path() / "label_model.pkl"))
    artifact_list = load_dataset(dataset)
    if (
        dataset in exp.task.test_datasets
        and exp.task.test_datasets[dataset] is not None
    ):
        label_from_datapoint_func = exp.task.test_datasets[dataset]
        df = pd.DataFrame(
            [
                [c.raw_data["_id"], c.raw_data["message"], label_from_datapoint_func(c)]
                for c in artifact_list
            ],
            columns=["sha", "message", "label"],
        )
    else:
        df = pd.DataFrame(
            [[c.raw_data["_id"], c.raw_data["message"]] for c in artifact_list],
            columns=["sha", "message"],
        )

    df_labeled = do_labeling(label_model, label_matrix.to_numpy(), df, exp.task.labels)

    if debug:
        label_model_weights_file = (
            path_config.exp_dir(exp).to_absolute_path() / f"label_model_weights.csv"
        )
        weights = pd.read_csv(label_model_weights_file, index_col="heuristic_name")

        for (
            heuristic_name,
            applied_heuristic_series,
        ) in label_matrix.iteritems():
            weights_for_heuristic = np.around(
                weights.loc[heuristic_name, :], decimals=3
            )
            formatted_weights = f'({weights_for_heuristic["00"]}/{weights_for_heuristic["01"]})__({weights_for_heuristic["10"]}/{weights_for_heuristic["11"]})'
            column_name = f"{heuristic_name}__{formatted_weights}"
            df_labeled[column_name] = applied_heuristic_series

            cond_weights = weights.apply(
                lambda row: row["01"]
                if row[column_name] == 0
                else (row["11"] if row[column_name] == 0 else 1.0),
                axis=1,
            )
            weights2[[out]] = weights[[out]]
            weights2[[zero, one]] = pd.DataFrame(
                _normalize_weights(weights2[[zero, one]].to_numpy()),
                index=weights.index,
            )

    dataset_dir = path_config.exp_dataset_dir(exp, dataset).to_absolute_path()
    target_file = dataset_dir / "labeled.csv"
    df_labeled.to_csv(target_file, index=False)
    print(f"Labeled dataset has been written to {target_file}.")


def do_labeling(
    label_model: LabelModel,
    matrix: np.ndarray,
    artifacts: pd.DataFrame,
    label_names: List[str],
) -> pd.DataFrame:
    labels, probs = label_model.predict(L=matrix, return_probs=True)
    probs = np.around(probs, decimals=2)
    df_labeled = artifacts.assign(predicted=Series(labels))

    # df_labeled[f"prob_{label_names[0]}"] = Series(probs[:, 0])
    df_labeled[f"prob_{label_names[1]}"] = Series(probs[:, 1])
    # df_labeled["prob_class"] = Series(np.around(np.copy(probs[:, 1]), decimals=1))
    return df_labeled
