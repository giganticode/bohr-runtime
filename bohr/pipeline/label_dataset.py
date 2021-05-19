from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import Series
from snorkel.labeling.model import LabelModel

from bohr.config.pathconfig import PathConfig
from bohr.datamodel.dataset import Dataset
from bohr.datamodel.task import Task


def label_dataset(
    task: Task,
    dataset: Dataset,
    path_config: Optional[PathConfig] = None,
    debug: bool = False,
):
    path_config = path_config or PathConfig.load()

    applied_heuristics_df = pd.read_pickle(
        str(path_config.generated / task.name / f"heuristic_matrix_{dataset.name}.pkl")
    )

    label_model = LabelModel()
    label_model.load(str(path_config.generated / task.name / "label_model.pkl"))
    df = dataset.load()
    df_labeled = do_labeling(
        label_model, applied_heuristics_df.to_numpy(), df, task.labels
    )

    if debug:
        for (
            heuristic_name,
            applied_heuristic_series,
        ) in applied_heuristics_df.iteritems():
            applied_heuristics_df[heuristic_name] = applied_heuristic_series.map(
                {0: heuristic_name, 1: heuristic_name, -1: ""}
            )
        col_lfs = applied_heuristics_df.apply(
            lambda row: ";".join([elm for elm in row if elm]), axis=1
        )
        df_labeled["lfs"] = col_lfs

    labeled_data_path = path_config.labeled_data / task.name
    if not labeled_data_path.exists():
        labeled_data_path.mkdir(parents=True)
    target_file = labeled_data_path / f"{dataset.name}.labeled.csv"
    df_labeled.to_csv(target_file, index=False)
    print(f"Labeled dataset has been written to {target_file}.")


def do_labeling(
    label_model: LabelModel,
    matrix: np.ndarray,
    df: pd.DataFrame,
    label_names: List[str],
) -> pd.DataFrame:
    labels, probs = label_model.predict(L=matrix, return_probs=True)
    probs = np.around(probs, decimals=2)
    df_labeled = df.assign(predicted=Series(labels))

    df_labeled[f"prob_{label_names[0]}"] = Series(probs[:, 0])
    df_labeled[f"prob_{label_names[1]}"] = Series(probs[:, 1])
    df_labeled["prob_class"] = Series(np.around(np.copy(probs[:, 1]), decimals=1))
    return df_labeled
