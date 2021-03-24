from typing import List

import numpy as np
import pandas as pd
from pandas import Series
from snorkel.labeling.model import LabelModel

from bohr.config import Config, get_dataset_loader


def label_dataset(
    task_name: str, dataset_name: str, config: Config, debug: bool = False
):
    task = config.tasks[task_name]
    dataset_loader = config.get_dataloader(dataset_name)

    applied_heuristics_df = pd.read_pickle(
        str(config.paths.generated / task.name / f"heuristic_matrix_{dataset_name}.pkl")
    )

    label_model = LabelModel()
    label_model.load(str(config.paths.generated / task_name / "label_model.pkl"))
    df = dataset_loader.load(config.project_root)
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

    labeled_data_path = config.paths.labeled_data
    if not labeled_data_path.exists():
        labeled_data_path.mkdir(parents=True)
    target_file = labeled_data_path / f"{dataset_name}.labeled.csv"
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
