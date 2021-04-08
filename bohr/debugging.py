import logging
from io import BytesIO
from typing import Dict, Optional, Tuple

import dvc.api
import numpy as np
import pandas as pd
from tabulate import tabulate

from bohr.pathconfig import load_path_config

logging.getLogger()


def _normalize_weights(x: np.ndarray) -> np.ndarray:
    """
    >>> _normalize_weights(np.array([[0.019144, 0.011664], [0.005839, 0.004232], [0.003673, 0.003746], [0.020836, 0.011061]]))
    array([[0.23081334, 0.05360646],
           [0.17752456, 0.04366187],
           [0.1628473 , 0.04270857],
           [0.23586317, 0.05297473]])
    >>> _normalize_weights(np.array([[0.71, 0.84], [0.84, 0.71]]))
    array([[0.16867129, 0.33132871],
           [0.33132871, 0.16867129]])
    """
    prod = np.exp(np.sum(np.log(x), axis=0))
    NBB = prod / np.sum(prod)
    x_modif = prod * np.log(prod) / np.log(x)
    norm_koef_x = NBB / np.sum(x_modif, axis=0)

    x_normalized = x_modif * norm_koef_x

    return x_normalized


def _load_output_matrix_and_weights(
    task_name: str,
    labeled_dataset: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_config = load_path_config()
    logging.disable(logging.WARNING)
    with dvc.api.open(
        path_config.generated_dir
        / task_name
        / f"heuristic_matrix_{labeled_dataset}.pkl",
        repo,
        rev=rev,
        mode="rb",
    ) as f:
        matrix = pd.read_pickle(BytesIO(f.read()))

    with dvc.api.open(
        path_config.generated_dir / task_name / f"label_model_weights.csv",
        repo,
        rev=rev,
    ) as f:
        weights = pd.read_csv(f, index_col="heuristic_name")
    logging.disable(logging.NOTSET)
    return matrix, weights


class DataPointDebugger:
    def __init__(
        self, task_name: str, labeled_dataset: str, rev: Optional[str] = "master"
    ):
        self.dataset_debugger = DatasetDebugger(task_name, labeled_dataset, rev)
        self.old_matrix, self.old_weights = _load_output_matrix_and_weights(
            task_name, labeled_dataset, "https://github.com/giganticode/bohr", rev
        )
        self.new_matrix, self.new_weights = _load_output_matrix_and_weights(
            task_name, labeled_dataset
        )

    def get_datapoint_info_(
        self, matrix, weights, datapoint: int, suffix: str
    ) -> pd.DataFrame:
        row = matrix.iloc[datapoint]

        zero = f"NonBug_{suffix}"
        one = f"Bug_{suffix}"
        out = f"h_output_{suffix}"

        weights[out] = row
        weights = weights[weights[out] > -1]
        if weights.empty:
            print("No heuristic fired.")
            return pd.DataFrame([], columns=[zero, one, out])
        weights2 = weights.apply(
            lambda row: row[["00", "01"]].rename(({"00": zero, "01": one}))
            if row[out] == 0
            else row[["10", "11"]].rename(({"10": zero, "11": one})),
            axis=1,
        )
        weights2[[out]] = weights[[out]]
        weights2[[zero, one]] = pd.DataFrame(
            _normalize_weights(weights2[[zero, one]].to_numpy()), index=weights.index
        )
        return weights2

    def show_datapoint_info(self, datapoint: int) -> None:
        self.dataset_debugger.show_datapoint(datapoint)
        old = self.get_datapoint_info_(
            self.old_matrix, self.old_weights, datapoint, "old"
        )
        new = self.get_datapoint_info_(
            self.new_matrix, self.new_weights, datapoint, "new"
        )
        concat_df = pd.concat([old, new], axis=1)
        print(tabulate(concat_df, headers=concat_df.columns))

    def get_fired_heuristics_with_weights(self) -> Dict[str, float]:
        pass


class DatasetDebugger:
    def __init__(
        self, task: str, labeled_dataset_path: str, rev: Optional[str] = "master"
    ):
        path_config = load_path_config()
        logging.disable(logging.WARNING)
        with dvc.api.open(
            path_config.labeled_data_dir / f"{labeled_dataset_path}.labeled.csv",
            "https://github.com/giganticode/bohr",
            rev=rev,
        ) as f:
            old_df = pd.read_csv(f)

        with dvc.api.open(
            path_config.labeled_data_dir / f"{labeled_dataset_path}.labeled.csv"
        ) as f:
            new_df = pd.read_csv(f)
        logging.disable(logging.NOTSET)

        self.combined_df = pd.concat(
            [
                old_df[["bug", "prob_CommitLabel.BugFix"]],
                new_df["prob_CommitLabel.BugFix"].rename("prob_CommitLabel.BugFix_new"),
            ],
            axis=1,
        )
        self.combined_df.loc[:, "improvement"] = (
            self.combined_df["prob_CommitLabel.BugFix_new"]
            - self.combined_df["prob_CommitLabel.BugFix"]
        ) * (self.combined_df["bug"] * 2 - 1)
        self.combined_df = pd.concat([self.combined_df, old_df["message"]], axis=1)
        self.combined_df.sort_values(by="improvement", inplace=True)

    def _show_datapoints(self, df: pd.DataFrame) -> None:
        pd.options.mode.chained_assignment = None
        df.loc[:, "message"] = df["message"].str.wrap(70)
        print(
            tabulate(
                df,
                headers=df.columns,
                tablefmt="fancy_grid",
            )
        )

    def show_worst_datapoints(self, n: Optional[int] = 10) -> None:
        worst_datapoints = self.combined_df.head(n)
        self._show_datapoints(worst_datapoints)

    def show_best_datapoints(self, n: Optional[int] = 10) -> None:
        worst_datapoints = self.combined_df.tail(n)
        self._show_datapoints(worst_datapoints)

    def show_datapoint(self, n: int) -> None:
        a = self.combined_df.loc[n].to_frame()
        print(tabulate(a))

    def debug_datapoint(self, n: int):
        pass
