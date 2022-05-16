import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from shutil import copy
from time import time
from typing import Dict, Optional, Tuple

import appdirs
import git
import numpy as np
import pandas as pd
from dvc.exceptions import FileMissingError
from git import Repo
from tabulate import tabulate

from bohrruntime import appauthor, appname, version
from bohrruntime.bohrfs import BohrFileSystem, BohrFsPath
from bohrruntime.formatting import tabulate_artifacts
from bohrruntime.util.paths import AbsolutePath
from bohrruntime.workspace import load_workspace

logger = logging.getLogger()


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
    exp: str,
    labeled_dataset: str,
    rev: Optional[str] = None,
    force_update: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fs = BohrFileSystem.init()
    logging.disable(logging.WARNING)
    repo = (
        get_path_to_revision(fs.root, rev, force_update=force_update)
        if rev is not None
        else None
    )
    import dvc.api

    workspace = load_workspace()
    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.task.get_dataset_by_id(labeled_dataset)

    with dvc.api.open(
        str(fs.exp_dataset_dir(exp, dataset) / f"heuristic_matrix.pkl"),
        repo,
        mode="rb",
    ) as f:
        matrix = pd.read_pickle(BytesIO(f.read()))

    with dvc.api.open(
        str(fs.exp_dir(exp) / f"label_model_weights.csv"),
        repo,
    ) as f:
        weights = pd.read_csv(f, index_col="heuristic_name")
    logging.disable(logging.NOTSET)
    return matrix, weights


class DataPointDebugger:
    def __init__(
        self,
        exp: str,
        labeled_dataset: str,
        dataset_debugger: "DatasetDebugger",
        rev: Optional[str] = "master",
        force_update: bool = False,
    ):
        self.dataset_debugger = dataset_debugger
        try:
            self.old_matrix, self.old_weights = _load_output_matrix_and_weights(
                exp, labeled_dataset, rev, force_update=force_update
            )
        except FileMissingError:
            self.old_matrix, self.old_weights = None, None
        self.new_matrix, self.new_weights = _load_output_matrix_and_weights(
            exp, labeled_dataset
        )

    @staticmethod
    def get_datapoint_info_(
        matrix: pd.DataFrame, weights: pd.DataFrame, datapoint: int, suffix: str
    ) -> pd.DataFrame:
        """
        >>> matrix = pd.DataFrame([[0, -1, 0],[1, -1, -1]], columns=['h1', 'h2', 'h3'])
        >>> weights = pd.DataFrame([[0.01, 0.99, 0.3, 0.3],[0.2, 0.2, 0.3, 0.3],[0.2, 0.2, 0.3, 0.3]], columns=['00', '01', '10', '11'], index=pd.Series(['h1', 'h2', 'h3'], name='heuristic_name'))
        >>> DataPointDebugger.get_datapoint_info_(matrix, weights, 0, 'new') # doctest: +NORMALIZE_WHITESPACE
                        NonBug_new  Bug_new  h_output_new
        heuristic_name
        h1                    0.01     0.99             0
        h3                    0.20     0.20             0
        """
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
        # weights2[[zero, one]] = pd.DataFrame(
        #     _normalize_weights(weights2[[zero, one]].to_numpy()), index=weights.index
        # )
        return weights2
        # return weights

    def show_datapoint_info(self, datapoint: int) -> None:
        self.dataset_debugger.show_datapoint(datapoint)
        if self.old_matrix is not None:
            old = DataPointDebugger.get_datapoint_info_(
                self.old_matrix, self.old_weights, datapoint, "old"
            )
        else:
            old = None
        new = DataPointDebugger.get_datapoint_info_(
            self.new_matrix, self.new_weights, datapoint, "new"
        )
        if old is not None:
            concat_df = pd.concat([old, new], axis=1)
        else:
            concat_df = new
        print(tabulate(concat_df, headers=concat_df.columns))

    def get_fired_heuristics_with_weights(self) -> Dict[str, float]:
        pass


def get_cloned_rev(repo: str, rev: str = "master") -> Repo:
    path = appdirs.user_cache_dir(
        appname=appname(), appauthor=appauthor(), version=version()
    )
    path_to_repo = Path(path) / repo / rev
    if not path_to_repo.exists():
        return Repo.clone_from(repo, path_to_repo, depth=1, b=rev)
    else:
        return Repo(path_to_repo)


def get_git_repo_of(path: AbsolutePath) -> Repo:
    current_path = path
    while True:
        try:
            return Repo(current_path)
        except git.exc.InvalidGitRepositoryError:
            if current_path == current_path.parent:
                raise ValueError(f"Path {path} or its parents are not a git repo!")
            current_path = current_path.parent


def is_update_needed(git_revision: Repo) -> bool:
    fetch_head_file = Path(git_revision.working_tree_dir) / ".git" / "FETCH_HEAD"
    if not fetch_head_file.exists():
        return True

    last_modification = fetch_head_file.stat().st_mtime
    updated_sec_ago = time() - last_modification
    logger.debug(
        f"Repo {git_revision} last attempt to pull {datetime.fromtimestamp(last_modification)}"
    )
    return updated_sec_ago > 300


def update(
    repo: Repo, dvc_project_root: AbsolutePath, rel_path_to_dvc_root: BohrFsPath
) -> None:
    logger.info("Updating the repo... ")
    repo.remotes.origin.pull()
    move_local_config_to_old_revision(
        dvc_project_root, repo.working_tree_dir / str(rel_path_to_dvc_root)
    )


def move_local_config_to_old_revision(src: AbsolutePath, dst: AbsolutePath):
    config_path = Path(".dvc") / "config.local"
    if (src / config_path).exists():
        logger.debug(f"Copying config from {src / config_path} to {dst / config_path}")
        copy(src / config_path, dst / config_path)


def get_path_to_revision(
    dvc_project_root: AbsolutePath, rev: str, force_update: bool = False
) -> Optional[AbsolutePath]:
    current_repo = get_git_repo_of(dvc_project_root)
    if len(current_repo.remotes) == 0:
        return None
    remote = current_repo.remote()
    old_revision: Repo = get_cloned_rev(remote.url, rev)
    dvc_root: BohrFsPath = BohrFsPath.from_absolute_path(
        dvc_project_root, current_repo.working_tree_dir
    )
    logger.info(
        f"Comparing to {remote.url} , revision: {rev}, \n"
        f"relative dvc root: {dvc_root}\n"
        f"(Cloned to {old_revision.working_tree_dir})"
    )
    if is_update_needed(old_revision) or force_update:
        if force_update:
            logger.debug("Forcing refresh ...")
        update(old_revision, dvc_project_root, dvc_root)
    else:
        logger.info(f"Pass `--force-refresh` to refresh the repository.")
    return old_revision.working_tree_dir / dvc_root


class DatasetDebugger:
    def __init__(
        self,
        exp: str,
        labeled_dataset_name: str,
        rev: Optional[str] = "master",
        force_update: bool = False,
    ):
        workspace = load_workspace()
        exp = workspace.get_experiment_by_name(exp)
        dataset = exp.task.get_dataset_by_id(labeled_dataset_name)

        fs = BohrFileSystem.init()
        path_to_old_revision = get_path_to_revision(fs.root, rev, force_update)
        self.labeled_dataset_name = labeled_dataset_name
        logging.disable(logging.WARNING)
        labeled_dataset_path = str(fs.exp_dataset_dir(exp, dataset) / "labeled.csv")
        import dvc.api

        label_to_int = {str(label): i for i, label in enumerate(exp.task.labels)}

        old_df = None
        if path_to_old_revision is not None:
            logger.warning(
                f"Reading dvc file {labeled_dataset_path}, repo = {path_to_old_revision}"
            )
            try:
                with dvc.api.open(labeled_dataset_path, path_to_old_revision) as f:
                    old_df = pd.read_csv(f)
            except FileMissingError:
                old_df = None
        logger.warning(f"Reading dvc file {labeled_dataset_path}")
        with dvc.api.open(labeled_dataset_path) as f:
            new_df = pd.read_csv(f)
        logging.disable(logging.NOTSET)

        self.is_test_set = "label" in new_df.columns

        self.result_df = pd.DataFrame()
        self.result_df.loc[:, ["sha", "message"]] = new_df[["sha", "message"]]
        if self.is_test_set:
            self.result_df.loc[:, "ground_truth"] = new_df["label"]
        if old_df is not None:
            self.result_df.loc[:, "ground_truth_old"] = old_df["label"]
            self.result_df.loc[:, "old_label"] = old_df["prob_CommitLabel.BugFix"]

        self.result_df.loc[:, "new_label"] = new_df["prob_CommitLabel.BugFix"]

        self.result_df.loc[:, "certainty"] = (
            np.abs(self.result_df["new_label"] - 0.5) * 2
        )
        if self.is_test_set:
            self.result_df.loc[:, "precision"] = self.result_df.apply(
                lambda row: 1
                - np.abs(row["new_label"] - label_to_int[row["ground_truth"]]),
                axis=1,
            )
            if old_df is not None:
                self.result_df.loc[:, "improvement"] = self.result_df.apply(
                    lambda row: (row["new_label"] - row["old_label"])
                    * (label_to_int[row["ground_truth"]] * 2 - 1),
                    axis=1,
                )
                pass

        if "url" in new_df.columns:
            self.result_df["url"] = new_df["url"]

        # tabulate_artifacts(new_df)

    def show_datapoints(
        self,
        value: str,
        n: Optional[int] = 10,
        reverse: bool = False,
    ) -> None:
        if not self.is_test_set and value in ["improvement", "precision", "no"]:
            raise ValueError(
                f"Dataset {self.labeled_dataset_name} has no ground-truth labels. "
                f"Cannot calculate {value}"
            )
        if value != "no":
            self.result_df.sort_values(by=value, inplace=True, ascending=reverse)
        df = self.result_df.head(n)
        tabulate_artifacts(df)

    def show_datapoint(self, n: int) -> None:
        a = self.result_df.loc[n].to_frame()
        print(tabulate(a))

    def debug_datapoint(self, n: int):
        pass
