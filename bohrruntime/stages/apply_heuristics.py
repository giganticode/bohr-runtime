from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset
from bohrlabels.labels import CommitLabel
from snorkel.labeling import LabelingFunction
from snorkel.labeling.apply.core import LFApplier

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import load_dataset
from bohrruntime.heuristics import get_labeling_functions_from_path


def apply_lfs_to_dataset(
    lfs: List[LabelingFunction], artifacts: List[Dict]
) -> np.ndarray:
    applier = LFApplier(lfs=lfs)
    applied_lf_matrix = applier.apply(artifacts)
    return applied_lf_matrix


def apply_heuristics_to_dataset(
    heuristic_group: str, dataset: Dataset, fs: BohrFileSystem
) -> None:
    save_to_matrix = fs.heuristic_matrix_file(
        dataset, heuristic_group
    ).to_absolute_path()

    heuristic_file = fs.heuristics / heuristic_group
    labeling_functions = get_labeling_functions_from_path(heuristic_file)
    if not labeling_functions:
        raise AssertionError(f"No labeling functions for in {heuristic_file}")
    artifact_df = load_dataset(dataset)
    applied_lf_matrix = apply_lfs_to_dataset(labeling_functions, artifacts=artifact_df)

    df = pd.DataFrame(applied_lf_matrix, columns=[lf.name for lf in labeling_functions])
    df.to_pickle(str(save_to_matrix))
