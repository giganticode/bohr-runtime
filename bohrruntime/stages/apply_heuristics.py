from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset
from bohrlabels.labels import CommitLabel
from snorkel.labeling import LabelingFunction
from snorkel.labeling.apply.core import LFApplier

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import load_dataset
from bohrruntime.heuristics import get_labeling_functions_from_path
from bohrruntime.labeling.cache import CategoryMappingCache


def apply_lfs_to_dataset(
    lfs: List[LabelingFunction], artifacts: List[Dict]
) -> np.ndarray:
    applier = LFApplier(lfs=lfs)
    applied_lf_matrix = applier.apply(artifacts)
    return applied_lf_matrix


def apply_heuristics_to_dataset(
    heuristic_group: str,
    dataset: Dataset,
    path_config: Optional[PathConfig] = None,
) -> None:
    path_config = path_config or PathConfig.load()

    save_to_matrix = path_config.heuristic_matrix_file(dataset, heuristic_group)

    heuristic_file = path_config.heuristics / heuristic_group
    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), [CommitLabel.NonBugFix, CommitLabel.BugFix])),
        maxsize=10000,
    )  # FIXME this should not be bugginess task-specific
    labeling_functions = get_labeling_functions_from_path(
        heuristic_file, category_mapping_cache
    )
    if not labeling_functions:
        raise AssertionError(f"No labeling functions for in {heuristic_file}")
    # projection = get_projection(heuristics)
    artifact_df = load_dataset(dataset, projection={})
    applied_lf_matrix = apply_lfs_to_dataset(labeling_functions, artifacts=artifact_df)

    df = pd.DataFrame(applied_lf_matrix, columns=[lf.name for lf in labeling_functions])
    df.to_pickle(str(save_to_matrix))
