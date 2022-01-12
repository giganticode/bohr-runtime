import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpyencoder import NumpyEncoder
from snorkel.labeling import LabelingFunction, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter

from bohrruntime.util.paths import AbsolutePath


def run_analysis(
    applied_lf_matrix: np.ndarray,
    lfs: List[LabelingFunction],
    save_csv_to: AbsolutePath,
    save_json_to: AbsolutePath,
    label_series: Optional[np.ndarray] = None,
) -> None:
    lf_analysis_summary = LFAnalysis(applied_lf_matrix, lfs).lf_summary(label_series)
    lf_analysis_summary.to_csv(save_csv_to)
    analysis_dict = lf_analysis_summary.to_dict()
    del analysis_dict["j"]
    with open(save_json_to, "w") as f:
        json.dump(analysis_dict, f, indent=4, sort_keys=True, cls=NumpyEncoder)


def majority_acc(line: np.ndarray, label_series: np.ndarray) -> float:
    majority_model = MajorityLabelVoter()
    maj_model_train_acc = majority_model.score(
        L=line, Y=label_series, tie_break_policy="abstain"
    )["accuracy"]
    return maj_model_train_acc


def calculate_metrics(
    applied_lf_matrix: np.ndarray,
    lfs: List[LabelingFunction],
    label_series: np.ndarray = None,
    save_to: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = {}
    coverage = sum((applied_lf_matrix != -1).any(axis=1)) / len(applied_lf_matrix)
    if label_series is not None:
        majority_accuracy = majority_acc(applied_lf_matrix, label_series)
        metrics["majority_accuracy_not_abstained"] = majority_accuracy
    else:
        metrics["n_labeling_functions"]: len(lfs)
    metrics[f"coverage"] = coverage
    if save_to:
        with open(save_to, "w") as f:
            json.dump(metrics, f)
    return metrics


def get_fired_datapoints(
    matrix: pd.DataFrame, heuristics: List[str], artifact_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    >>> matrix = pd.DataFrame([[0, -1, 0],[1, -1, -1],[1, -1, 1]], columns=['h1', 'h2', 'h3'])
    >>> artifact_df = pd.DataFrame([['commit1'],['commit2'], ['commit3']], columns=['message'])
    >>> ones, zeros, stats = get_fired_datapoints(matrix, ["h2", "h3"], artifact_df) # doctest: +NORMALIZE_WHITESPACE
    >>> ones
       message
    2  commit3
    >>> zeros
       message
    0  commit1
    >>> stats
    {'total': 3, 'ones': 1, 'zeros': 1}
    """
    total = len(matrix)
    for heuristic in heuristics:
        if heuristic not in matrix:
            raise ValueError(
                f"Unknown heursitic: {heuristic}. Some possible values: {matrix.columns.tolist()}. Have you just added a heuristic? Reproduce the pipeline to debug it!"
            )
    columns = matrix[heuristics]
    ones = columns[(columns == 1).any(axis=1)]
    zeros = columns[(columns == 0).any(axis=1)]
    ones_count = len(ones)
    zeros_count = len(zeros)

    ones_df = artifact_df.loc[ones.index]
    zeros_df = artifact_df.loc[zeros.index]

    return (
        ones_df,
        zeros_df,
        {"total": total, "ones": ones_count, "zeros": zeros_count},
    )
