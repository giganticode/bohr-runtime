import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpyencoder import NumpyEncoder
from pandas import Series
from snorkel.labeling import LabelingFunction, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter

from bohr.util.paths import AbsolutePath


def run_analysis(
    applied_lf_matrix: np.ndarray,
    lfs: List[LabelingFunction],
    save_csv_to: AbsolutePath,
    save_json_to: AbsolutePath,
    label_series: Optional[Series] = None,
) -> None:
    lf_analysis_summary = LFAnalysis(applied_lf_matrix, lfs).lf_summary(
        Y=label_series.values if label_series is not None else None
    )
    lf_analysis_summary.to_csv(save_csv_to)
    analysis_dict = lf_analysis_summary.to_dict()
    del analysis_dict["j"]
    with open(save_json_to, "w") as f:
        json.dump(analysis_dict, f, indent=4, sort_keys=True, cls=NumpyEncoder)


def majority_acc(line: np.ndarray, label_series: Series) -> float:
    majority_model = MajorityLabelVoter()
    maj_model_train_acc = majority_model.score(
        L=line, Y=label_series.values, tie_break_policy="random"
    )["accuracy"]
    return maj_model_train_acc


def calculate_metrics(
    applied_lf_matrix: np.ndarray,
    lfs: List[LabelingFunction],
    label_series: Optional[Series] = None,
    save_to: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = {}
    coverage = sum((applied_lf_matrix != -1).any(axis=1)) / len(applied_lf_matrix)
    if label_series is not None:
        majority_accuracy = majority_acc(applied_lf_matrix, label_series)
        metrics[f"majority_accuracy"] = majority_accuracy
    else:
        metrics["n_labeling_functions"]: len(lfs)
    metrics[f"coverage"] = coverage
    if save_to:
        with open(save_to, "w") as f:
            json.dump(metrics, f)
    return metrics
