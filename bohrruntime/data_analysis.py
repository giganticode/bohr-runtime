import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpyencoder import NumpyEncoder
from snorkel.labeling import LabelingFunction, LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter

from bohrruntime.util.paths import AbsolutePath


def save_analysis(
    lf_analysis_summary,
    save_csv_to: AbsolutePath,
    save_json_to: AbsolutePath,
) -> None:
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


def calculate_lf_metrics(
    label_matrix: np.ndarray,
    label_series: np.ndarray = None,
    save_to: Optional[Path] = None,
) -> Dict[str, Any]:
    metrics = {}
    coverage = sum((label_matrix != -1).any(axis=1)) / len(label_matrix)
    if label_series is not None:
        majority_accuracy = majority_acc(label_matrix, label_series)
        metrics["majority_accuracy_not_abstained"] = majority_accuracy
    metrics["n_labeling_functions"] = label_matrix.shape[1]
    metrics[f"coverage"] = coverage
    if save_to:
        with open(save_to, "w") as f:
            json.dump(metrics, f)
    return metrics
