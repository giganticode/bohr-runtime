from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Experiment
from bohrlabels.core import LabelSet
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import load_ground_truth_labels
from bohrruntime.data_analysis import calculate_lf_metrics, save_analysis
from bohrruntime.labeling.cache import CategoryMappingCache


def calculate_label_model_metrics(
        label_matrix: np.array,
        label_model: LabelModel,
        true_labels: np.ndarray
) -> Dict[str, float]:
    """
    >>> from collections import namedtuple; import tempfile
    >>> def mocked_predictions(l,return_probs,tie_break_policy): return np.array([1, 0, 1]), np.array([[0.1, 0.9], [0.8, 0.2], [0.25, 0.75]])
    >>> def mocked_scores(L,Y,tie_break_policy,metrics):
    ...     return {"f1": 1.0} if metrics == ['f1'] else {"roc_auc": 0.78}
    >>> lm = namedtuple('LM', ['predict', 'score'])(mocked_predictions, mocked_scores)
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
    ...     calculate_label_model_metrics(lm, "test_set", np.array([1, 1, 0]), Path(tmpdirname))
    {'label_model_accuracy_test_set': 0.333, 'label_model_auc_test_set': 0.78, 'label_model_f1_test_set': 1.0, 'label_model_mse_test_set': 0.404}
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
    ...     calculate_label_model_metrics(lm, "test_set", np.array([0, 1, 0]), Path(tmpdirname))
    {'label_model_accuracy_test_set': 0.0, 'label_model_auc_test_set': 0.78, 'label_model_f1_test_set': 1.0, 'label_model_mse_test_set': 0.671}
    """
    # label_matrix = np.load(str(save_to / f"heuristic_matrix.pkl"), allow_pickle=True)

    tie_break_policy = "random"

    Y_pred, Y_prob = label_model.predict(
        label_matrix, return_probs=True, tie_break_policy=tie_break_policy
    )

    try:
        auc = label_model.score(
            L=label_matrix, Y=true_labels, tie_break_policy="random", metrics=["roc_auc"]
        )["roc_auc"]
        auc = round(auc, 3)
    except ValueError:
        auc = "n/a"
    f1 = label_model.score(
        L=label_matrix, Y=true_labels, tie_break_policy="random", metrics=["f1"]
    )["f1"]
    accuracy = sum(Y_pred == true_labels) / float(len(Y_pred))
    mse = np.mean((Y_prob[:, 1] - true_labels) ** 2)

    return {
        f"label_model_accuracy": round(accuracy, 3),
        f"label_model_auc": auc,
        f"label_model_f1": round(f1, 3),
        f"label_model_mse": round(mse, 3),
    }


def calculate_experiment_metrics(exp: Experiment, dataset: Dataset, fs: Optional[BohrFileSystem]):
    fs = fs or BohrFileSystem.init()
    task_dir = fs.exp_dir(exp)
    dataset_dir = fs.exp_dataset_dir(exp, dataset).to_absolute_path()
    all_heuristics_file = dataset_dir / f"heuristic_matrix.pkl"
    save_metrics_to = dataset_dir / f"metrics.txt"

    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), exp.task.labels)), maxsize=10000
    )
    label_series = load_ground_truth_labels(exp.task, dataset)
    if label_series is not None:
        label_series = np.array(
            list(
                map(lambda x: category_mapping_cache[LabelSet.of(x)], label_series)
            )
        )

    heuristic_matrix = pd.read_pickle(all_heuristics_file)
    label_matrix = heuristic_matrix.to_numpy()
    metrics = calculate_lf_metrics(
        label_matrix,
        label_series
    )
    label_model = LabelModel()
    label_model.load(str(task_dir.to_absolute_path() / "label_model.pkl"))
    if label_series is not None:
        label_model_metrics = calculate_label_model_metrics(
            label_matrix,
            label_model,
            label_series
        )
        metrics = {**metrics, **label_model_metrics}
    with open(save_metrics_to, 'w') as f:
        for metric_key, metric_value in metrics.items():
            f.write(f'{metric_key} = {metric_value}\n')

    labeling_functions = [SimpleNamespace(name=heuristic) for heuristic in heuristic_matrix.columns]

    lf_analysis_summary = LFAnalysis(label_matrix, labeling_functions).lf_summary(label_series)
    save_analysis(lf_analysis_summary, dataset_dir / f"analysis.csv", dataset_dir / f"analysis.json")