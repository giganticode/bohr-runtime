from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Type, Union

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Experiment, Task
from bohrlabels.core import LabelSet
from snorkel.analysis import Scorer
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import (
    BohrLabelModel,
    Model,
    RandomModel,
    load_dataset,
    load_ground_truth_labels,
)
from bohrruntime.data_analysis import calculate_lf_metrics, save_analysis
from bohrruntime.labeling.cache import CategoryMappingCache


def calculate_model_metrics(
        model: Model,
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
    ...     calculate_model_metrics(lm, "test_set", np.array([1, 1, 0]), Path(tmpdirname))
    {'label_model_accuracy_test_set': 0.333, 'label_model_auc_test_set': 0.78, 'label_model_f1_test_set': 1.0, 'label_model_mse_test_set': 0.404}
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
    ...     calculate_model_metrics(lm, "test_set", np.array([0, 1, 0]), Path(tmpdirname))
    {'label_model_accuracy_test_set': 0.0, 'label_model_auc_test_set': 0.78, 'label_model_f1_test_set': 1.0, 'label_model_mse_test_set': 0.671}
    """
    # label_matrix = np.load(str(save_to / f"heuristic_matrix.pkl"), allow_pickle=True)

    Y_pred, Y_prob = model.predict()

    try:
        auc = Scorer(metrics=["roc_auc"]).score(true_labels, Y_pred, Y_prob)["roc_auc"]
        auc = round(auc, 3)
    except ValueError:
        auc = "n/a"
    f1 = Scorer(metrics=["f1"]).score(true_labels, Y_pred, Y_prob)["f1"]
    f1_macro = Scorer(metrics=["f1_macro"]).score(true_labels, Y_pred, Y_prob)["f1_macro"]
    accuracy = sum(Y_pred == true_labels) / float(len(Y_pred))
    mse = np.mean((Y_prob[:, 1] - true_labels) ** 2)

    return {
        f"label_model_accuracy": round(accuracy, 3),
        f"label_model_auc": auc,
        f"label_model_f1": round(f1, 3),
        f"label_model_f1_macro": round(f1_macro, 3),
        f"label_model_mse": round(mse, 3),
    }


@dataclass()
class SynteticExperiment:
    name: str
    model_type: Type
    task: Task


def calculate_experiment_metrics(exp: Union[Experiment, SynteticExperiment], dataset: Dataset, fs: BohrFileSystem):

    category_mapping_cache = CategoryMappingCache(
        list(map(lambda x: str(x), exp.task.labels)), maxsize=10000
    )
    artifact_df = load_dataset(dataset, projection={})
    label_series = load_ground_truth_labels(exp.task, dataset, pre_loaded_artifacts=artifact_df)
    if label_series is not None:
        label_series = np.array(
            list(
                map(lambda x: category_mapping_cache[LabelSet.of(x)], label_series)
            )
        )

    if type(exp).__name__ == 'Experiment':
        all_heuristics_file = fs.experiment_label_matrix_file(exp, dataset).to_absolute_path()
        heuristic_matrix = pd.read_pickle(all_heuristics_file)
        label_matrix = heuristic_matrix.to_numpy()
        lf_metrics = calculate_lf_metrics(
            label_matrix,
            label_series
        )

    if label_series is not None:
        if type(exp).__name__ == 'Experiment':
            label_model = LabelModel()
            label_model.load(str(fs.label_model(exp).to_absolute_path()))
            model = BohrLabelModel(label_model, label_matrix, tie_break_policy = "random")
        elif type(exp).__name__ == 'SynteticExperiment':
            model = exp.model_type(len(artifact_df))
        else:
            raise AssertionError()
        metrics = calculate_model_metrics(
            model,
            label_series
        )
    else:
        metrics = {}
    if type(exp).__name__ == 'Experiment':
        metrics = {**lf_metrics, **metrics}

    save_metrics_to = fs.experiment_metrics(exp, dataset).to_absolute_path()
    with open(save_metrics_to, 'w') as f:
        for metric_key, metric_value in metrics.items():
            f.write(f'{metric_key} = {metric_value}\n')

    if type(exp).__name__ == 'Experiment':
        labeling_functions = [SimpleNamespace(name=heuristic) for heuristic in heuristic_matrix.columns]
        lf_analysis_summary = LFAnalysis(label_matrix, labeling_functions).lf_summary(label_series)
        save_analysis(lf_analysis_summary,
                      fs.analysis_csv(exp, dataset).to_absolute_path(),
                      fs.analysis_json(exp, dataset).to_absolute_path())