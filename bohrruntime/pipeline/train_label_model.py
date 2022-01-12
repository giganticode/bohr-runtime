import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from bohrapi.core import Dataset, Experiment
from bohrlabels.core import LabelSet
from snorkel.labeling.model import LabelModel

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.labeling.cache import CategoryMappingCache
from bohrruntime.pipeline.apply_heuristics import load_ground_truth_labels
from bohrruntime.util.paths import AbsolutePath

random.seed(13)


class GroundTruthColumnNotFound(Exception):
    pass


def calculate_metrics(
    label_model: LabelModel,
    test_set_id: str,
    true_labels: np.ndarray,
    save_to: AbsolutePath,
) -> Dict[str, float]:
    """
    >>> from collections import namedtuple; import tempfile
    >>> def mocked_predictions(l,return_probs,tie_break_policy): return np.array([1, 0, 1]), np.array([[0.1, 0.9], [0.8, 0.2], [0.25, 0.75]])
    >>> def mocked_scores(L,Y,tie_break_policy,metrics):
    ...     return {"f1": 1.0} if metrics == ['f1'] else {"roc_auc": 0.78}
    >>> lm = namedtuple('LM', ['predict', 'score'])(mocked_predictions, mocked_scores)
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
    ...     calculate_metrics(lm, "test_set", np.array([1, 1, 0]), Path(tmpdirname))
    {'label_model_accuracy_test_set': 0.333, 'label_model_auc_test_set': 0.78, 'label_model_f1_test_set': 1.0, 'label_model_mse_test_set': 0.404}
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
    ...     calculate_metrics(lm, "test_set", np.array([0, 1, 0]), Path(tmpdirname))
    {'label_model_accuracy_test_set': 0.0, 'label_model_auc_test_set': 0.78, 'label_model_f1_test_set': 1.0, 'label_model_mse_test_set': 0.671}
    """
    lines = np.load(str(save_to / f"heuristic_matrix.pkl"), allow_pickle=True)

    tie_break_policy = "random"

    Y_pred, Y_prob = label_model.predict(
        lines, return_probs=True, tie_break_policy=tie_break_policy
    )

    try:
        auc = label_model.score(
            L=lines, Y=true_labels, tie_break_policy="random", metrics=["roc_auc"]
        )["roc_auc"]
        auc = round(auc, 3)
    except ValueError:
        auc = "n/a"
    f1 = label_model.score(
        L=lines, Y=true_labels, tie_break_policy="random", metrics=["f1"]
    )["f1"]
    accuracy = sum(Y_pred == true_labels) / float(len(Y_pred))
    mse = np.mean((Y_prob[:, 1] - true_labels) ** 2)

    return {
        f"label_model_accuracy_{test_set_id}": round(accuracy, 3),
        f"label_model_auc_{test_set_id}": auc,
        f"label_model_f1_{test_set_id}": round(f1, 3),
        f"label_model_mse_{test_set_id}": round(mse, 3),
    }


def fit_label_model(lines_train: np.ndarray) -> LabelModel:
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lines_train, n_epochs=100, log_freq=10, seed=123)
    return label_model


def train_label_model(
    exp: Experiment, training_dataset: Dataset, path_config: Optional[PathConfig] = None
) -> Dict[str, Any]:
    path_config = path_config or PathConfig.load()
    dataset_dir = path_config.exp_dataset_dir(exp, training_dataset)
    task_dir = path_config.exp_dir(exp)

    lines_train = pd.read_pickle(str(dataset_dir / f"heuristic_matrix.pkl"))
    label_model = fit_label_model(lines_train.to_numpy())
    label_model.save(str(task_dir / "label_model.pkl"))
    label_model.eval()

    label_model_weights_file = task_dir / f"label_model_weights.csv"

    df = pd.DataFrame(
        label_model.mu.cpu().detach().numpy().reshape(-1, 4),
        columns=["00", "01", "10", "11"],
        index=lines_train.columns,
    )
    df.to_csv(label_model_weights_file, index_label="heuristic_name")

    stats = {}
    for test_set in exp.task.test_datasets:
        label_series = load_ground_truth_labels(exp.task, test_set)
        if label_series is not None:
            category_mapping_cache = CategoryMappingCache(
                list(map(lambda x: str(x), exp.task.labels)), maxsize=10000
            )
            label_series = np.array(
                list(
                    map(lambda x: category_mapping_cache[LabelSet.of(x)], label_series)
                )
            )
            stats.update(
                calculate_metrics(
                    label_model,
                    test_set.id,
                    label_series,
                    save_to=path_config.exp_dataset_dir(exp, test_set),
                )
            )

    return stats
