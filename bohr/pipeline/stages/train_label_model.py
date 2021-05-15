import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from snorkel.labeling.model import LabelModel

from bohr.config import load_config
from bohr.datamodel import AbsolutePath, Dataset, Task
from bohr.pathconfig import PathConfig

random.seed(13)


class GroundTruthColumnNotFound(Exception):
    pass


def calculate_metrics(
    label_model: LabelModel,
    dataset_name: str,
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
    lines = np.load(
        str(save_to / f"heuristic_matrix_{dataset_name}.pkl"), allow_pickle=True
    )

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
        f"label_model_accuracy_{dataset_name}": round(accuracy, 3),
        f"label_model_auc_{dataset_name}": auc,
        f"label_model_f1_{dataset_name}": round(f1, 3),
        f"label_model_mse_{dataset_name}": round(mse, 3),
    }


def fit_label_model(lines_train: np.ndarray) -> LabelModel:
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lines_train, n_epochs=100, log_freq=100, seed=123)
    return label_model


def train_label_model(
    task: Task, target_dataset: Dataset, path_config: PathConfig
) -> Dict[str, Any]:

    task_dir_generated = path_config.generated / task.name
    if not task_dir_generated.exists():
        task_dir_generated.mkdir()

    lines_train = pd.read_pickle(
        str(task_dir_generated / f"heuristic_matrix_{target_dataset.name}.pkl")
    )
    label_model = fit_label_model(lines_train.to_numpy())
    label_model.save(str(task_dir_generated / "label_model.pkl"))
    label_model.eval()

    label_model_weights_file = (
        path_config.generated / task.name / f"label_model_weights.csv"
    )

    df = pd.DataFrame(
        label_model.mu.cpu().detach().numpy().reshape(-1, 4),
        columns=["00", "01", "10", "11"],
        index=lines_train.columns,
    )
    df.to_csv(label_model_weights_file, index_label="heuristic_name")

    stats = {}
    for test_set_name, test_set in task._test_datasets.items():
        df = test_set.load()
        if task.label_column_name not in df.columns:
            raise GroundTruthColumnNotFound(
                f"Dataset {test_set_name} is added as a test set to the {task.name} task.\n"
                f"However, column with ground-thruth labels '{task.label_column_name}' not found."
            )
        stats.update(
            calculate_metrics(
                label_model,
                test_set_name,
                df[task.label_column_name].values,
                save_to=task_dir_generated,
            )
        )

    return stats


if __name__ == "__main__":
    config = load_config()
    task = config.tasks["bugginess"]
    dataset = config.datasets["berger"]
    train_label_model(task, dataset, config.paths)
