import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from snorkel.labeling.model import LabelModel

from bohr.config import Config, load_config

random.seed(13)


def calculate_metrics(
    label_model: LabelModel,
    dataset_name: str,
    true_labels: np.ndarray,
    save_to: Path,
) -> Dict[str, float]:
    """
    >>> from collections import namedtuple; import tempfile
    >>> def mocked_predictions(l,return_probs,tie_break_policy): return np.array([1, 0, 1]), np.array([[0.1, 0.9], [0.8, 0.2], [0.25, 0.75]])
    >>> lm = namedtuple('LM', 'predict')(mocked_predictions)
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
    ...     calculate_metrics(lm, "test_set", np.array([1, 1, 0]), Path(tmpdirname))
    {'label_model_acc_test_set': 0.33, 'label_model_neg_log_loss_test_set': 1.491}
    """
    lines = np.load(
        str(save_to / f"heuristic_matrix_{dataset_name}.pkl"), allow_pickle=True
    )

    tie_break_policy = "random"

    Y_pred, Y_prob = label_model.predict(
        lines, return_probs=True, tie_break_policy=tie_break_policy
    )

    accuracy = sum(Y_pred == true_labels) / float(len(Y_pred))
    neg_log_loss = np.mean(
        -np.log2(np.take_along_axis(Y_prob, true_labels[:, None], axis=1))[:, 0]
    )

    return {
        f"label_model_acc_{dataset_name}": round(accuracy, 2),
        f"label_model_neg_log_loss_{dataset_name}": round(neg_log_loss, 3),
    }


def fit_label_model(lines_train: np.ndarray) -> LabelModel:
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lines_train, n_epochs=100, log_freq=100, seed=123)
    return label_model


def train_label_model(
    task_name: str, target_dataset: str, config: Config
) -> Dict[str, Any]:

    task_dir_generated = config.paths.generated / task_name
    if not task_dir_generated.exists():
        task_dir_generated.mkdir()

    lines_train = pd.read_pickle(
        str(task_dir_generated / f"heuristic_matrix_{target_dataset}.pkl")
    )
    label_model = fit_label_model(lines_train.to_numpy())
    label_model.save(str(task_dir_generated / "label_model.pkl"))
    label_model.eval()

    task = config.tasks[task_name]
    stats = {}
    for test_set_name, test_set in task.test_datasets.items():
        df = test_set.load(config.project_root)
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
    train_label_model("bugginess", load_config())
