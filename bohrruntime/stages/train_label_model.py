import random
from typing import Optional

import numpy as np
import pandas as pd
from bohrapi.core import Experiment
from snorkel.labeling.model import LabelModel

from bohrruntime.bohrfs import BohrFileSystem

random.seed(13)


class GroundTruthColumnNotFound(Exception):
    pass


def fit_label_model(lines_train: np.ndarray) -> LabelModel:
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lines_train, n_epochs=100, log_freq=10, seed=123)
    return label_model


def train_label_model(exp: Experiment, fs: BohrFileSystem) -> None:
    dataset_dir = fs.exp_dataset_dir(exp, exp.train_dataset).to_absolute_path()
    task_dir = fs.exp_dir(exp).to_absolute_path()

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
