import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Sequence, Type, TypeVar

import numpy as np
import pandas as pd
from fs.base import FS
from snorkel.analysis import Scorer


@dataclass
class GroundTruthLabels:
    labels: Sequence[int]


@dataclass
class HeuristicOutputs:
    label_matrix: pd.DataFrame


@dataclass
class CombinedHeuristicOutputs:
    labels: Sequence[int]
    probs: Sequence[float]


random.seed(13)

ModelSubclass = TypeVar("ModelSubclass", bound="Model")
ModelType = Type[ModelSubclass]

# model shoudl not contain logic about where it is stored


class Model(ABC):
    @abstractmethod
    def predict(self, heuristic_outputs: HeuristicOutputs) -> CombinedHeuristicOutputs:
        pass

    def calculate_model_metrics(
        self,
        heuristic_outputs: HeuristicOutputs,
        ground_truth_labels: GroundTruthLabels,
    ) -> Dict[str, float]:
        # """
        # >>> from collections import namedtuple
        # >>> import tempfile
        # >>> def mocked_predictions(): return np.array([1, 0, 1]), np.array([[0.1, 0.9], [0.8, 0.2], [0.25, 0.75]])
        # >>> lm = namedtuple('LM', ['predict'])(mocked_predictions)
        # >>> with tempfile.TemporaryDirectory() as tmpdirname:
        # ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
        # ...     lm.calculate_metrics(HeuristicOutputs(np.array([1, 1, 0])))
        # {'label_model_accuracy': 0.333, 'label_model_auc': 0.5, 'label_model_f1': 0.5, 'label_model_f1_macro': 0.25, 'label_model_mse': 0.404}
        # >>> with tempfile.TemporaryDirectory() as tmpdirname:
        # ...     np.ndarray([]).dump(f"{tmpdirname}/heuristic_matrix_test_set.pkl")
        # ...     lm.calculate_metrics(HeuristicOutputs(np.array([0, 1, 0])))
        # {'label_model_accuracy': 0.0, 'label_model_auc': 0.0, 'label_model_f1': 0.0, 'label_model_f1_macro': 0.0, 'label_model_mse': 0.671}
        # """
        combined_heuristic_output = self.predict(heuristic_outputs)
        Y_pred, Y_prob = (
            combined_heuristic_output.labels,
            combined_heuristic_output.probs,
        )
        true_labels = ground_truth_labels.labels

        try:
            auc = Scorer(metrics=["roc_auc"]).score(true_labels, Y_pred, Y_prob)[
                "roc_auc"
            ]
            auc = round(auc, 3)
        except ValueError:
            auc = "n/a"
        f1 = Scorer(metrics=["f1"]).score(true_labels, Y_pred, Y_prob)["f1"]
        f1_macro = Scorer(metrics=["f1_macro"]).score(true_labels, Y_pred, Y_prob)[
            "f1_macro"
        ]
        accuracy = sum(Y_pred == true_labels) / float(len(Y_pred))
        mse = np.mean((Y_prob[:, 1] - true_labels) ** 2)

        return {
            f"label_model_accuracy": round(accuracy, 3),
            f"label_model_auc": auc,
            f"label_model_f1": round(f1, 3),
            f"label_model_f1_macro": round(f1_macro, 3),
            f"label_model_mse": round(mse, 3),
        }


class SynteticModel(Model):
    pass


class RealModel(Model):
    pass


class ModelTrainer(ABC):
    def __init__(self, fs: FS):
        self.fs = fs

    @abstractmethod
    def train(self, heuristic_outputs: HeuristicOutputs) -> "Model":
        pass

    @abstractmethod
    def get_random_model(self) -> SynteticModel:
        pass

    @abstractmethod
    def get_zero_model(self) -> SynteticModel:
        pass

    @abstractmethod
    def load_model(self, task_path: str) -> ModelSubclass:
        pass

    @abstractmethod
    def save_model(self, model: Model, path: str) -> None:
        pass
