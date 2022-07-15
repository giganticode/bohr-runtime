import pickle
from typing import List

import numpy as np
import pandas as pd
from fs.base import FS
from snorkel.labeling.model import LabelModel

from bohrruntime.datamodel.model import (
    CombinedHeuristicOutputs,
    HeuristicOutputs,
    Model,
    ModelSubclass,
    ModelTrainer,
    RealModel,
    SynteticModel,
)


class LabelModelTrainer(ModelTrainer):
    def __init__(self, fs: FS, class_balance: List[float]):
        super(LabelModelTrainer, self).__init__(fs)
        self.class_balance = class_balance

    def train(self, heuristic_outputs: HeuristicOutputs) -> Model:
        model = LabelModel(cardinality=2, verbose=True)
        model.fit(
            heuristic_outputs.label_matrix.to_numpy(),
            n_epochs=250,
            log_freq=10,
            seed=123,
            class_balance=self.class_balance,
        )
        model.eval()
        return BohrLabelModel(model, list(heuristic_outputs.label_matrix.columns))

    def get_random_model(self) -> SynteticModel:
        return RandomBohrLabelModel()

    def get_zero_model(self) -> SynteticModel:
        return ZeroBohrLabelModel()

    def load_model(self, task_path: str) -> ModelSubclass:
        label_model = LabelModel()
        with self.fs.open(f"{task_path}/label_model.pkl", "rb") as f:
            tmp_dict = pickle.load(f)
            label_model.__dict__.update(tmp_dict)
        with self.fs.open(f"{task_path}/label_model_weights.csv") as f:
            heuristic_names = pd.read_csv(f).index
        return BohrLabelModel(label_model, heuristic_names)

    def save_model(self, model: "BohrLabelModel", task_path: str) -> None:
        weights = model.get_weights()
        label_model = model.label_model
        with self.fs.open(f"{task_path}/label_model.pkl", "wb") as f:
            pickle.dump(label_model.__dict__, f)
        with self.fs.open(f"{task_path}/label_model_weights.csv", "w") as f:
            weights.to_csv(f, index_label="heuristic_name")


class BohrLabelModel(RealModel):
    def __init__(self, label_model: LabelModel, heuristic_names: List[str]):
        self.label_model = label_model
        self.heuristic_names = heuristic_names

    def predict(self, heuristic_outputs: HeuristicOutputs) -> CombinedHeuristicOutputs:
        Y_pred, Y_prob = self.label_model.predict(
            heuristic_outputs.label_matrix.to_numpy(),
            return_probs=True,
            tie_break_policy="random",  # TODO should it be random?
        )
        Y_prob = np.around(Y_prob, decimals=2)
        return CombinedHeuristicOutputs(Y_pred, Y_prob)

    def get_weights(self) -> pd.DataFrame:
        weights = self.label_model.mu.cpu().detach().numpy().reshape(-1, 4)
        df = pd.DataFrame(
            weights,
            columns=["00", "01", "10", "11"],
            index=self.heuristic_names,
        )
        return df

    def get_model(self):
        return self.label_model


class RandomBohrLabelModel(SynteticModel):
    def predict(self, heuristic_outputs: HeuristicOutputs) -> CombinedHeuristicOutputs:
        n_datapoints = heuristic_outputs.label_matrix.to_numpy().shape[0]
        Y_pred = [a for _ in range(n_datapoints // 2) for a in [0, 1]]
        if n_datapoints % 2 == 1:
            Y_pred.append(0)
        Y_prob = [[1.0, 0.0] for _ in range(n_datapoints)]
        return CombinedHeuristicOutputs(np.array(Y_pred), np.array(Y_prob))


class ZeroBohrLabelModel(SynteticModel):
    def predict(self, heuristic_outputs: HeuristicOutputs) -> CombinedHeuristicOutputs:
        n_datapoints = heuristic_outputs.label_matrix.to_numpy().shape[0]
        Y_pred = [0 for _ in range(n_datapoints)]
        Y_prob = [[1.0, 0.0] for _ in range(n_datapoints)]
        return CombinedHeuristicOutputs(np.array(Y_pred), np.array(Y_prob))
