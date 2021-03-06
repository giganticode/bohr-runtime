import numpy as np
import pandas as pd
from pandas import Series
from snorkel.labeling.model import LabelModel


def train_lmodel(lines_train: np.ndarray) -> LabelModel:
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(lines_train, n_epochs=100, log_freq=100, seed=123)
    return label_model


def label(
    label_model: LabelModel, matrix: np.ndarray, df: pd.DataFrame
) -> pd.DataFrame:
    labels, probs = label_model.predict(L=matrix, return_probs=True)
    df_labeled = df.assign(bug_predicted=Series(labels))

    df_labeled["prob_bugless"] = Series(probs[:, 0])
    df_labeled["prob_bug"] = Series(probs[:, 1])
    return df_labeled
