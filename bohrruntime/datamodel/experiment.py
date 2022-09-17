import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from bohrapi.core import DataPointToLabelFunction
from frozendict import frozendict
from fs.base import FS
from numpyencoder import NumpyEncoder
from snorkel.labeling import LFAnalysis

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.model import (
    GroundTruthLabels,
    HeuristicOutputs,
    Model,
    ModelTrainer,
)
from bohrruntime.datamodel.task import Task

# Experiment is an attempt to solve the given task, and produces the model.
# Experiments can be "syntetic", they contain trivial model that does not need to be trained,
# e.g. random or zero model, only the task determines how such model would behave
from bohrruntime.util.paths import AbsolutePath

"""
Implementation of Experiment concept
"""

@dataclass(frozen=True)
class Experiment:
    name: str
    task: Task
    train_dataset: Optional[Dataset]
    heuristics_classifier: Optional[str] = None  # TODO Heuristic classifier object?
    extra_test_datasets: Dict[Dataset, DataPointToLabelFunction] = field(
        default_factory=frozendict
    )

    def __post_init__(self):
        dataset_name_set = set()
        for dataset in self.datasets:
            count_before = len(dataset_name_set)
            dataset_name_set.add(dataset.id)
            count_after = len(dataset_name_set)
            if count_after == count_before:
                raise ValueError(f"Dataset {dataset.id} is present more than once.")

    @property
    def heuristic_groups(self) -> Optional[List[str]]:
        return self._parse_heuristic_classifier()[0]

    @property
    def revision(self) -> Optional[str]:
        return self._parse_heuristic_classifier()[1]

    def _parse_heuristic_classifier(self) -> Tuple[Optional[List[str]], Optional[str]]:
        if self.heuristics_classifier is None:
            return None, None
        CLASSIFIER_REGEX = re.compile(
            "(?P<groups>(([^:@]+:)*[^:@]+)?)(@(?P<revision>[0-9a-fA-F]{7}|[0-9a-fA-F]{40}))?"
        )
        m = CLASSIFIER_REGEX.fullmatch(self.heuristics_classifier)
        if not m:
            raise ValueError(
                f"Invalid heuristic classifier syntax: {self.heuristics_classifier}. Correct syntax is: [[path1]:path2][@revision_sha]"
            )
        lst = m.group("groups")
        return None if lst == "" else lst.split(":"), m.group("revision")

    def get_dataset_by_id(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise ValueError(f"Unknown dataset: {dataset_id}")

    @property
    def datasets(self) -> List[Dataset]:
        return (
            self.task.get_test_datasets()
            + ([self.train_dataset] if self.train_dataset else [])
            + list(self.extra_test_datasets.keys())
        )

    def do_analysis(
        self,
        exp_dataset_subfs: FS,
        heuristic_outputs: HeuristicOutputs,
        label_series: GroundTruthLabels,
    ):
        labeling_functions = [
            SimpleNamespace(name=heuristic)
            for heuristic in heuristic_outputs.label_matrix.columns
        ]
        lf_analysis_summary = LFAnalysis(
            heuristic_outputs.label_matrix.to_numpy(), labeling_functions
        ).lf_summary(label_series.labels if label_series else None)
        save_analysis(lf_analysis_summary, exp_dataset_subfs)

    def load_model(self, fs: FS, task_path: str) -> Model:
        model_trainer = self.task.get_model_trainer(fs)
        return model_trainer.load_model(task_path)

    def save_model(self, model: Model, fs: FS, path: str):
        model_trainer = self.task.get_model_trainer(fs)
        model_trainer.save_model(model, path)


class SynteticExperiment(Experiment):
    syntetic_model_map = {"zero": "get_zero_model", "random": "get_random_model"}

    def __init__(self, name: str, task: Task, type: str):
        if type not in SynteticExperiment.syntetic_model_map:
            raise ValueError()

        super(SynteticExperiment, self).__init__(name, task, None)
        object.__setattr__(self, "type", type)

    def get_model_trainer(self, fs: FS) -> ModelTrainer:
        return self.task.get_model_trainer(fs, type)

    def load_model(self, fs: FS, task_path: str) -> Model:
        trainer: ModelTrainer = self.task.get_model_trainer(fs)
        model = getattr(trainer, SynteticExperiment.syntetic_model_map[self.type])()
        return model

    def do_analysis(
        self,
        exp_dataset_subfs: FS,
        heuristic_outputs: HeuristicOutputs,
        label_series: GroundTruthLabels,
    ):
        fake_analysis = pd.DataFrame()
        save_analysis(fake_analysis, exp_dataset_subfs)


def save_analysis(lf_analysis_summary: pd.DataFrame, exp_dataset_subfs: FS) -> None:
    with exp_dataset_subfs.open("analysis.csv", "w") as f:
        lf_analysis_summary.to_csv(f)
    analysis_dict = lf_analysis_summary.to_dict()
    if "j" in analysis_dict:
        del analysis_dict["j"]
    with exp_dataset_subfs.open("analysis.json", "w") as f:
        json.dump(analysis_dict, f, indent=4, sort_keys=True, cls=NumpyEncoder)
