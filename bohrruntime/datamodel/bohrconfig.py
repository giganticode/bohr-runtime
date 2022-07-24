from dataclasses import dataclass
from typing import List

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment
from bohrruntime.datamodel.task import Task


@dataclass(frozen=True)
class BohrConfig:
    bohr_runtime_version: str
    experiments: List[Experiment]

    def get_experiment_by_name(self, exp_name: str) -> Experiment:
        for exp in self.experiments:
            if exp.name == exp_name:
                return exp
        raise ValueError(
            f"Unknown experiment: {exp_name}, possible values are {list(map(lambda e: e.name, self.experiments))}"
        )

    def get_task_by_name(self, task_name: str) -> Task:
        for exp in self.experiments:
            if exp.task.name == task_name:
                return exp.task
        raise ValueError(f"Unknown task: {task_name}")

    def get_dataset_by_id(self, id: str) -> Dataset:
        for exp in self.experiments:
            try:
                return exp.get_dataset_by_id(id)
            except ValueError:
                pass
        raise ValueError(f"Unknown dataset: {id}")
