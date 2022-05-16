import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional

import bohrapi.core as proxies
from frozendict import frozendict
from frozenlist import FrozenList

from bohrruntime import version
from bohrruntime.dataset import Dataset
from bohrruntime.fs import find_project_root
from bohrruntime.matching.dataset import MatchingDataset
from bohrruntime.task import Experiment, Task
from bohrruntime.util.paths import AbsolutePath


@dataclass(frozen=True)
class Workspace:
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


def convert_proxy_to_dataset(dataset_proxy: proxies.Dataset) -> Dataset:
    dataset_type = type(dataset_proxy).__name__
    m = {"Dataset": Dataset, "MatchingDataset": MatchingDataset}
    return m[dataset_type](
        dataset_proxy.id,
        dataset_proxy.top_artifact,
        dataset_proxy.query,
        dataset_proxy.projection,
        dataset_proxy.n_datapoints,
    )


def convert_proxy_to_task(
    task_proxy: proxies.Task, dataset_proxies: Dict[str, Dataset]
) -> Task:
    return Task(
        task_proxy.name,
        task_proxy.author,
        task_proxy.description,
        task_proxy.top_artifact,
        task_proxy.labels,
        frozendict(
            {dataset_proxies[k.id]: v for k, v in task_proxy.test_datasets.items()}
        ),
        task_proxy.hierarchy,
    )


def convert_proxies_to_real_objects(proxy_workspace: proxies.Workspace) -> Workspace:
    task_proxies: Dict[str, Task] = {}
    dataset_proxies: Dict[str, Dataset] = {}
    for proxy_experiment in proxy_workspace.experiments:
        proxy_train_dataset = proxy_experiment.train_dataset
        if proxy_train_dataset.id not in dataset_proxies:
            dataset_proxies[proxy_train_dataset.id] = convert_proxy_to_dataset(
                proxy_train_dataset
            )
        for proxy_extra_test_dataset in proxy_experiment.extra_test_datasets.keys():
            if proxy_extra_test_dataset.id not in dataset_proxies:
                dataset_proxies[proxy_extra_test_dataset.id] = convert_proxy_to_dataset(
                    proxy_extra_test_dataset
                )
        for proxy_test_dataset in proxy_experiment.task.test_datasets.keys():
            if proxy_test_dataset.id not in dataset_proxies:
                dataset_proxies[proxy_test_dataset.id] = convert_proxy_to_dataset(
                    proxy_test_dataset
                )
    experiments = []
    for proxy_experiment in proxy_workspace.experiments:
        proxy_task = proxy_experiment.task
        if proxy_task.name not in task_proxies:
            task_proxies[proxy_task.name] = convert_proxy_to_task(
                proxy_task, dataset_proxies
            )
        experiments.append(
            Experiment(
                proxy_experiment.name,
                task_proxies[proxy_experiment.task.name],
                dataset_proxies[proxy_experiment.train_dataset.id],
                proxy_experiment.class_balance,
                proxy_experiment.heuristics_classifier,
                frozendict(
                    {
                        dataset_proxies[k.id]: v
                        for k, v in proxy_experiment.extra_test_datasets
                    }
                ),
            )
        )

    return Workspace(proxy_workspace.bohr_runtime_version, experiments)


def load_workspace(project_root: Optional[AbsolutePath] = None) -> Workspace:
    project_root = project_root or find_project_root()
    file = project_root / "bohr.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("heuristic.module", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in inspect.getmembers(module):
        if isinstance(obj, proxies.Workspace):
            workspace = obj

            version_installed = version()
            if str(workspace.bohr_runtime_version) != version_installed:
                raise EnvironmentError(
                    f"Version of bohr framework from config: {workspace.bohr_runtime_version}. "
                    f"Version of bohr installed: {version_installed}"
                )
            try:
                return convert_proxies_to_real_objects(workspace)
            except Exception as e:
                raise ValueError("Error parsing bohr.py.") from e
    raise ValueError(
        f"Object of type {proxies.Workspace.__name__} not found in bohr.py"
    )
