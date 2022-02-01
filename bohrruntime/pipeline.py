import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml
from bohrapi.artifacts import Commit
from bohrapi.core import Dataset, Experiment, Task, Workspace
from git import Repo

from bohrruntime.bohrfs import (
    DATASET_TEMPLATE,
    EXPERIMENT_TEMPLATE,
    TASK_TEMPLATE,
    BohrFileSystem,
    BohrFsPath,
)
from bohrruntime.heuristics import get_heuristic_files, is_heuristic_file
from bohrruntime.util.paths import AbsolutePath, normalize_paths

logger = logging.getLogger(__name__)


# todo move this to bohr api
def iterate_workspace(
    workspace: Workspace, fs: BohrFileSystem, iterate_heuristics: bool = True
) -> Union[List[Tuple[Experiment, Dataset]], List[Tuple[Experiment, Dataset, str]]]:
    for experiment in sorted(workspace.experiments, key=lambda x: x.name):
        heuristic_groups = get_heuristic_files(
            fs.heuristics, experiment.task.top_artifact
        )
        for dataset in experiment.datasets:
            if iterate_heuristics:
                for heuristic_group in heuristic_groups:
                    yield (experiment, dataset, heuristic_group)
            else:
                yield (experiment, dataset, None)


def stringify_paths(
    paths: List[Union[BohrFsPath, Dict[BohrFsPath, Any]]]
) -> List[Union[str, Dict[str, Any]]]:
    res = []
    for path in paths:
        if isinstance(path, BohrFsPath):
            res.append(str(path))
        elif isinstance(path, Dict):
            if len(path) != 1:
                raise AssertionError()
            key, value = next(iter(path.items()))
            if not isinstance(key, BohrFsPath):
                raise AssertionError()
            res.append({str(key): value})
        else:
            raise AssertionError()
    return res


@dataclass
class DvcCommand(ABC):
    fs: BohrFileSystem

    def to_dvc_config_dict(
        self,
    ) -> Dict:
        cmd = self.get_cmd()
        params = self.get_params()
        deps = stringify_paths(self.get_deps())
        outs = stringify_paths(self.get_outs())
        metrics = [{m: {"cache": False}} for m in self.get_metrics()]
        dct = {
            self.stage_name(): {
                "cmd": cmd,
                "params": params + [{"bohr.lock": ["bohr_runtime_version"]}],
                "deps": deps,
                "outs": outs,
                "metrics": metrics,
            }
        }
        return dct

    def summary(self) -> str:
        return self.stage_name()

    def stage_name(self) -> str:
        return type(self).__name__[: -len("Command")]

    def n_stages(self) -> int:
        return 1

    def get_params(self) -> List:
        return []

    def get_deps(self) -> List:
        return []

    def get_outs(self) -> List:
        return []

    def get_metrics(self) -> List:
        return []

    @abstractmethod
    def get_cmd(self) -> str:
        pass


@dataclass
class ForEachDvcCommand(DvcCommand):
    workspace: Workspace

    def n_stages(self) -> int:
        return len(self.get_iterating_over())

    def get_for_each(self) -> Dict[str, Dict[str, Any]]:
        foreach: Dict[str, Dict[str, Any]] = {}
        for entry in self.get_iterating_over():
            foreach = {
                **foreach,
                **self.generate_for_each_entry(entry, self.fs),
            }
        return foreach

    def to_dvc_config_dict(
        self,
    ) -> Dict:
        parent = next(
            iter(super(ForEachDvcCommand, self).to_dvc_config_dict().values())
        )
        foreach = self.get_for_each()
        dct = {
            self.stage_name(): {
                "foreach": foreach,
                "do": parent,
            }
        }
        return dct

    def get_iterating_over(self) -> Sequence:
        return sorted(
            {(e, d) for e, d, h in iterate_workspace(self.workspace, self.fs, False)},
            key=lambda d: (d[0].name, d[1]),
        )

    def generate_for_each_entry(
        self, entry, fs: BohrFileSystem
    ) -> Dict[str, Dict[str, str]]:
        experiment, dataset = entry
        return {
            f"{experiment.name}__{dataset.id}": {
                "dataset": dataset.id,
                "exp": experiment.name,
                "task": experiment.task.name,
            }
        }


class LoadDatasetsCommand(ForEachDvcCommand):
    def get_iterating_over(self) -> Sequence:
        return sorted(
            {d.id for exp in self.workspace.experiments for d in exp.datasets}
        )

    def get_for_each(self):
        return self.get_iterating_over()

    def get_cmd(self) -> str:
        return 'bohr porcelain load-dataset "${item}"'

    def get_outs(self) -> List:
        outs = [
            self.fs.dataset("${item}"),
            {self.fs.dataset_metadata("${item}"): {"cache": False}},
        ]
        return outs


class ApplyHeuristicsCommand(ForEachDvcCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'

    def get_deps(self) -> List[str]:
        deps = [
            self.fs.heuristic_group("${item.heuristic_group}"),
            self.fs.dataset(DATASET_TEMPLATE.id),
        ]
        return deps

    def get_outs(self) -> List[Any]:
        outs = [
            self.fs.heuristic_matrix_file(DATASET_TEMPLATE, "${item.heuristic_group}")
        ]
        return outs

    def get_iterating_over(self) -> Sequence:
        return sorted(
            {(d, h) for _, d, h in iterate_workspace(self.workspace, self.fs, True)},
            key=lambda d: (d[0], d[1]),
        )

    def generate_for_each_entry(
        self, entry, fs: BohrFileSystem
    ) -> Dict[str, Dict[str, str]]:
        dataset, heuristic_group = entry
        relative_heuristic_group = str(heuristic_group)

        return {
            f"{dataset.id}__{relative_heuristic_group}": {
                "dataset": dataset.id,
                "heuristic_group": str(heuristic_group),
            }
        }


@dataclass()
class ComputeSingleHeuristicMetricsCommand(DvcCommand):
    task: Task

    def stage_name(self) -> str:
        return f"ComputeSingleHeuristicMetrics__{self.task.name}"

    def get_cmd(self) -> str:
        return f"bohr porcelain compute-single-heuristic-metric {self.task.name}"

    def get_deps(self) -> List:
        deps = []
        for dataset in self.task.test_datasets:
            deps.append(self.fs.dataset(dataset.id))
        for heuristic_group in get_heuristic_files(
            self.fs.heuristics, self.task.top_artifact
        ):
            deps.append(self.fs.heuristic_group(heuristic_group))
            for dataset in self.task.test_datasets:
                deps.append(self.fs.heuristic_matrix_file(dataset, heuristic_group))
        return deps

    def get_outs(self):
        outputs = []
        for dataset in self.task.test_datasets:
            heuristic_groups = get_heuristic_files(
                self.fs.heuristics, self.task.top_artifact
            )
            for heuristic_group in heuristic_groups:
                outputs.append(
                    {
                        self.fs.single_heuristic_metrics(
                            self.task, dataset, heuristic_group
                        ): {"cache": False}
                    }
                )
        return outputs


class CombineHeuristicsCommand(ForEachDvcCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain combine-heuristics "${item.exp}" --dataset "${item.dataset}"'

    def get_deps(self) -> List[BohrFsPath]:
        return [self.fs.heuristic_dataset_dir(DATASET_TEMPLATE)]

    def get_params(self) -> List:
        return [{"bohr.lock": ["experiments.${item.exp}.heuristics_classifier"]}]

    def get_outs(self) -> List[BohrFsPath]:
        outs = [
            self.fs.experiment_label_matrix_file(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE)
        ]
        return outs


@dataclass
class RunMetricsAndAnalysisCommand(ForEachDvcCommand):
    def get_metrics(self) -> List:
        metrics = [
            str(self.fs.experiment_metrics(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE))
        ]
        return metrics

    def get_cmd(self) -> str:
        return 'bohr porcelain run-metrics-and-analysis "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            self.fs.experiment_label_matrix_file(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE),
            self.fs.label_model(EXPERIMENT_TEMPLATE),
            self.fs.dataset(DATASET_TEMPLATE.id),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            {
                self.fs.analysis_json(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE): {
                    "cache": False
                }
            },
            {
                self.fs.analysis_csv(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE): {
                    "cache": False
                }
            },
        ]
        return outs


@dataclass
class ComputePredefinedModelMetricsCommand(ForEachDvcCommand):
    @abstractmethod
    def get_model_name(self) -> str:
        pass

    def get_metrics(self) -> List:
        metrics = [
            str(
                self.fs.experiment_metrics(
                    SimpleNamespace(name=self.get_model_name(), task=TASK_TEMPLATE),
                    DATASET_TEMPLATE,
                )
            )
        ]
        return metrics

    def get_deps(self) -> List:
        return [self.fs.dataset(DATASET_TEMPLATE.id)]

    def get_iterating_over(self) -> Sequence:
        all_tasks = {exp.task for exp in self.workspace.experiments}
        for task in sorted(all_tasks, key=lambda k: k.name):
            for dataset in sorted(task.test_datasets):
                yield (task, dataset)

    def generate_for_each_entry(
        self, entry, fs: BohrFileSystem
    ) -> Dict[str, Dict[str, str]]:
        task, dataset = entry
        return {
            f"{task.name}__{dataset.id}": {
                "dataset": dataset.id,
                "task": task.name,
            }
        }


class ComputeRandomModelMetricsCommand(ComputePredefinedModelMetricsCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain compute-random-model-metrics "${item.task}" "${item.dataset}"'

    def get_model_name(self) -> str:
        return "random_model"


class ComputeZeroModelMetricsCommand(ComputePredefinedModelMetricsCommand):
    def get_cmd(self) -> str:
        return (
            'bohr porcelain compute-zero-model-metrics "${item.task}" "${item.dataset}"'
        )

    def get_model_name(self) -> str:
        return "zero_model"


@dataclass
class TrainLabelModelCommand(DvcCommand):
    exp: Experiment

    def stage_name(self) -> str:
        return f"TrainLabelModel__{self.exp.name}"

    def get_cmd(self):
        return f"bohr porcelain train-label-model {self.exp.name}"

    def get_params(self):
        return [{"bohr.lock": [f"experiments.{self.exp.name}.train_set"]}]

    def get_deps(self):
        deps = [
            self.fs.experiment_label_matrix_file(self.exp, dataset)
            for dataset in self.exp.datasets
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            self.fs.label_model(self.exp),
            self.fs.label_model_weights(self.exp),
        ]
        return outs


class LabelDatasetCommand(ForEachDvcCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain label-dataset "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            self.fs.experiment_label_matrix_file(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE),
            self.fs.label_model(EXPERIMENT_TEMPLATE),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [self.fs.labeled_dataset(EXPERIMENT_TEMPLATE, DATASET_TEMPLATE)]
        return outs


def dvc_config_from_tasks(workspace: Workspace, fs: BohrFileSystem) -> Dict:
    """
    >>> dvc_config_from_tasks([], BohrFileSystem.init(Path('/')))
    Traceback (most recent call last):
    ...
    ValueError: At least of task should be specified
    >>> train = [Dataset("id.train", Commit, lambda x:x)]
    >>> test = [Dataset("id.test", Commit, lambda x:x)]
    >>> labels = [CommitLabel.BugFix, CommitLabel.NonBugFix]
    >>> tasks = [Task("name", "author", "desc", Commit, labels, train, test, ["hg1", "hg2"])]
    >>> dvc_config_from_tasks(experiments, BohrFileSystem.init(Path('/')))
    {'stages': {'apply_heuristic': {'foreach': {'name': [{'dataset': 'id.train'}, {'dataset': 'id.test'}]}, \
'do': {'cmd': 'bohr porcelain label-dataset  "${key}" "${item.dataset}"', 'params': [], \
'deps': \
['/Users/hlib/dev/bohr-workdir/${key}/${item.dataset}/heuristic_matrix.pkl', \
'/Users/hlib/dev/bohr-workdir/${key}/label_model.pkl'], \
'outs': ['/Users/hlib/dev/bohr-workdir/${key}/${item.dataset}/labeled.csv'], 'metrics': []}}}}
    """
    if len(workspace.experiments) == 0:
        raise ValueError("At least of task should be specified")

    all_tasks = sorted(
        {exp.task for exp in workspace.experiments}, key=lambda t: t.name
    )

    train_model_commands = [
        TrainLabelModelCommand(fs, exp)
        for exp in sorted(workspace.experiments, key=lambda x: x.name)
    ]
    single_heuristic_commands = [
        ComputeSingleHeuristicMetricsCommand(fs, task) for task in all_tasks
    ]
    commands: List[ForEachDvcCommand] = (
        [
            LoadDatasetsCommand(fs, workspace),
            ApplyHeuristicsCommand(fs, workspace),
            CombineHeuristicsCommand(fs, workspace),
            RunMetricsAndAnalysisCommand(fs, workspace),
            ComputeRandomModelMetricsCommand(fs, workspace),
            ComputeZeroModelMetricsCommand(fs, workspace),
            LabelDatasetCommand(fs, workspace),
        ]
        + train_model_commands
        + single_heuristic_commands
    )

    final_dict = {"stages": {}}
    for command in commands:
        name, dvc_dct = next(iter(command.to_dvc_config_dict().items()))
        final_dict["stages"][name] = dvc_dct
    return final_dict


def fetch_heuristics_if_needed(
    heuristics_revision: str, heuristics_root: AbsolutePath
) -> None:
    clone_repo = False
    if heuristics_root.exists():
        repo = Repo(heuristics_root)
        head_sha = repo.head.commit.hexsha
        if head_sha == heuristics_revision:
            if repo.is_dirty():
                print("Warning: downloaded heuristics have been modified!")
        else:
            if repo.is_dirty():
                raise RuntimeError(
                    f"Need to checkout revision {heuristics_revision}, "
                    f"however the current revision {head_sha} is dirty"
                )
            shutil.rmtree(heuristics_root)
            clone_repo = True

    else:
        clone_repo = True

    if clone_repo:
        repo = Repo.clone_from(
            "https://github.com/giganticode/bohr", heuristics_root, no_checkout=True
        )
        repo.git.checkout(heuristics_revision)


def write_tasks_to_dvc_file(
    workspace: Workspace,
    fs: BohrFileSystem,
) -> None:
    fetch_heuristics_if_needed(
        workspace.experiments[0].revision, fs.cloned_bohr.to_absolute_path()
    )
    dvc_config = dvc_config_from_tasks(workspace, fs)
    params = {"bohr_runtime_version": workspace.bohr_runtime_version, "experiments": {}}
    for exp in workspace.experiments:
        heuristic_groups = exp.heuristic_groups
        params["experiments"][exp.name] = {
            "train_set": exp.train_dataset.id,
            "heuristics_classifier": ":".join(
                normalize_paths(
                    heuristic_groups,
                    fs.heuristics.to_absolute_path(),
                    is_heuristic_file,
                )
            )
            if heuristic_groups is not None
            else ".",
        }
    params_yaml = yaml.dump(params)
    with (fs.root / "bohr.lock").open("w") as f:
        f.write(params_yaml)
    dvc_config_yaml = yaml.dump(dvc_config)
    with (fs.root / "dvc.yaml").open("w") as f:
        f.write(dvc_config_yaml)
