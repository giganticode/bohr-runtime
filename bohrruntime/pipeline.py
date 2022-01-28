import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml
from bohrapi.artifacts import Commit
from bohrapi.core import Dataset, Experiment, Task, Workspace
from git import Repo

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.heuristics import get_heuristic_files, is_heuristic_file
from bohrruntime.util.paths import normalize_paths, relative_to_safe

logger = logging.getLogger(__name__)


# todo move this to bohr api
def iterate_workspace(
    workspace: Workspace, path_config: PathConfig, iterate_heuristics: bool = True
) -> Union[List[Tuple[Experiment, Dataset]], List[Tuple[Experiment, Dataset, str]]]:
    for experiment in sorted(workspace.experiments, key=lambda x: x.name):
        heuristic_groups = get_heuristic_files(
            path_config.heuristics, experiment.task.top_artifact
        )
        for dataset in experiment.datasets:
            if iterate_heuristics:
                for heuristic_group in heuristic_groups:
                    yield (experiment, dataset, heuristic_group)
            else:
                yield (experiment, dataset, None)


@dataclass
class DvcCommand(ABC):
    path_config: PathConfig

    def to_dvc_config_dict(
        self,
    ) -> Dict:
        cmd = self.get_cmd()
        params = self.get_params()
        deps = self.get_deps()
        outs = self.get_outs()
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
                **self.generate_for_each_entry(entry, self.path_config),
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
            {
                (e, d)
                for e, d, h in iterate_workspace(
                    self.workspace, self.path_config, False
                )
            },
            key=lambda d: (d[0].name, d[1].id),
        )

    def generate_for_each_entry(
        self, entry, path_config: PathConfig
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
            str(self.path_config.cached_dataset_dir / "${item}.jsonl"),
            {
                str(
                    self.path_config.cached_dataset_dir / "${item}.jsonl.metadata.json"
                ): {"cache": False}
            },
        ]
        return outs


class ApplyHeuristicsCommand(ForEachDvcCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'

    def get_deps(self) -> List[str]:
        deps = [
            str(
                self.path_config.cloned_bohr_dir
                / self.path_config.heuristics_dir
                / "${item.heuristic_group}"
            ),
            str(self.path_config.cached_dataset_dir / "${item.dataset}.jsonl"),
        ]
        return deps

    def get_outs(self) -> List[Any]:
        outs = [
            str(
                self.path_config.runs_dir
                / "__heuristics"
                / "${item.dataset}"
                / "${item.heuristic_group}"
                / "heuristic_matrix.pkl"
            )
        ]
        return outs

    def get_iterating_over(self) -> Sequence:
        return sorted(
            {
                (d, h)
                for _, d, h in iterate_workspace(self.workspace, self.path_config, True)
            },
            key=lambda d: (d[0].id, d[1]),
        )

    def generate_for_each_entry(
        self, entry, path_config: PathConfig
    ) -> Dict[str, Dict[str, str]]:
        dataset, heuristic_group = entry
        relative_heuristic_group = str(
            relative_to_safe(heuristic_group, self.path_config.heuristics)
        )
        return {
            f"{dataset.id}__{relative_heuristic_group}": {
                "dataset": dataset.id,
                "heuristic_group": str(
                    relative_to_safe(heuristic_group, self.path_config.heuristics)
                ),
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
            deps.append(
                str(self.path_config.cached_dataset_dir / f"{dataset.id}.jsonl")
            )
        heuristic_groups = get_heuristic_files(
            self.path_config.heuristics, self.task.top_artifact
        )
        for heuristic_group in heuristic_groups:
            deps.append(
                str(
                    self.path_config.cloned_bohr_dir
                    / self.path_config.heuristics_dir
                    / relative_to_safe(heuristic_group, self.path_config.heuristics)
                )
            )
            for dataset in self.task.test_datasets:
                deps.append(
                    str(
                        self.path_config.runs_dir
                        / "__heuristics"
                        / dataset.id
                        / relative_to_safe(heuristic_group, self.path_config.heuristics)
                        / "heuristic_matrix.pkl"
                    ),
                )
        return deps

    def get_outs(self):
        outputs = []
        for dataset in self.task.test_datasets:
            heuristic_groups = get_heuristic_files(
                self.path_config.heuristics, self.task.top_artifact
            )
            for heuristic_group in heuristic_groups:
                outputs.append(
                    {
                        str(
                            self.path_config.runs_dir
                            / "__single_heuristic_metrics"
                            / self.task.name
                            / dataset.id
                            / relative_to_safe(
                                heuristic_group, self.path_config.heuristics
                            )
                            / "metrics.txt"
                        ): {"cache": False}
                    }
                )
        return outputs


class CombineHeuristicsCommand(ForEachDvcCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain combine-heuristics "${item.exp}" --dataset "${item.dataset}"'

    def get_deps(self) -> List[str]:
        return [str(self.path_config.runs_dir / "__heuristics" / "${item.dataset}")]

    def get_params(self) -> List:
        return [{"bohr.lock": ["experiments.${item.exp}.heuristics_classifier"]}]

    def get_outs(self) -> List:
        outs = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "heuristic_matrix.pkl"
            )
        ]
        return outs


@dataclass
class RunMetricsAndAnalysisCommand(ForEachDvcCommand):
    def get_metrics(self) -> List:
        metrics = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "metrics.txt"
            )
        ]
        return metrics

    def get_cmd(self) -> str:
        return 'bohr porcelain run-metrics-and-analysis "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "heuristic_matrix.pkl"
            ),
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "label_model.pkl"
            ),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            {
                str(
                    self.path_config.runs_dir
                    / "${item.task}"
                    / "${item.exp}"
                    / "${item.dataset}"
                    / "analysis.json"
                ): {"cache": False}
            },
            {
                str(
                    self.path_config.runs_dir
                    / "${item.task}"
                    / "${item.exp}"
                    / "${item.dataset}"
                    / "analysis.csv"
                ): {"cache": False}
            },
        ]
        return outs


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
            str(
                self.path_config.runs_dir  # TODO use exp_dataset_dir after adding relative option to it
                / self.exp.task.name
                / self.exp.name
                / dataset.id
                / "heuristic_matrix.pkl"
            )
            for dataset in self.exp.datasets
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            str(
                self.path_config.runs_dir
                / self.exp.task.name
                / self.exp.name
                / "label_model.pkl"
            ),
            str(
                self.path_config.runs_dir
                / self.exp.task.name
                / self.exp.name
                / "label_model_weights.csv"
            ),
        ]
        return outs


class LabelDatasetCommand(ForEachDvcCommand):
    def get_cmd(self) -> str:
        return 'bohr porcelain label-dataset "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "heuristic_matrix.pkl"
            ),
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "label_model.pkl"
            ),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "labeled.csv"
            )
        ]
        return outs


def dvc_config_from_tasks(workspace: Workspace, path_config: PathConfig) -> Dict:
    """
    >>> dvc_config_from_tasks([], PathConfig.load(Path('/')))
    Traceback (most recent call last):
    ...
    ValueError: At least of task should be specified
    >>> train = [Dataset("id.train", Commit, lambda x:x)]
    >>> test = [Dataset("id.test", Commit, lambda x:x)]
    >>> labels = [CommitLabel.BugFix, CommitLabel.NonBugFix]
    >>> tasks = [Task("name", "author", "desc", Commit, labels, train, test, ["hg1", "hg2"])]
    >>> dvc_config_from_tasks(experiments, PathConfig.load(Path('/')))
    {'stages': {'apply_heuristic': {'foreach': {'name': [{'dataset': 'id.train'}, {'dataset': 'id.test'}]}, \
'do': {'cmd': 'bohr porcelain label-dataset  "${key}" "${item.dataset}"', 'params': [], \
'deps': \
['/Users/hlib/dev/bohr-workdir/${key}/${item.dataset}/heuristic_matrix.pkl', \
'/Users/hlib/dev/bohr-workdir/${key}/label_model.pkl'], \
'outs': ['/Users/hlib/dev/bohr-workdir/${key}/${item.dataset}/labeled.csv'], 'metrics': []}}}}
    """
    if len(workspace.experiments) == 0:
        raise ValueError("At least of task should be specified")

    all_tasks = sorted({exp.task for exp in workspace.experiments})

    train_model_commands = [
        TrainLabelModelCommand(path_config, exp)
        for exp in sorted(workspace.experiments, key=lambda x: x.name)
    ]
    single_heuristic_commands = [
        ComputeSingleHeuristicMetricsCommand(path_config, task) for task in all_tasks
    ]
    commands: List[ForEachDvcCommand] = (
        [
            LoadDatasetsCommand(path_config, workspace),
            ApplyHeuristicsCommand(path_config, workspace),
            CombineHeuristicsCommand(path_config, workspace),
            RunMetricsAndAnalysisCommand(path_config, workspace),
            LabelDatasetCommand(path_config, workspace),
        ]
        + train_model_commands
        + single_heuristic_commands
    )

    final_dict = {"stages": {}}
    for command in commands:
        name, dvc_dct = next(iter(command.to_dvc_config_dict().items()))
        final_dict["stages"][name] = dvc_dct
    return final_dict


def fetch_heuristics_if_needed(heuristics_revision: str, heuristics_root: Path) -> None:
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
    path_config: PathConfig,
) -> None:
    fetch_heuristics_if_needed(
        workspace.experiments[0].revision, path_config.cloned_bohr
    )
    dvc_config = dvc_config_from_tasks(workspace, path_config)
    params = {"bohr_runtime_version": workspace.bohr_runtime_version, "experiments": {}}
    for exp in workspace.experiments:
        heuristic_groups = exp.heuristic_groups
        params["experiments"][exp.name] = {
            "train_set": exp.train_dataset.id,
            "heuristics_classifier": ":".join(
                normalize_paths(
                    heuristic_groups, path_config.heuristics, is_heuristic_file
                )
            )
            if heuristic_groups is not None
            else ".",
        }
    params_yaml = yaml.dump(params)
    with (path_config.project_root / "bohr.lock").open("w") as f:
        f.write(params_yaml)
    dvc_config_yaml = yaml.dump(dvc_config)
    with (path_config.project_root / "dvc.yaml").open("w") as f:
        f.write(dvc_config_yaml)
