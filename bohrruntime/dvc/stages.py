import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from bohrapi.artifacts import Commit
from bohrapi.core import Dataset, Experiment, Task, Workspace
from git import Repo

from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import is_heuristic_file, normalize_paths
from bohrruntime.heuristics import get_heuristic_files
from bohrruntime.util.paths import AbsolutePath, relative_to_safe

logger = logging.getLogger(__name__)


@dataclass
class DvcCommand(ABC):
    path_config: PathConfig
    experiments: List[Experiment]

    @abstractmethod
    def to_dvc_template_dict(self) -> Dict:
        pass

    def summary(self) -> str:
        return self.stage_name()

    def stage_name(self) -> str:
        return type(self).__name__[: -len("Command")]

    def _to_template_dict(
        self, foreach, cmd, params, deps, outs, metrics, kwargs=None
    ) -> Dict:
        metrics = [{m: {"cache": False}} for m in metrics]
        dct = {
            self.stage_name(): {
                "foreach": foreach,
                "do": {
                    "cmd": cmd,
                    "params": params + [{"bohr.lock": ["bohr_runtime_version"]}],
                    "deps": deps,
                    "outs": outs,
                    "metrics": metrics,
                },
            }
        }
        if kwargs is not None:
            dct[self.stage_name()]["do"] = {**dct[self.stage_name()]["do"], **kwargs}
        return dct

    @staticmethod
    def generate_for_each_entry(
        exp: Experiment,
        dataset: Dataset,
        heuristic_group: Optional[AbsolutePath],
        path_config: PathConfig,
    ) -> Tuple[str, Dict[str, str]]:
        exp_name = exp.name
        dataset_id = dataset.id
        dct = {"task": exp.task.name, "exp": exp_name, "dataset": dataset_id}
        key = f"{exp_name}__{dataset_id}"
        if heuristic_group is not None:
            heuristic_group = str(
                relative_to_safe(heuristic_group, path_config.heuristics)
            )
            dct["heuristic_group"] = heuristic_group
            key += f"__{heuristic_group}"
        return key, dct

    def generate_foreach(
        self, iterate_heuristics: bool = False
    ) -> Dict[str, Dict[str, str]]:
        foreach = {}
        for experiment in self.experiments:
            for dataset in experiment.task.datasets:
                heuristic_groups = get_heuristic_files(
                    self.path_config.heuristics, experiment.task.top_artifact
                )
                if iterate_heuristics:
                    for heuristic_group in heuristic_groups:
                        key, value = DvcCommand.generate_for_each_entry(
                            experiment, dataset, heuristic_group, self.path_config
                        )
                        foreach[key] = value
                else:
                    key, value = DvcCommand.generate_for_each_entry(
                        experiment, dataset, None, self.path_config
                    )
                    foreach[key] = value
        return foreach


class LoadDatasetsCommand(DvcCommand):
    def to_dvc_template_dict(self) -> Dict:
        cmd = 'bohr porcelain load-dataset "${item}"'
        outs = [
            str(self.path_config.cached_dataset_dir / "${item}.jsonl"),
            {
                str(
                    self.path_config.cached_dataset_dir / "${item}.jsonl.metadata.json"
                ): {"cache": False}
            },
        ]
        dataset_set = {d.id for run in self.experiments for d in run.task.datasets}
        foreach = sorted(dataset_set)

        return self._to_template_dict(
            foreach, cmd, params=[], deps=[], outs=outs, metrics=[]
        )


class ApplyHeuristicsCommand(DvcCommand):
    def to_dvc_template_dict(self) -> Dict:
        cmd = 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'
        deps = [
            str(
                self.path_config.cloned_bohr_dir
                / self.path_config.heuristics_dir
                / "${item.heuristic_group}"
            ),
            str(self.path_config.cached_dataset_dir / "${item.dataset}.jsonl"),
        ]
        outs = [
            str(
                self.path_config.runs_dir
                / "__heuristics"
                / "${item.dataset}"
                / "${item.heuristic_group}"
                / "heuristic_matrix.pkl"
            )
        ]
        foreach = self.generate_foreach(iterate_heuristics=True)

        return self._to_template_dict(foreach, cmd, [], deps, outs, [])

    def generate_foreach(
        self, iterate_heuristics: bool = False
    ) -> Dict[str, Dict[str, str]]:
        all_datasets = {d for exp in self.experiments for d in exp.task.datasets}
        foreach = {}

        for dataset in all_datasets:
            heuristic_groups = get_heuristic_files(
                self.path_config.heuristics, dataset.top_artifact
            )
            for heuristic_group in heuristic_groups:
                relative_heuristic_group = str(
                    relative_to_safe(heuristic_group, self.path_config.heuristics)
                )
                foreach[f"{dataset.id}__{relative_heuristic_group}"] = {
                    "dataset": dataset.id,
                    "heuristic_group": str(
                        relative_to_safe(heuristic_group, self.path_config.heuristics)
                    ),
                }
        return foreach


class ComputeSingleHeuristicMetrics(DvcCommand):
    def to_dvc_template_dict(self) -> Dict:
        cmd = 'bohr porcelain compute-single-heuristic-metric --task "${item.task}" --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'
        deps = [
            str(
                self.path_config.cloned_bohr_dir
                / self.path_config.heuristics_dir
                / "${item.heuristic_group}"
            ),
            str(self.path_config.cached_dataset_dir / "${item.dataset}.jsonl"),
            str(
                self.path_config.runs_dir
                / "__heuristics"
                / "${item.dataset}"
                / "${item.heuristic_group}"
                / "heuristic_matrix.pkl"
            ),
        ]
        outputs = [
            {
                str(
                    self.path_config.runs_dir
                    / "__single_heuristic_metrics"
                    / "${item.task}"
                    / "${item.dataset}"
                    / "${item.heuristic_group}"
                    / "heuristic_metrics.json"
                ): {"cache": False}
            }
        ]
        foreach = self.generate_foreach(iterate_heuristics=True)
        kwargs = {}
        # kwargs= {'frozen': True}
        return self._to_template_dict(foreach, cmd, [], deps, outputs, [], kwargs)

    def generate_foreach(
        self, iterate_heuristics: bool = False
    ) -> Dict[str, Dict[str, str]]:
        all_tasks = {exp.task for exp in self.experiments}
        foreach = {}

        for task in all_tasks:
            for dataset in task.datasets:
                heuristic_groups = get_heuristic_files(
                    self.path_config.heuristics, dataset.top_artifact
                )
                for heuristic_group in heuristic_groups:
                    relative_heuristic_group = str(
                        relative_to_safe(heuristic_group, self.path_config.heuristics)
                    )
                    foreach[
                        f"{task.name}__{dataset.id}__{relative_heuristic_group}"
                    ] = {
                        "task": task.name,
                        "dataset": dataset.id,
                        "heuristic_group": relative_heuristic_group,
                    }
        return foreach


class CombineHeuristicsCommand(DvcCommand):
    def to_dvc_template_dict(self) -> Dict:
        cmd = 'bohr porcelain combine-heuristics "${item.exp}" --dataset "${item.dataset}"'
        deps = [str(self.path_config.runs_dir / "__heuristics" / "${item.dataset}")]
        params = [{"bohr.lock": ["${item.exp}.heuristics_classifier"]}]
        outs = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "heuristic_matrix.pkl"
            ),
            {
                str(
                    self.path_config.runs_dir
                    / "${item.task}"
                    / "${item.exp}"
                    / "${item.dataset}"
                    / "analysis.json"
                ): {"cache": False}
            },
        ]
        metrics = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "heuristic_metrics.json"
            )
        ]
        foreach = self.generate_foreach()

        return self._to_template_dict(foreach, cmd, params, deps, outs, metrics)


@dataclass
class TrainLabelModelCommand(DvcCommand):
    exp: Experiment

    def stage_name(self) -> str:
        return f"TrainLabelModel@{self.exp.name}"

    def to_dvc_template_dict(self) -> Dict:
        cmd = f"bohr porcelain train-label-model {self.exp.name}"
        deps = [
            str(
                self.path_config.runs_dir
                / self.exp.task.name
                / self.exp.name
                / dataset.id
                / "heuristic_matrix.pkl"
            )
            for dataset in self.exp.task.datasets
        ]
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
        metrics = [
            str(
                self.path_config.runs_dir
                / self.exp.task.name
                / self.exp.name
                / "label_model_metrics.json"
            )
        ]
        foreach = [0]

        return self._to_template_dict(foreach, cmd, [], deps, outs, metrics)


class LabelDatasetCommand(DvcCommand):
    def to_dvc_template_dict(self) -> Dict:
        cmd = 'bohr porcelain label-dataset "${item.exp}" "${item.dataset}"'
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
        outs = [
            str(
                self.path_config.runs_dir
                / "${item.task}"
                / "${item.exp}"
                / "${item.dataset}"
                / "labeled.csv"
            )
        ]
        metrics = []
        foreach = self.generate_foreach()

        return self._to_template_dict(foreach, cmd, [], deps, outs, metrics)


def dvc_config_from_tasks(
    experiments: List[Experiment], path_config: PathConfig
) -> Dict:
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
    all_experiments = sorted(experiments, key=lambda x: x.name)
    logger.info(
        f"Following tasks are added to the pipeline: {list(map(lambda x: x.name, all_experiments))}"
    )
    if len(all_experiments) == 0:
        raise ValueError("At least of task should be specified")

    train_model_commands = [
        TrainLabelModelCommand(path_config, all_experiments, run)
        for run in all_experiments
    ]
    commands: List[DvcCommand] = [
        LoadDatasetsCommand(path_config, all_experiments),
        ApplyHeuristicsCommand(path_config, all_experiments),
        ComputeSingleHeuristicMetrics(path_config, all_experiments),
        CombineHeuristicsCommand(path_config, all_experiments),
        LabelDatasetCommand(path_config, all_experiments),
    ] + train_model_commands

    final_dict = {"stages": {}}
    for command in commands:
        name, dvc_dct = next(iter(command.to_dvc_template_dict().items()))
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
    dvc_config = dvc_config_from_tasks(workspace.experiments, path_config)
    params = {"bohr_runtime_version": workspace.bohr_runtime_version}
    for run in workspace.experiments:
        heuristic_groups = run.heuristic_groups
        params[run.name] = {
            "heuristics_classifier": ":".join(
                normalize_paths(
                    heuristic_groups, path_config.heuristics, is_heuristic_file
                )
            )
            if heuristic_groups is not None
            else "."
        }
    params_yaml = yaml.dump(params)
    with (path_config.project_root / "bohr.lock").open("w") as f:
        f.write(params_yaml)
    dvc_config_yaml = yaml.dump(dvc_config)
    with (path_config.project_root / "dvc.yaml").open("w") as f:
        f.write(dvc_config_yaml)
