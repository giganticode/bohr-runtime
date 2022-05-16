import logging
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml
from bohrapi.artifacts import Commit
from git import Repo

from bohrruntime.bohrfs import (
    DATASET_TEMPLATE,
    EXPERIMENT_TEMPLATE,
    TASK_TEMPLATE,
    BohrFileSystem,
    BohrFsPath,
)
from bohrruntime.core import BOHR_REMOTE_URL
from bohrruntime.dataset import Dataset
from bohrruntime.heuristics import get_heuristic_files, is_heuristic_file
from bohrruntime.task import Experiment, Task
from bohrruntime.util.paths import AbsolutePath, normalize_paths
from bohrruntime.workspace import Workspace

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
        cmd = 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'
        if self.workspace.experiments[0].task.is_matching_task():
            cmd += " --match"
        return cmd

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

    def get_deps(self):
        deps = [self.fs.experiment_label_matrix_file(self.exp, self.exp.train_dataset)]
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
    >>> dvc_config_from_tasks(Workspace('0.x.x', []), BohrFileSystem.init(Path('/')))
    Traceback (most recent call last):
    ...
    ValueError: At least of task should be specified
    >>> from bohrlabels.core import Label
    >>> from enum import auto
    >>> class TestLabel(Label): Yes = auto(); No = auto()
    >>> train = Dataset("id.train", Commit)
    >>> test = Dataset("id.test", Commit)
    >>> labels = [TestLabel.Yes, TestLabel.No]
    >>> task = Task("name", "author", "desc", Commit, labels, {test: lambda x:x}, TestLabel)
    >>> from bohrruntime import bohr_framework_root
    >>> dvc_config_from_tasks(Workspace('0.x.x', [Experiment('exp', task, train, 'bugginess/conventional_commit_regex')]), BohrFileSystem.init(bohr_framework_root.parent / Path('test-b2b/scenario1')))
    {'stages': {'LoadDatasets': {'foreach': ['id.test', 'id.train'], 'do': {'cmd': 'bohr porcelain load-dataset "${item}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': [], 'outs': ['cached-datasets/${item}.jsonl', {'cached-datasets/${item}.jsonl.metadata.json': {'cache': False}}], 'metrics': []}}, 'ApplyHeuristics': {'foreach': {'id.test__bugginess/conventional_commit_regex.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/conventional_commit_regex.py'}, 'id.test__bugginess/filemetrics/all_files_test_add.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/all_files_test_add.py'}, 'id.test__bugginess/filemetrics/all_files_test_fix.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/all_files_test_fix.py'}, 'id.test__bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py'}, 'id.test__bugginess/filemetrics/buggless_if_many_lines_changed.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/buggless_if_many_lines_changed.py'}, 'id.test__bugginess/filemetrics/bugless_if_at_least_2_removed_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_2_removed_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_at_least_5_added_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_5_added_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_many_files_changes.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_many_files_changes.py'}, 'id.test__bugginess/filemetrics/bugless_if_not_code_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_not_code_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_one_added_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_added_file.py'}, 'id.test__bugginess/filemetrics/bugless_if_one_removed_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_removed_file.py'}, 'id.test__bugginess/filemetrics/no_files_have_modified_status.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/no_files_have_modified_status.py'}, 'id.test__bugginess/filemetrics/refactoring_if_at_least_2_renamed.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/refactoring_if_at_least_2_renamed.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_issue_body.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_body.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_issue_label.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_message.py'}, 'id.test__bugginess/keywords/buggless_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/buggless_keywords_lookup_in_message.py'}, 'id.test__bugginess/keywords/bugless_keywords_lookup_in_issue_body.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py'}, 'id.test__bugginess/keywords/bugless_keywords_lookup_in_issue_label.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'}, 'id.test__bugginess/keywords/github_ref_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/github_ref_in_message.py'}, 'id.test__bugginess/keywords/init_commit_message_keywords.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/init_commit_message_keywords.py'}, 'id.test__bugginess/keywords/version_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/version_in_message.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_init.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_init.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_merge.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_merge.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_refactoring_miner.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_refactoring_miner.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_sstubs.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_sstubs.py'}, 'id.test__bugginess/versionbump/contains_digit_replacement_change.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_digit_replacement_change.py'}, 'id.test__bugginess/versionbump/contains_package_lock_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_package_lock_file.py'}, 'id.test__bugginess/versionbump/contains_python_version_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_python_version_file.py'}, 'id.test__bugginess/versionbump/contains_ruby_version_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_ruby_version_file.py'}, 'id.test__bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py'}, 'id.test__bugginess/versionbump/maven_plugin_version_bump.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/maven_plugin_version_bump.py'}, 'id.test__bugginess/versionbump/version_bump_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_bump_keywords_lookup_in_message.py'}, 'id.test__bugginess/versionbump/version_regex.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex.py'}, 'id.test__bugginess/versionbump/version_regex2.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex2.py'}, 'id.test__bugginess/versionbump/version_regex3.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex3.py'}, 'id.test__bugginess/wip/removed_fixme.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/removed_fixme.py'}, 'id.test__bugginess/wip/removed_todo.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/removed_todo.py'}, 'id.test__bugginess/wip/wip_keyword_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/wip_keyword_in_message.py'}, 'id.train__bugginess/conventional_commit_regex.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/conventional_commit_regex.py'}, 'id.train__bugginess/filemetrics/all_files_test_add.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/all_files_test_add.py'}, 'id.train__bugginess/filemetrics/all_files_test_fix.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/all_files_test_fix.py'}, 'id.train__bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py'}, 'id.train__bugginess/filemetrics/buggless_if_many_lines_changed.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/buggless_if_many_lines_changed.py'}, 'id.train__bugginess/filemetrics/bugless_if_at_least_2_removed_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_2_removed_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_at_least_5_added_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_5_added_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_many_files_changes.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_many_files_changes.py'}, 'id.train__bugginess/filemetrics/bugless_if_not_code_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_not_code_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_one_added_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_added_file.py'}, 'id.train__bugginess/filemetrics/bugless_if_one_removed_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_removed_file.py'}, 'id.train__bugginess/filemetrics/no_files_have_modified_status.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/no_files_have_modified_status.py'}, 'id.train__bugginess/filemetrics/refactoring_if_at_least_2_renamed.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/refactoring_if_at_least_2_renamed.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_issue_body.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_body.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_issue_label.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_message.py'}, 'id.train__bugginess/keywords/buggless_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/buggless_keywords_lookup_in_message.py'}, 'id.train__bugginess/keywords/bugless_keywords_lookup_in_issue_body.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py'}, 'id.train__bugginess/keywords/bugless_keywords_lookup_in_issue_label.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'}, 'id.train__bugginess/keywords/github_ref_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/github_ref_in_message.py'}, 'id.train__bugginess/keywords/init_commit_message_keywords.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/init_commit_message_keywords.py'}, 'id.train__bugginess/keywords/version_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/version_in_message.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_init.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_init.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_merge.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_merge.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_refactoring_miner.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_refactoring_miner.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_sstubs.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_sstubs.py'}, 'id.train__bugginess/versionbump/contains_digit_replacement_change.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_digit_replacement_change.py'}, 'id.train__bugginess/versionbump/contains_package_lock_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_package_lock_file.py'}, 'id.train__bugginess/versionbump/contains_python_version_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_python_version_file.py'}, 'id.train__bugginess/versionbump/contains_ruby_version_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_ruby_version_file.py'}, 'id.train__bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py'}, 'id.train__bugginess/versionbump/maven_plugin_version_bump.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/maven_plugin_version_bump.py'}, 'id.train__bugginess/versionbump/version_bump_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_bump_keywords_lookup_in_message.py'}, 'id.train__bugginess/versionbump/version_regex.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex.py'}, 'id.train__bugginess/versionbump/version_regex2.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex2.py'}, 'id.train__bugginess/versionbump/version_regex3.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex3.py'}, 'id.train__bugginess/wip/removed_fixme.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/removed_fixme.py'}, 'id.train__bugginess/wip/removed_todo.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/removed_todo.py'}, 'id.train__bugginess/wip/wip_keyword_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/wip_keyword_in_message.py'}}, 'do': {'cmd': 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cloned-bohr/heuristics/${item.heuristic_group}', 'cached-datasets/${item.dataset}.jsonl'], 'outs': ['runs/__heuristics/${item.dataset}/${item.heuristic_group}/heuristic_matrix.pkl'], 'metrics': []}}, 'CombineHeuristics': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain combine-heuristics "${item.exp}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['experiments.${item.exp}.heuristics_classifier']}, {'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/__heuristics/${item.dataset}'], 'outs': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl'], 'metrics': []}}, 'RunMetricsAndAnalysis': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain run-metrics-and-analysis "${item.exp}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl', 'runs/${item.task}/${item.exp}/label_model.pkl', 'cached-datasets/${item.dataset}.jsonl'], 'outs': [{'runs/${item.task}/${item.exp}/${item.dataset}/analysis.json': {'cache': False}}, {'runs/${item.task}/${item.exp}/${item.dataset}/analysis.csv': {'cache': False}}], 'metrics': [{'runs/${item.task}/${item.exp}/${item.dataset}/metrics.txt': {'cache': False}}]}}, 'ComputeRandomModelMetrics': {'foreach': {'name__id.test': {'dataset': 'id.test', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain compute-random-model-metrics "${item.task}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/${item.dataset}.jsonl'], 'outs': [], 'metrics': [{'runs/${item.task}/random_model/${item.dataset}/metrics.txt': {'cache': False}}]}}, 'ComputeZeroModelMetrics': {'foreach': {'name__id.test': {'dataset': 'id.test', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain compute-zero-model-metrics "${item.task}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/${item.dataset}.jsonl'], 'outs': [], 'metrics': [{'runs/${item.task}/zero_model/${item.dataset}/metrics.txt': {'cache': False}}]}}, 'LabelDataset': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain label-dataset "${item.exp}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl', 'runs/${item.task}/${item.exp}/label_model.pkl'], 'outs': ['runs/${item.task}/${item.exp}/${item.dataset}/labeled.csv'], 'metrics': []}}, 'TrainLabelModel__exp': {'cmd': 'bohr porcelain train-label-model exp', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/name/exp/id.train/heuristic_matrix.pkl'], 'outs': ['runs/name/exp/label_model.pkl', 'runs/name/exp/label_model_weights.csv'], 'metrics': []}, 'ComputeSingleHeuristicMetrics__name': {'cmd': 'bohr porcelain compute-single-heuristic-metric name', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/id.test.jsonl', 'cloned-bohr/heuristics/bugginess/conventional_commit_regex.py', 'runs/__heuristics/id.test/bugginess/conventional_commit_regex.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/all_files_test_add.py', 'runs/__heuristics/id.test/bugginess/filemetrics/all_files_test_add.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/all_files_test_fix.py', 'runs/__heuristics/id.test/bugginess/filemetrics/all_files_test_fix.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/buggless_if_many_lines_changed.py', 'runs/__heuristics/id.test/bugginess/filemetrics/buggless_if_many_lines_changed.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_at_least_2_removed_files.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_at_least_2_removed_files.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_at_least_5_added_files.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_at_least_5_added_files.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_many_files_changes.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_many_files_changes.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_not_code_files.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_not_code_files.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_one_added_file.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_one_added_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_one_removed_file.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_one_removed_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/no_files_have_modified_status.py', 'runs/__heuristics/id.test/bugginess/filemetrics/no_files_have_modified_status.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/refactoring_if_at_least_2_renamed.py', 'runs/__heuristics/id.test/bugginess/filemetrics/refactoring_if_at_least_2_renamed.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bug_keywords_lookup_in_issue_body.py', 'runs/__heuristics/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_body.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bug_keywords_lookup_in_issue_label.py', 'runs/__heuristics/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_label.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bug_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/bug_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/buggless_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/buggless_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bugless_keywords_lookup_in_issue_body.py', 'runs/__heuristics/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_body.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bugless_keywords_lookup_in_issue_label.py', 'runs/__heuristics/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_label.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/github_ref_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/github_ref_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/init_commit_message_keywords.py', 'runs/__heuristics/id.test/bugginess/keywords/init_commit_message_keywords.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/version_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/version_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_init.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_init.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_merge.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_merge.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_refactoring_miner.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_refactoring_miner.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_sstubs.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_sstubs.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_digit_replacement_change.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_digit_replacement_change.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_package_lock_file.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_package_lock_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_python_version_file.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_python_version_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_ruby_version_file.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_ruby_version_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/maven_plugin_version_bump.py', 'runs/__heuristics/id.test/bugginess/versionbump/maven_plugin_version_bump.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_bump_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_bump_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_regex.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_regex.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_regex2.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_regex2.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_regex3.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_regex3.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/wip/removed_fixme.py', 'runs/__heuristics/id.test/bugginess/wip/removed_fixme.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/wip/removed_todo.py', 'runs/__heuristics/id.test/bugginess/wip/removed_todo.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/wip/wip_keyword_in_message.py', 'runs/__heuristics/id.test/bugginess/wip/wip_keyword_in_message.py/heuristic_matrix.pkl'], 'outs': [{'runs/__single_heuristic_metrics/name/id.test/bugginess/conventional_commit_regex.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/all_files_test_add.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/all_files_test_fix.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/buggless_if_many_lines_changed.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_at_least_2_removed_files.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_at_least_5_added_files.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_many_files_changes.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_not_code_files.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_one_added_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_one_removed_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/no_files_have_modified_status.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/refactoring_if_at_least_2_renamed.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_body.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_label.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bug_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/buggless_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_body.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_label.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/github_ref_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/init_commit_message_keywords.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/version_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_init.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_merge.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_refactoring_miner.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_sstubs.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_digit_replacement_change.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_package_lock_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_python_version_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_ruby_version_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/maven_plugin_version_bump.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_bump_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_regex.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_regex2.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_regex3.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/wip/removed_fixme.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/wip/removed_todo.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/wip/wip_keyword_in_message.py/metrics.txt': {'cache': False}}], 'metrics': []}}}
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
    rev_to_checkout: Optional[str], heuristics_root: AbsolutePath
) -> None:
    """
    >>> fetch_heuristics_if_needed('f1690ba', '/some/path')
    Traceback (most recent call last):
    ...
    ValueError: Invalid revision format (has to be SHA hashsum): f1690ba
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     fetch_heuristics_if_needed(None, Path(tmpdirname) / 'repo')
    ...     fetch_heuristics_if_needed(None, Path(tmpdirname) / 'repo')
    ...     fetch_heuristics_if_needed('f1690baca451b147b0e704c1ad996338318cc9a6', Path(tmpdirname) / 'repo2')
    ...     fetch_heuristics_if_needed('1e0880b5ddb005b46d5c489cc51a984229d962dc', Path(tmpdirname) / 'repo2')
    ...     fetch_heuristics_if_needed('1e0880b5ddb005b46d5c489cc51a984229d962dc', Path(tmpdirname) / 'repo2')
    ...     shutil.rmtree(Path(tmpdirname) / 'repo2' / 'heuristics')
    ...     fetch_heuristics_if_needed('1e0880b5ddb005b46d5c489cc51a984229d962dc', Path(tmpdirname) / 'repo2')
    Warning: downloaded heuristics have been modified!
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     fetch_heuristics_if_needed('1e0880b5ddb005b46d5c489cc51a984229d962dc', Path(tmpdirname) / 'repo')
    ...     shutil.rmtree(Path(tmpdirname) / 'repo' / 'heuristics')
    ...     fetch_heuristics_if_needed('f1690baca451b147b0e704c1ad996338318cc9a6', Path(tmpdirname) / 'repo')
    Traceback (most recent call last):
    ...
    RuntimeError: Need to checkout revision f1690baca451b147b0e704c1ad996338318cc9a6, however the current revision 1e0880b5ddb005b46d5c489cc51a984229d962dc is dirty
    """
    if rev_to_checkout is not None and not re.fullmatch(
        "[0-9a-fA-F]{40}", rev_to_checkout
    ):
        raise ValueError(
            f"Invalid revision format (has to be SHA hashsum): {rev_to_checkout}"
        )

    clone_repo = False
    if heuristics_root.exists():
        repo = Repo(heuristics_root)
        if rev_to_checkout is None:
            last_remote_commit = repo.remote().fetch()[0].commit
        head_sha = repo.head.commit.hexsha
        if head_sha == (
            rev_to_checkout if rev_to_checkout else last_remote_commit.hexsha
        ):
            if repo.is_dirty():
                print("Warning: downloaded heuristics have been modified!")
        else:
            if repo.is_dirty():
                raise RuntimeError(
                    f"Need to checkout revision {(rev_to_checkout if rev_to_checkout else last_remote_commit.hexsha)}, "
                    f"however the current revision {head_sha} is dirty"
                )
            shutil.rmtree(heuristics_root)
            clone_repo = True

    else:
        clone_repo = True

    if clone_repo:
        repo = Repo.clone_from(
            BOHR_REMOTE_URL,
            heuristics_root,
            no_checkout=rev_to_checkout is not None,
        )
        if rev_to_checkout is not None:
            repo.git.checkout(rev_to_checkout)


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
