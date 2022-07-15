import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from bohrapi.artifacts import Commit
from git import Repo
from tqdm import tqdm

from bohrruntime import BOHR_REMOTE_URL
from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment, SynteticExperiment
from bohrruntime.datamodel.task import Task
from bohrruntime.datamodel.workspace import Workspace
from bohrruntime.heuristicuri import HeuristicURI, PathTemplate
from bohrruntime.stages import (
    apply_heuristics_to_dataset,
    calculate_experiment_metrics,
    calculate_single_heuristic_metrics,
    combine_applied_heuristics,
    load_dataset,
    prepare_dataset,
)
from bohrruntime.storageengine import StorageEngine
from bohrruntime.tasktypes.labeling.lfs import SnorkelHeuristicApplier  # TODO!!
from bohrruntime.util.paths import AbsolutePath, normalize_paths

logger = logging.getLogger(__name__)


TASK_TEMPLATE = SimpleNamespace(name="${item.task}")
EXPERIMENT_TEMPLATE = SimpleNamespace(name="${item.exp}", task=TASK_TEMPLATE)
DATASET_TEMPLATE = SimpleNamespace(id="${item.dataset}")


def iterate_workspace(
    workspace: Workspace, fs: StorageEngine, iterate_heuristics: bool = True
) -> Union[List[Tuple[Experiment, Dataset]], List[Tuple[Experiment, Dataset, str]]]:
    """
    >>> from bohrruntime.testtools import get_stub_experiment, VirtualStorageEngine
    >>> workspace = Workspace('0.0.1', [get_stub_experiment(no_training_dataset=True)])
    >>> fs = VirtualStorageEngine()
    >>> (experiment1, dataset1, heuristic_group1), (experiment2, dataset2, heuristic_group2) = iterate_workspace(workspace, fs)
    >>> experiment1 is experiment2
    True
    >>> experiment1
    Experiment(name='stub-exp', task=LabelingTask(name='stub-task', author='stub-author', description='stub-description', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, test_datasets={Dataset(id='stub-dataset', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, query=None, projection=None, n_datapoints=None): None}, labels=(NumericLabel(label=536870896, hierarchy=<enum 'CommitLabel'>), NumericLabel(label=15, hierarchy=<enum 'CommitLabel'>)), class_balance=None), train_dataset=None, heuristics_classifier=None, extra_test_datasets=frozendict.frozendict({}))
    >>> dataset1 is dataset2
    True
    >>> dataset1
    Dataset(id='stub-dataset', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, query=None, projection=None, n_datapoints=None)
    >>> heuristic_group1
    '/heuristic1'
    >>> heuristic_group2
    '/heuristic2'
    >>> (experiment3, dataset3, heuristic_group3), = iterate_workspace(workspace, fs, iterate_heuristics=False)
    >>> experiment3 is experiment1
    True
    >>> dataset3 is dataset1
    True
    >>> heuristic_group3 is None
    True
    """
    heuristic_loader = fs.get_heuristic_loader()
    for experiment in sorted(workspace.experiments, key=lambda x: x.name):
        heuristic_groups = heuristic_loader.get_heuristic_uris(
            input_artifact_type=experiment.task.heuristic_input_artifact_type
        )
        for dataset in experiment.datasets:
            if iterate_heuristics:
                for heuristic_group in heuristic_groups:
                    yield (experiment, dataset, str(heuristic_group.path))
            else:
                yield (experiment, dataset, None)


def stringify_paths(
    paths: List[Union[str, Dict[str, Any]]]
) -> List[Union[str, Dict[str, Any]]]:
    res = []
    for path in paths:
        if isinstance(path, str):
            res.append(str(path))
        elif isinstance(path, Dict):
            if len(path) != 1:
                raise AssertionError()
            key, value = next(iter(path.items()))
            if not isinstance(key, str):
                raise AssertionError()
            res.append({str(key): value})
        else:
            raise AssertionError()
    return res


@dataclass(repr=False)
class Stage(ABC):
    fs: StorageEngine

    @abstractmethod
    def execute(self) -> None:
        pass

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
                "always_changed": self.is_always_changed(),
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

    def __str__(self):
        return str(self.to_dvc_config_dict())

    def __repr__(self):
        return str(self.to_dvc_config_dict())

    def is_always_changed(self):
        return False


@dataclass(repr=False)
class MultipleCommandStage(Stage):
    workspace: Workspace

    def n_stages(self) -> int:
        return len(self.get_iterating_over())

    def get_for_each(self) -> Dict[str, Dict[str, Any]]:
        foreach: Dict[str, Dict[str, Any]] = {}
        for entry in self.get_iterating_over():
            foreach = {
                **foreach,
                **self.generate_for_each_entry(entry),
            }
        return foreach

    def to_dvc_config_dict(
        self,
    ) -> Dict:
        parent = next(
            iter(super(MultipleCommandStage, self).to_dvc_config_dict().values())
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

    def generate_for_each_entry(self, entry) -> Dict[str, Dict[str, str]]:
        experiment, dataset = entry
        return {
            f"{experiment.name}__{dataset.id}": {
                "dataset": dataset.id,
                "exp": experiment.name,
                "task": experiment.task.name,
            }
        }


class LoadDatasetsCommand(MultipleCommandStage):
    def execute(self) -> None:
        for dataset in self.get_iterating_over():
            load_dataset(dataset, self.fs)

    def get_iterating_over(self) -> Sequence:
        return sorted({d for exp in self.workspace.experiments for d in exp.datasets})

    def get_for_each(self) -> List[str]:
        return [d.id for d in self.get_iterating_over()]

    def get_cmd(self) -> str:
        return 'bohr porcelain load-dataset "${item}"'

    def get_outs(self) -> List[str]:
        outs = [
            self.fs.path_structure.dataset("${item}"),
            {self.fs.path_structure.dataset_metadata("${item}"): {"cache": False}},
        ]
        return outs

    def is_always_changed(self):
        for dataset in self.get_iterating_over():
            if dataset.path is not None:
                return True  # TODO here if at least one dataset is local all the load stages will be reprodices - fix
        return False


class ApplyHeuristicsCommand(MultipleCommandStage):
    def execute(self) -> None:
        for dataset, heuristic_group in self.get_iterating_over():
            apply_heuristics_to_dataset(
                SnorkelHeuristicApplier(), heuristic_group, dataset, self.fs
            )

    def get_cmd(self) -> str:
        cmd = 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'
        return cmd

    def get_deps(self) -> List[str]:
        deps = [
            self.fs.path_structure.heuristic_group(
                PathTemplate("${item.heuristic_group}")
            ),
            self.fs.path_structure.dataset(DATASET_TEMPLATE.id),
        ]
        return deps

    def get_outs(self) -> List[Any]:
        outs = [
            self.fs.heuristic_matrix_file(
                DATASET_TEMPLATE, PathTemplate("${item.heuristic_group}")
            )
        ]
        return outs

    def get_iterating_over(self) -> Sequence:
        return sorted(
            {(d, h) for _, d, h in iterate_workspace(self.workspace, self.fs, True)},
            key=lambda d: (d[0], d[1]),
        )

    def generate_for_each_entry(self, entry) -> Dict[str, Dict[str, str]]:
        dataset, heuristic_group = entry

        return {
            f"{dataset.id}__{heuristic_group}": {
                "dataset": dataset.id,
                "heuristic_group": heuristic_group,
            }
        }


@dataclass(repr=False)
class ComputeSingleHeuristicMetricsStage(Stage):
    task: Task

    def execute(self) -> None:
        calculate_single_heuristic_metrics(self.task, self.fs)

    def stage_name(self) -> str:
        return f"ComputeSingleHeuristicMetrics__{self.task.name}"

    def get_cmd(self) -> str:
        return f"bohr porcelain compute-single-heuristic-metric {self.task.name}"

    def get_deps(self) -> List[str]:
        deps = []
        heuristic_loader = self.fs.get_heuristic_loader()
        for dataset in self.task.test_datasets:
            deps.append(self.fs.path_structure.dataset(dataset.id))
        for heuristic_group in heuristic_loader.get_heuristic_uris(
            input_artifact_type=self.task.heuristic_input_artifact_type
        ):
            deps.append(self.fs.path_structure.heuristic_group(heuristic_group))
            for dataset in self.task.test_datasets:
                deps.append(self.fs.heuristic_matrix_file(dataset, heuristic_group))
        return deps

    def get_outs(self) -> List[str]:
        outputs = []
        heuristic_loader = self.fs.get_heuristic_loader()
        for dataset in self.task.test_datasets:
            heuristic_groups = heuristic_loader.get_heuristic_uris(
                input_artifact_type=self.task.heuristic_input_artifact_type
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


class FetchMultipleHeuristicOutputsCommand(MultipleCommandStage):
    def execute(self) -> None:
        for experiment, dataset in self.get_iterating_over():
            combine_applied_heuristics(experiment, dataset, self.fs)

    def get_cmd(self) -> str:
        return 'bohr porcelain combine-heuristics "${item.exp}" --dataset "${item.dataset}"'

    def get_deps(self) -> List[str]:
        return [self.fs.path_structure.heuristic_dataset_dir(DATASET_TEMPLATE)]

    def get_params(self) -> List:
        return [{"bohr.lock": ["experiments.${item.exp}.heuristics_classifier"]}]

    def get_outs(self) -> List[str]:
        outs = [
            self.fs.path_structure.experiment_label_matrix_file(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            )
        ]
        return outs


class CalculateMetricsCommand(MultipleCommandStage):
    def execute(self) -> None:
        for experiment, dataset in self.get_iterating_over():
            calculate_experiment_metrics(experiment, dataset, self.fs)

    def get_metrics(self) -> List:
        metrics = [
            str(
                self.fs.path_structure.experiment_metrics(
                    EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
                )
            )
        ]
        return metrics

    def get_cmd(self) -> str:
        return 'bohr porcelain run-metrics-and-analysis "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            self.fs.path_structure.experiment_label_matrix_file(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            ),
            self.fs.path_structure.label_model(EXPERIMENT_TEMPLATE),
            self.fs.path_structure.dataset(DATASET_TEMPLATE.id),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            {
                self.fs.path_structure.analysis_json(
                    EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
                ): {"cache": False}
            },
            {
                self.fs.path_structure.analysis_csv(
                    EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
                ): {"cache": False}
            },
        ]
        return outs


class ComputePredefinedModelMetricsCommand(MultipleCommandStage):
    def execute(self) -> None:
        for task, dataset in self.get_iterating_over():
            exp = SynteticExperiment("random_model", task, type="random")
            calculate_experiment_metrics(exp, dataset, self.fs)

    def get_model_name(self) -> str:
        return f"{self.get_model_type()}_model"

    @abstractmethod
    def get_model_type(self) -> str:
        pass

    def get_metrics(self) -> List:
        metrics = [
            str(
                self.fs.path_structure.experiment_metrics(
                    SimpleNamespace(name=self.get_model_name(), task=TASK_TEMPLATE),
                    DATASET_TEMPLATE,
                )
            )
        ]
        return metrics

    def get_deps(self) -> List:
        return [self.fs.path_structure.dataset(DATASET_TEMPLATE.id)]

    def get_iterating_over(self) -> Sequence:
        all_tasks = {exp.task for exp in self.workspace.experiments}
        for task in sorted(all_tasks, key=lambda k: k.name):
            for dataset in sorted(task.test_datasets):
                yield (task, dataset)

    def generate_for_each_entry(self, entry) -> Dict[str, Dict[str, str]]:
        task, dataset = entry
        return {
            f"{task.name}__{dataset.id}": {
                "dataset": dataset.id,
                "task": task.name,
            }
        }

    def get_cmd(self) -> str:
        return (
            "bohr porcelain compute-"
            + self.get_model_type()
            + '-model-metrics "${item.task}" "${item.dataset}"'
        )


class ComputeRandomModelMetricsCommand(ComputePredefinedModelMetricsCommand):
    def get_model_type(self) -> str:
        return "random"


class ComputeZeroModelMetricsCommand(ComputePredefinedModelMetricsCommand):
    def get_model_type(self) -> str:
        return "zero"


@dataclass(repr=False)
class TrainModelStage(Stage):
    exp: Experiment

    def execute(self) -> None:
        pass

    def stage_name(self) -> str:
        return f"TrainModel__{self.exp.name}"

    def get_cmd(self):
        return f"bohr porcelain train-model {self.exp.name}"

    def get_deps(self) -> List[str]:
        deps = [
            self.fs.path_structure.experiment_label_matrix_file(
                self.exp, self.exp.train_dataset
            )
        ]
        return deps

    def get_outs(self) -> List[str]:
        outs = [
            self.fs.path_structure.label_model(self.exp),
            self.fs.path_structure.label_model_weights(self.exp),
        ]
        return outs


class PrepareDatasetCommand(MultipleCommandStage):
    def execute(self) -> None:
        for experiment, dataset in tqdm(self.get_iterating_over()):
            prepare_dataset(experiment, dataset, self.fs)

    def get_cmd(self) -> str:
        return 'bohr porcelain prepare-dataset "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            self.fs.path_structure.experiment_label_matrix_file(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            ),
            self.fs.path_structure.label_model(EXPERIMENT_TEMPLATE),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            self.fs.path_structure.labeled_dataset(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            )
        ]
        return outs


def get_stages_list(workspace: Workspace, fs: StorageEngine) -> List[Stage]:
    """
    >>> from bohrlabels.core import Label
    >>> from enum import auto
    >>> from bohrruntime.tasktypes.labeling.core import LabelingTask
    >>> from frozendict import frozendict
    >>> class TestLabel(Label): Yes = auto(); No = auto()
    >>> train = Dataset("id.train", Commit)
    >>> test = Dataset("id.test", Commit)
    >>> labels = (TestLabel.Yes, TestLabel.No)
    >>> task = LabelingTask("name", "author", "desc", Commit, labels, frozendict({test: lambda x:x}), TestLabel)
    >>> from bohrruntime import bohr_framework_root
    >>> get_stages_list(Workspace('0.x.x', [Experiment('exp', task, train, 'bugginess/conventional_commit_regex')]), StorageEngine.init(bohr_framework_root.parent / Path('test-b2b/scenario1')))
    [{'LoadDatasets': {'foreach': ['id.test', 'id.train'], 'do': {'cmd': 'bohr porcelain load-dataset "${item}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': [], 'outs': ['cached-datasets/${item}.jsonl', {'cached-datasets/${item}.jsonl.metadata.json': {'cache': False}}], 'metrics': []}}}, {'ApplyHeuristics': {'foreach': {'id.test__bugginess/conventional_commit_regex.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/conventional_commit_regex.py'}, 'id.test__bugginess/filemetrics/all_files_test_add.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/all_files_test_add.py'}, 'id.test__bugginess/filemetrics/all_files_test_fix.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/all_files_test_fix.py'}, 'id.test__bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py'}, 'id.test__bugginess/filemetrics/buggless_if_many_lines_changed.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/buggless_if_many_lines_changed.py'}, 'id.test__bugginess/filemetrics/bugless_if_at_least_2_removed_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_2_removed_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_at_least_5_added_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_5_added_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_many_files_changes.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_many_files_changes.py'}, 'id.test__bugginess/filemetrics/bugless_if_not_code_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_not_code_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_one_added_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_added_file.py'}, 'id.test__bugginess/filemetrics/bugless_if_one_removed_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_removed_file.py'}, 'id.test__bugginess/filemetrics/no_files_have_modified_status.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/no_files_have_modified_status.py'}, 'id.test__bugginess/filemetrics/refactoring_if_at_least_2_renamed.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/refactoring_if_at_least_2_renamed.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_issue_body.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_body.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_issue_label.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_message.py'}, 'id.test__bugginess/keywords/buggless_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/buggless_keywords_lookup_in_message.py'}, 'id.test__bugginess/keywords/bugless_keywords_lookup_in_issue_body.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py'}, 'id.test__bugginess/keywords/bugless_keywords_lookup_in_issue_label.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'}, 'id.test__bugginess/keywords/github_ref_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/github_ref_in_message.py'}, 'id.test__bugginess/keywords/init_commit_message_keywords.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/init_commit_message_keywords.py'}, 'id.test__bugginess/keywords/version_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/version_in_message.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_init.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_init.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_merge.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_merge.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_refactoring_miner.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_refactoring_miner.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_sstubs.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_sstubs.py'}, 'id.test__bugginess/versionbump/contains_digit_replacement_change.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_digit_replacement_change.py'}, 'id.test__bugginess/versionbump/contains_package_lock_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_package_lock_file.py'}, 'id.test__bugginess/versionbump/contains_python_version_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_python_version_file.py'}, 'id.test__bugginess/versionbump/contains_ruby_version_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_ruby_version_file.py'}, 'id.test__bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py'}, 'id.test__bugginess/versionbump/maven_plugin_version_bump.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/maven_plugin_version_bump.py'}, 'id.test__bugginess/versionbump/version_bump_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_bump_keywords_lookup_in_message.py'}, 'id.test__bugginess/versionbump/version_regex.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex.py'}, 'id.test__bugginess/versionbump/version_regex2.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex2.py'}, 'id.test__bugginess/versionbump/version_regex3.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex3.py'}, 'id.test__bugginess/wip/removed_fixme.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/removed_fixme.py'}, 'id.test__bugginess/wip/removed_todo.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/removed_todo.py'}, 'id.test__bugginess/wip/wip_keyword_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/wip_keyword_in_message.py'}, 'id.train__bugginess/conventional_commit_regex.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/conventional_commit_regex.py'}, 'id.train__bugginess/filemetrics/all_files_test_add.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/all_files_test_add.py'}, 'id.train__bugginess/filemetrics/all_files_test_fix.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/all_files_test_fix.py'}, 'id.train__bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py'}, 'id.train__bugginess/filemetrics/buggless_if_many_lines_changed.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/buggless_if_many_lines_changed.py'}, 'id.train__bugginess/filemetrics/bugless_if_at_least_2_removed_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_2_removed_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_at_least_5_added_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_5_added_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_many_files_changes.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_many_files_changes.py'}, 'id.train__bugginess/filemetrics/bugless_if_not_code_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_not_code_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_one_added_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_added_file.py'}, 'id.train__bugginess/filemetrics/bugless_if_one_removed_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_removed_file.py'}, 'id.train__bugginess/filemetrics/no_files_have_modified_status.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/no_files_have_modified_status.py'}, 'id.train__bugginess/filemetrics/refactoring_if_at_least_2_renamed.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/refactoring_if_at_least_2_renamed.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_issue_body.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_body.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_issue_label.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_message.py'}, 'id.train__bugginess/keywords/buggless_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/buggless_keywords_lookup_in_message.py'}, 'id.train__bugginess/keywords/bugless_keywords_lookup_in_issue_body.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py'}, 'id.train__bugginess/keywords/bugless_keywords_lookup_in_issue_label.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'}, 'id.train__bugginess/keywords/github_ref_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/github_ref_in_message.py'}, 'id.train__bugginess/keywords/init_commit_message_keywords.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/init_commit_message_keywords.py'}, 'id.train__bugginess/keywords/version_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/version_in_message.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_init.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_init.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_merge.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_merge.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_refactoring_miner.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_refactoring_miner.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_sstubs.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_sstubs.py'}, 'id.train__bugginess/versionbump/contains_digit_replacement_change.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_digit_replacement_change.py'}, 'id.train__bugginess/versionbump/contains_package_lock_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_package_lock_file.py'}, 'id.train__bugginess/versionbump/contains_python_version_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_python_version_file.py'}, 'id.train__bugginess/versionbump/contains_ruby_version_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_ruby_version_file.py'}, 'id.train__bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py'}, 'id.train__bugginess/versionbump/maven_plugin_version_bump.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/maven_plugin_version_bump.py'}, 'id.train__bugginess/versionbump/version_bump_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_bump_keywords_lookup_in_message.py'}, 'id.train__bugginess/versionbump/version_regex.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex.py'}, 'id.train__bugginess/versionbump/version_regex2.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex2.py'}, 'id.train__bugginess/versionbump/version_regex3.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex3.py'}, 'id.train__bugginess/wip/removed_fixme.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/removed_fixme.py'}, 'id.train__bugginess/wip/removed_todo.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/removed_todo.py'}, 'id.train__bugginess/wip/wip_keyword_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/wip_keyword_in_message.py'}}, 'do': {'cmd': 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cloned-bohr/heuristics/${item.heuristic_group}', 'cached-datasets/${item.dataset}.jsonl'], 'outs': ['runs/__heuristics/${item.dataset}/${item.heuristic_group}/heuristic_matrix.pkl'], 'metrics': []}}}, [{'ComputeSingleHeuristicMetrics__name': {'cmd': 'bohr porcelain compute-single-heuristic-metric name', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/id.test.jsonl', 'cloned-bohr/heuristics/bugginess/conventional_commit_regex.py', 'runs/__heuristics/id.test/bugginess/conventional_commit_regex.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/all_files_test_add.py', 'runs/__heuristics/id.test/bugginess/filemetrics/all_files_test_add.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/all_files_test_fix.py', 'runs/__heuristics/id.test/bugginess/filemetrics/all_files_test_fix.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/buggless_if_many_lines_changed.py', 'runs/__heuristics/id.test/bugginess/filemetrics/buggless_if_many_lines_changed.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_at_least_2_removed_files.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_at_least_2_removed_files.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_at_least_5_added_files.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_at_least_5_added_files.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_many_files_changes.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_many_files_changes.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_not_code_files.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_not_code_files.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_one_added_file.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_one_added_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/bugless_if_one_removed_file.py', 'runs/__heuristics/id.test/bugginess/filemetrics/bugless_if_one_removed_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/no_files_have_modified_status.py', 'runs/__heuristics/id.test/bugginess/filemetrics/no_files_have_modified_status.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/filemetrics/refactoring_if_at_least_2_renamed.py', 'runs/__heuristics/id.test/bugginess/filemetrics/refactoring_if_at_least_2_renamed.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bug_keywords_lookup_in_issue_body.py', 'runs/__heuristics/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_body.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bug_keywords_lookup_in_issue_label.py', 'runs/__heuristics/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_label.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bug_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/bug_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/buggless_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/buggless_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bugless_keywords_lookup_in_issue_body.py', 'runs/__heuristics/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_body.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/bugless_keywords_lookup_in_issue_label.py', 'runs/__heuristics/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_label.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/github_ref_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/github_ref_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/init_commit_message_keywords.py', 'runs/__heuristics/id.test/bugginess/keywords/init_commit_message_keywords.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/keywords/version_in_message.py', 'runs/__heuristics/id.test/bugginess/keywords/version_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_init.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_init.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_merge.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_merge.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_refactoring_miner.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_refactoring_miner.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/sstubs/commit_explorer_output_sstubs.py', 'runs/__heuristics/id.test/bugginess/sstubs/commit_explorer_output_sstubs.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_digit_replacement_change.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_digit_replacement_change.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_package_lock_file.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_package_lock_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_python_version_file.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_python_version_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/contains_ruby_version_file.py', 'runs/__heuristics/id.test/bugginess/versionbump/contains_ruby_version_file.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/maven_plugin_version_bump.py', 'runs/__heuristics/id.test/bugginess/versionbump/maven_plugin_version_bump.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_bump_keywords_lookup_in_message.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_bump_keywords_lookup_in_message.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_regex.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_regex.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_regex2.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_regex2.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/versionbump/version_regex3.py', 'runs/__heuristics/id.test/bugginess/versionbump/version_regex3.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/wip/removed_fixme.py', 'runs/__heuristics/id.test/bugginess/wip/removed_fixme.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/wip/removed_todo.py', 'runs/__heuristics/id.test/bugginess/wip/removed_todo.py/heuristic_matrix.pkl', 'cloned-bohr/heuristics/bugginess/wip/wip_keyword_in_message.py', 'runs/__heuristics/id.test/bugginess/wip/wip_keyword_in_message.py/heuristic_matrix.pkl'], 'outs': [{'runs/__single_heuristic_metrics/name/id.test/bugginess/conventional_commit_regex.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/all_files_test_add.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/all_files_test_fix.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/buggless_if_many_lines_changed.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_at_least_2_removed_files.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_at_least_5_added_files.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_many_files_changes.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_not_code_files.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_one_added_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/bugless_if_one_removed_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/no_files_have_modified_status.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/filemetrics/refactoring_if_at_least_2_renamed.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_body.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bug_keywords_lookup_in_issue_label.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bug_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/buggless_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_body.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/bugless_keywords_lookup_in_issue_label.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/github_ref_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/init_commit_message_keywords.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/keywords/version_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_init.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_merge.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_refactoring_miner.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/sstubs/commit_explorer_output_sstubs.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_digit_replacement_change.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_package_lock_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_python_version_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/contains_ruby_version_file.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/maven_plugin_version_bump.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_bump_keywords_lookup_in_message.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_regex.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_regex2.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/versionbump/version_regex3.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/wip/removed_fixme.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/wip/removed_todo.py/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test/bugginess/wip/wip_keyword_in_message.py/metrics.txt': {'cache': False}}], 'metrics': []}}], {'FetchMultipleHeuristicOutputs': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain combine-heuristics "${item.exp}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['experiments.${item.exp}.heuristics_classifier']}, {'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/__heuristics/${item.dataset}'], 'outs': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl'], 'metrics': []}}}, [{'TrainModel__exp': {'cmd': 'bohr porcelain train-model exp', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/name/exp/id.train/heuristic_matrix.pkl'], 'outs': ['runs/name/exp/label_model.pkl', 'runs/name/exp/label_model_weights.csv'], 'metrics': []}}], {'PrepareDataset': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain prepare-dataset "${item.exp}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl', 'runs/${item.task}/${item.exp}/label_model.pkl'], 'outs': ['runs/${item.task}/${item.exp}/${item.dataset}/labeled.csv'], 'metrics': []}}}, {'ComputeRandomModelMetrics': {'foreach': {'name__id.test': {'dataset': 'id.test', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain compute-random-model-metrics "${item.task}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/${item.dataset}.jsonl'], 'outs': [], 'metrics': [{'runs/${item.task}/random_model/${item.dataset}/metrics.txt': {'cache': False}}]}}}, {'ComputeZeroModelMetrics': {'foreach': {'name__id.test': {'dataset': 'id.test', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain compute-zero-model-metrics "${item.task}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/${item.dataset}.jsonl'], 'outs': [], 'metrics': [{'runs/${item.task}/zero_model/${item.dataset}/metrics.txt': {'cache': False}}]}}}, {'CalculateMetrics': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr porcelain run-metrics-and-analysis "${item.exp}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl', 'runs/${item.task}/${item.exp}/label_model.pkl', 'cached-datasets/${item.dataset}.jsonl'], 'outs': [{'runs/${item.task}/${item.exp}/${item.dataset}/analysis.json': {'cache': False}}, {'runs/${item.task}/${item.exp}/${item.dataset}/analysis.csv': {'cache': False}}], 'metrics': [{'runs/${item.task}/${item.exp}/${item.dataset}/metrics.txt': {'cache': False}}]}}}]
    """
    if len(workspace.experiments) == 0:
        raise ValueError("At least of task should be specified")

    all_tasks = sorted(
        {exp.task for exp in workspace.experiments}, key=lambda t: t.name
    )

    stages: List[Union[MultipleCommandStage, List[Stage]]] = [
        LoadDatasetsCommand(fs, workspace),
        ApplyHeuristicsCommand(fs, workspace),
        [ComputeSingleHeuristicMetricsStage(fs, task) for task in all_tasks],
        FetchMultipleHeuristicOutputsCommand(fs, workspace),
        [
            TrainModelStage(fs, exp)
            for exp in sorted(workspace.experiments, key=lambda x: x.name)
        ],
        PrepareDatasetCommand(fs, workspace),
        ComputeRandomModelMetricsCommand(fs, workspace),
        ComputeZeroModelMetricsCommand(fs, workspace),
        CalculateMetricsCommand(fs, workspace),
    ]
    return stages


def dvc_config_from_tasks(stages: List[Stage]) -> Dict:
    """
    >>> from bohrlabels.core import Label
    >>> from enum import auto
    >>> from bohrruntime.tasktypes.labeling.core import LabelingTask
    >>> class TestLabel(Label): Yes = auto(); No = auto()
    >>> train = Dataset("id.train", Commit)
    >>> test = Dataset("id.test", Commit)
    >>> labels = (TestLabel.Yes, TestLabel.No)
    >>> task = LabelingTask("name", "author", "desc", Commit, labels, {test: lambda x:x}, TestLabel)
    >>> from bohrruntime import bohr_framework_root
    >>> fs = StorageEngine.init(bohr_framework_root.parent / Path('test-b2b/scenario1'))
    >>> workspace = Workspace('0.x.x', [Experiment('exp', task, train, 'bugginess/conventional_commit_regex')])
    >>> stages = [LoadDatasetsCommand(fs, workspace), ApplyHeuristicsCommand(fs, workspace)]
    >>> dvc_config_from_tasks(stages)
    {'stages': {'LoadDatasets': {'foreach': ['id.test', 'id.train'], 'do': {'cmd': 'bohr porcelain load-dataset "${item}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': [], 'outs': ['cached-datasets/${item}.jsonl', {'cached-datasets/${item}.jsonl.metadata.json': {'cache': False}}], 'metrics': []}}, 'ApplyHeuristics': {'foreach': {'id.test__bugginess/conventional_commit_regex.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/conventional_commit_regex.py'}, 'id.test__bugginess/filemetrics/all_files_test_add.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/all_files_test_add.py'}, 'id.test__bugginess/filemetrics/all_files_test_fix.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/all_files_test_fix.py'}, 'id.test__bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py'}, 'id.test__bugginess/filemetrics/buggless_if_many_lines_changed.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/buggless_if_many_lines_changed.py'}, 'id.test__bugginess/filemetrics/bugless_if_at_least_2_removed_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_2_removed_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_at_least_5_added_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_5_added_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_many_files_changes.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_many_files_changes.py'}, 'id.test__bugginess/filemetrics/bugless_if_not_code_files.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_not_code_files.py'}, 'id.test__bugginess/filemetrics/bugless_if_one_added_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_added_file.py'}, 'id.test__bugginess/filemetrics/bugless_if_one_removed_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_removed_file.py'}, 'id.test__bugginess/filemetrics/no_files_have_modified_status.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/no_files_have_modified_status.py'}, 'id.test__bugginess/filemetrics/refactoring_if_at_least_2_renamed.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/filemetrics/refactoring_if_at_least_2_renamed.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_issue_body.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_body.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_issue_label.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'}, 'id.test__bugginess/keywords/bug_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_message.py'}, 'id.test__bugginess/keywords/buggless_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/buggless_keywords_lookup_in_message.py'}, 'id.test__bugginess/keywords/bugless_keywords_lookup_in_issue_body.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py'}, 'id.test__bugginess/keywords/bugless_keywords_lookup_in_issue_label.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'}, 'id.test__bugginess/keywords/github_ref_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/github_ref_in_message.py'}, 'id.test__bugginess/keywords/init_commit_message_keywords.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/init_commit_message_keywords.py'}, 'id.test__bugginess/keywords/version_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/keywords/version_in_message.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_init.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_init.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_merge.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_merge.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_refactoring_miner.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_refactoring_miner.py'}, 'id.test__bugginess/sstubs/commit_explorer_output_sstubs.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_sstubs.py'}, 'id.test__bugginess/versionbump/contains_digit_replacement_change.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_digit_replacement_change.py'}, 'id.test__bugginess/versionbump/contains_package_lock_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_package_lock_file.py'}, 'id.test__bugginess/versionbump/contains_python_version_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_python_version_file.py'}, 'id.test__bugginess/versionbump/contains_ruby_version_file.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/contains_ruby_version_file.py'}, 'id.test__bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py'}, 'id.test__bugginess/versionbump/maven_plugin_version_bump.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/maven_plugin_version_bump.py'}, 'id.test__bugginess/versionbump/version_bump_keywords_lookup_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_bump_keywords_lookup_in_message.py'}, 'id.test__bugginess/versionbump/version_regex.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex.py'}, 'id.test__bugginess/versionbump/version_regex2.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex2.py'}, 'id.test__bugginess/versionbump/version_regex3.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/versionbump/version_regex3.py'}, 'id.test__bugginess/wip/removed_fixme.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/removed_fixme.py'}, 'id.test__bugginess/wip/removed_todo.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/removed_todo.py'}, 'id.test__bugginess/wip/wip_keyword_in_message.py': {'dataset': 'id.test', 'heuristic_group': 'bugginess/wip/wip_keyword_in_message.py'}, 'id.train__bugginess/conventional_commit_regex.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/conventional_commit_regex.py'}, 'id.train__bugginess/filemetrics/all_files_test_add.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/all_files_test_add.py'}, 'id.train__bugginess/filemetrics/all_files_test_fix.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/all_files_test_fix.py'}, 'id.train__bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bug_if_only_changed_lines_in_one_code_file.py'}, 'id.train__bugginess/filemetrics/buggless_if_many_lines_changed.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/buggless_if_many_lines_changed.py'}, 'id.train__bugginess/filemetrics/bugless_if_at_least_2_removed_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_2_removed_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_at_least_5_added_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_at_least_5_added_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_many_files_changes.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_many_files_changes.py'}, 'id.train__bugginess/filemetrics/bugless_if_not_code_files.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_not_code_files.py'}, 'id.train__bugginess/filemetrics/bugless_if_one_added_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_added_file.py'}, 'id.train__bugginess/filemetrics/bugless_if_one_removed_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/bugless_if_one_removed_file.py'}, 'id.train__bugginess/filemetrics/no_files_have_modified_status.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/no_files_have_modified_status.py'}, 'id.train__bugginess/filemetrics/refactoring_if_at_least_2_renamed.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/filemetrics/refactoring_if_at_least_2_renamed.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_issue_body.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_body.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_issue_label.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_issue_label.py'}, 'id.train__bugginess/keywords/bug_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bug_keywords_lookup_in_message.py'}, 'id.train__bugginess/keywords/buggless_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/buggless_keywords_lookup_in_message.py'}, 'id.train__bugginess/keywords/bugless_keywords_lookup_in_issue_body.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_body.py'}, 'id.train__bugginess/keywords/bugless_keywords_lookup_in_issue_label.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/bugless_keywords_lookup_in_issue_label.py'}, 'id.train__bugginess/keywords/github_ref_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/github_ref_in_message.py'}, 'id.train__bugginess/keywords/init_commit_message_keywords.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/init_commit_message_keywords.py'}, 'id.train__bugginess/keywords/version_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/keywords/version_in_message.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_init.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_init.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_merge.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_merge.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_refactoring_miner.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_refactoring_miner.py'}, 'id.train__bugginess/sstubs/commit_explorer_output_sstubs.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/sstubs/commit_explorer_output_sstubs.py'}, 'id.train__bugginess/versionbump/contains_digit_replacement_change.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_digit_replacement_change.py'}, 'id.train__bugginess/versionbump/contains_package_lock_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_package_lock_file.py'}, 'id.train__bugginess/versionbump/contains_python_version_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_python_version_file.py'}, 'id.train__bugginess/versionbump/contains_ruby_version_file.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/contains_ruby_version_file.py'}, 'id.train__bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/dependency_bump_keywords_lookup_in_message.py'}, 'id.train__bugginess/versionbump/maven_plugin_version_bump.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/maven_plugin_version_bump.py'}, 'id.train__bugginess/versionbump/version_bump_keywords_lookup_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_bump_keywords_lookup_in_message.py'}, 'id.train__bugginess/versionbump/version_regex.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex.py'}, 'id.train__bugginess/versionbump/version_regex2.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex2.py'}, 'id.train__bugginess/versionbump/version_regex3.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/versionbump/version_regex3.py'}, 'id.train__bugginess/wip/removed_fixme.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/removed_fixme.py'}, 'id.train__bugginess/wip/removed_todo.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/removed_todo.py'}, 'id.train__bugginess/wip/wip_keyword_in_message.py': {'dataset': 'id.train', 'heuristic_group': 'bugginess/wip/wip_keyword_in_message.py'}}, 'do': {'cmd': 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cloned-bohr/heuristics/${item.heuristic_group}', 'cached-datasets/${item.dataset}.jsonl'], 'outs': ['runs/__heuristics/${item.dataset}/${item.heuristic_group}/heuristic_matrix.pkl'], 'metrics': []}}}}
    """
    final_dict = {"stages": {}}
    for multi_stage in stages:
        for stage in multi_stage if isinstance(multi_stage, list) else [multi_stage]:
            name, dvc_dct = next(iter(stage.to_dvc_config_dict().items()))
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
    if heuristics_root.exists() and len(os.listdir(heuristics_root)) > 0:
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


def get_params(workspace: Workspace, fs: StorageEngine) -> Dict:
    params = {"bohr_runtime_version": workspace.bohr_runtime_version, "experiments": {}}
    heuristic_subfs = fs.heuristics_subfs()
    for exp in workspace.experiments:
        heuristic_groups = exp.heuristic_groups
        params["experiments"][exp.name] = {
            "heuristics_classifier": ":".join(
                normalize_paths(
                    heuristic_groups,
                    heuristic_subfs,
                    lambda h: HeuristicURI.from_path_and_fs(
                        h, heuristic_subfs
                    ).is_heuristic_file(),
                )
            )
            if heuristic_groups is not None
            else ".",
        }
    return params
