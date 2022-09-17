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
from bohrruntime.datamodel.bohrconfig import BohrConfig
from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment, SynteticExperiment
from bohrruntime.datamodel.task import Task
from bohrruntime.heuristics import HeuristicURI, PathTemplate
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

"""
Implements classes for each stage of pipeline and their convertion to pipeline manager config 
(DVC hardcoded as a pipeline manager for now)
"""

logger = logging.getLogger(__name__)


TASK_TEMPLATE = SimpleNamespace(name="${item.task}")
EXPERIMENT_TEMPLATE = SimpleNamespace(name="${item.exp}", task=TASK_TEMPLATE)
DATASET_TEMPLATE = SimpleNamespace(id="${item.dataset}")


def iterate_workspace(
    workspace: BohrConfig,
    storage_engine: StorageEngine,
    iterate_heuristics: bool = True,
) -> Union[List[Tuple[Experiment, Dataset]], List[Tuple[Experiment, Dataset, str]]]:
    """
    >>> from bohrruntime.testtools import get_stub_experiment, StubHeuristicLoader, get_stub_storage_engine
    >>> workspace = BohrConfig('0.0.1', [get_stub_experiment(no_training_dataset=True)])
    >>> storage_engine = get_stub_storage_engine()
    >>> (experiment1, dataset1, heuristic_group1), (experiment2, dataset2, heuristic_group2) = iterate_workspace(workspace, storage_engine)
    >>> experiment1 is experiment2
    True
    >>> experiment1
    Experiment(name='stub-exp', task=LabelingTask(name='stub-task', author='stub-author', description='stub-description', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, test_datasets={Dataset(id='stub-dataset', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, query=None, projection=None, n_datapoints=None, path=None): None}, labels=(CommitLabel.NonBugFix, CommitLabel.BugFix), class_balance=None), train_dataset=None, heuristics_classifier=None, extra_test_datasets=frozendict.frozendict({}))
    >>> dataset1 is dataset2
    True
    >>> dataset1
    Dataset(id='stub-dataset', heuristic_input_artifact_type=<class 'bohrruntime.testtools.StubArtifact'>, query=None, projection=None, n_datapoints=None, path=None)
    >>> heuristic_group1
    '/heuristic1'
    >>> heuristic_group2
    '/heuristic2'
    >>> (experiment3, dataset3, heuristic_group3), = iterate_workspace(workspace, storage_engine, iterate_heuristics=False)
    >>> experiment3 is experiment1
    True
    >>> dataset3 is dataset1
    True
    >>> heuristic_group3 is None
    True
    """
    heuristic_loader = storage_engine.get_heuristic_loader()
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

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def get_substage_names(self) -> List[str]:
        pass


@dataclass(repr=False)
class SimpleStage(Stage):
    storage_engine: StorageEngine

    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def dvc_stage_name(self) -> str:
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
            self.dvc_stage_name(): {
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
        return type(self).__name__[: -len("Stage")]

    @abstractmethod
    def n_substages(self) -> int:
        pass

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
class Substage(SimpleStage):
    def n_substages(self) -> int:
        return 1

    @abstractmethod
    def get_stage_parameter(self) -> str:
        pass

    def get_substage_names(self) -> List[str]:
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    def dvc_stage_name(self) -> str:
        return f"{self.stage_name()}#{self.get_stage_parameter()}"


@dataclass(repr=False)
class CompoundStage(Stage):
    stages: List[Substage]

    def __post_init__(self):
        for stage in self.stages:
            if not isinstance(stage, Substage):
                raise ValueError(f"Only sub-stages are allowed in compound stages, found: {type(stage)}")

    def get_substages(self) -> List[Substage]:
        return self.stages

    def summary(self) -> str:
        return self.stages[0].stage_name()

    def get_substage_names(self) -> List[str]:
        return [f"{stage.dvc_stage_name()}" for stage in self.stages]

    def __str__(self):
        return ", ".join([str(stage.to_dvc_config_dict()) for stage in self.stages])

    def __repr__(self):
        return ", ".join([str(stage.to_dvc_config_dict()) for stage in self.stages])


@dataclass(repr=False)
class TemplateStage(SimpleStage):
    workspace: BohrConfig

    def dvc_stage_name(self) -> str:
        return self.stage_name()

    def n_substages(self) -> int:
        return len(self.get_iterating_over())

    def get_for_each(self) -> Dict[str, Dict[str, Any]]:
        foreach: Dict[str, Dict[str, Any]] = {}
        for entry in self.get_iterating_over():
            foreach = {
                **foreach,
                **self.generate_for_each_entry(entry),
            }
        return foreach

    def get_substage_names(self) -> List[str]:
        stage = self.stage_name()
        return [f"{stage}@{each}" for each in self.get_for_each()]

    def to_dvc_config_dict(
        self,
    ) -> Dict:
        parent = next(iter(super(TemplateStage, self).to_dvc_config_dict().values()))
        foreach = self.get_for_each()
        dct = {
            self.dvc_stage_name(): {
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
                    self.workspace, self.storage_engine, False
                )
            },
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


class LoadDatasetsStage(TemplateStage):
    def execute(self) -> None:
        for dataset in self.get_iterating_over():
            load_dataset(dataset, self.storage_engine)

    def get_iterating_over(self) -> Sequence:
        return sorted({d for exp in self.workspace.experiments for d in exp.datasets})

    def get_for_each(self) -> List[str]:
        return [d.id for d in self.get_iterating_over()]

    def get_cmd(self) -> str:
        return 'bohr-internal load-dataset "${item}"'

    def get_outs(self) -> List[str]:
        outs = [
            self.storage_engine.path_structure.dataset("${item}"),
            {
                self.storage_engine.path_structure.dataset_metadata("${item}"): {
                    "cache": False
                }
            },
        ]
        return outs

    def is_always_changed(self):
        for dataset in self.get_iterating_over():
            if dataset.path is not None:
                return True  # TODO here if at least one dataset is local all the load stages will be reprodices - fix
        return False


class ApplyHeuristicsStage(TemplateStage):
    def execute(self) -> None:
        for dataset, heuristic_group in self.get_iterating_over():
            apply_heuristics_to_dataset(
                SnorkelHeuristicApplier(), heuristic_group, dataset, self.storage_engine
            )

    def get_cmd(self) -> str:
        cmd = 'bohr-internal apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"'
        return cmd

    def get_deps(self) -> List[str]:
        deps = [
            self.storage_engine.path_structure.heuristic_group(
                PathTemplate("${item.heuristic_group}")
            ),
            self.storage_engine.path_structure.dataset(DATASET_TEMPLATE.id),
        ]
        return deps

    def get_outs(self) -> List[Any]:
        outs = [
            self.storage_engine.heuristic_matrix_file(
                DATASET_TEMPLATE, PathTemplate("${item.heuristic_group}")
            )
        ]
        return outs

    def get_iterating_over(self) -> Sequence:
        return sorted(
            {
                (d, h)
                for _, d, h in iterate_workspace(
                    self.workspace, self.storage_engine, True
                )
            },
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
class ComputeSingleHeuristicMetricsStage(Substage):
    task: Task

    def get_stage_parameter(self) -> str:
        return self.task.name

    def execute(self) -> None:
        calculate_single_heuristic_metrics(self.task, self.storage_engine)

    def get_cmd(self) -> str:
        return f"bohr-internal compute-single-heuristic-metric {self.task.name}"

    def get_deps(self) -> List[str]:
        deps = []
        heuristic_loader = self.storage_engine.get_heuristic_loader()
        for dataset in self.task.test_datasets:
            deps.append(self.storage_engine.path_structure.dataset(dataset.id))
        for heuristic_group in heuristic_loader.get_heuristic_uris(
            input_artifact_type=self.task.heuristic_input_artifact_type
        ):
            deps.append(
                self.storage_engine.path_structure.heuristic_group(heuristic_group)
            )
            for dataset in self.task.test_datasets:
                deps.append(
                    self.storage_engine.heuristic_matrix_file(dataset, heuristic_group)
                )
        return deps

    def get_outs(self) -> List[str]:
        outputs = []
        heuristic_loader = self.storage_engine.get_heuristic_loader()
        for dataset in self.task.test_datasets:
            heuristic_groups = heuristic_loader.get_heuristic_uris(
                input_artifact_type=self.task.heuristic_input_artifact_type
            )
            for heuristic_group in heuristic_groups:
                outputs.append(
                    {
                        self.storage_engine.single_heuristic_metrics(
                            self.task, dataset, heuristic_group
                        ): {"cache": False}
                    }
                )
        return outputs


class FetchMultipleHeuristicOutputsStage(TemplateStage):
    def execute(self) -> None:
        for experiment, dataset in self.get_iterating_over():
            combine_applied_heuristics(experiment, dataset, self.storage_engine)

    def get_cmd(self) -> str:
        return 'bohr-internal combine-heuristics "${item.exp}" --dataset "${item.dataset}"'

    def get_deps(self) -> List[str]:
        return [
            self.storage_engine.path_structure.heuristic_dataset_dir(DATASET_TEMPLATE)
        ]

    def get_params(self) -> List:
        return [{"bohr.lock": ["experiments.${item.exp}.heuristics_classifier"]}]

    def get_outs(self) -> List[str]:
        outs = [
            self.storage_engine.path_structure.experiment_label_matrix_file(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            )
        ]
        return outs


class CalculateMetricsStage(TemplateStage):
    def execute(self) -> None:
        for experiment, dataset in self.get_iterating_over():
            calculate_experiment_metrics(experiment, dataset, self.storage_engine)

    def get_metrics(self) -> List:
        metrics = [
            str(
                self.storage_engine.path_structure.experiment_metrics(
                    EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
                )
            )
        ]
        return metrics

    def get_cmd(self) -> str:
        return 'bohr-internal run-metrics-and-analysis "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            self.storage_engine.path_structure.experiment_label_matrix_file(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            ),
            self.storage_engine.path_structure.label_model(EXPERIMENT_TEMPLATE),
            self.storage_engine.path_structure.dataset(DATASET_TEMPLATE.id),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            {
                self.storage_engine.path_structure.analysis_json(
                    EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
                ): {"cache": False}
            },
            {
                self.storage_engine.path_structure.analysis_csv(
                    EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
                ): {"cache": False}
            },
        ]
        return outs


class ComputePredefinedModelMetricsStage(TemplateStage):
    def execute(self) -> None:
        for task, dataset in self.get_iterating_over():
            exp = SynteticExperiment("random_model", task, type="random")
            calculate_experiment_metrics(exp, dataset, self.storage_engine)

    def get_model_name(self) -> str:
        return f"{self.get_model_type()}_model"

    @abstractmethod
    def get_model_type(self) -> str:
        pass

    def get_metrics(self) -> List:
        metrics = [
            str(
                self.storage_engine.path_structure.experiment_metrics(
                    SimpleNamespace(name=self.get_model_name(), task=TASK_TEMPLATE),
                    DATASET_TEMPLATE,
                )
            )
        ]
        return metrics

    def get_deps(self) -> List:
        return [self.storage_engine.path_structure.dataset(DATASET_TEMPLATE.id)]

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
            "bohr-internal compute-"
            + self.get_model_type()
            + '-model-metrics "${item.task}" "${item.dataset}"'
        )


class ComputeRandomModelMetricsStage(ComputePredefinedModelMetricsStage):
    def get_model_type(self) -> str:
        return "random"


class ComputeZeroModelMetricsStage(ComputePredefinedModelMetricsStage):
    def get_model_type(self) -> str:
        return "zero"


@dataclass(repr=False)
class TrainModelStage(Substage):
    exp: Experiment

    def get_stage_parameter(self) -> str:
        return self.exp.name

    def execute(self) -> None:
        pass

    def get_cmd(self):
        return f"bohr-internal train-model {self.exp.name}"

    def get_deps(self) -> List[str]:
        deps = [
            self.storage_engine.path_structure.experiment_label_matrix_file(
                self.exp, self.exp.train_dataset
            )
        ]
        return deps

    def get_outs(self) -> List[str]:
        outs = [
            self.storage_engine.path_structure.label_model(self.exp),
            self.storage_engine.path_structure.label_model_weights(self.exp),
        ]
        return outs


class PrepareDatasetStage(TemplateStage):
    def execute(self) -> None:
        for experiment, dataset in tqdm(self.get_iterating_over()):
            prepare_dataset(experiment, dataset, self.storage_engine)

    def get_cmd(self) -> str:
        return 'bohr-internal prepare-dataset "${item.exp}" "${item.dataset}"'

    def get_deps(self) -> List:
        deps = [
            self.storage_engine.path_structure.experiment_label_matrix_file(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            ),
            self.storage_engine.path_structure.label_model(EXPERIMENT_TEMPLATE),
        ]
        return deps

    def get_outs(self) -> List:
        outs = [
            self.storage_engine.path_structure.labeled_dataset(
                EXPERIMENT_TEMPLATE, DATASET_TEMPLATE
            )
        ]
        return outs


def get_stage_list(
    workspace: BohrConfig, storage_engine: StorageEngine
) -> List[Stage]:
    """
    >>> from bohrlabels.core import Label, LabelSet
    >>> from enum import auto
    >>> from bohrruntime.tasktypes.labeling.core import LabelingTask
    >>> from bohrruntime.heuristics import add_template_heuristic
    >>> from bohrruntime.testtools import StubHeuristicLoader, get_stub_storage_engine
    >>> from frozendict import frozendict
    >>> class TestLabel(Label): Yes = auto(); No = auto()
    >>> train = Dataset("id.train", Commit)
    >>> test = Dataset("id.test", Commit)
    >>> labels = (LabelSet.of(TestLabel.Yes), LabelSet.of(TestLabel.No))
    >>> task = LabelingTask("name", "author", "desc", Commit, frozendict({test: lambda x:x}), labels)
    >>> get_stage_list(BohrConfig('0.x.x', [Experiment('exp', task, train, '/')]), get_stub_storage_engine())
    [{'LoadDatasets': {'foreach': ['id.test', 'id.train'], 'do': {'cmd': 'bohr-internal load-dataset "${item}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': [], 'outs': ['cached-datasets/${item}.jsonl', {'cached-datasets/${item}.jsonl.metadata.json': {'cache': False}}], 'metrics': [], 'always_changed': False}}}, {'ApplyHeuristics': {'foreach': {'id.test__/heuristic1': {'dataset': 'id.test', 'heuristic_group': '/heuristic1'}, 'id.test__/heuristic2': {'dataset': 'id.test', 'heuristic_group': '/heuristic2'}, 'id.train__/heuristic1': {'dataset': 'id.train', 'heuristic_group': '/heuristic1'}, 'id.train__/heuristic2': {'dataset': 'id.train', 'heuristic_group': '/heuristic2'}}, 'do': {'cmd': 'bohr-internal apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cloned-bohr/heuristics/${item.heuristic_group}', 'cached-datasets/${item.dataset}.jsonl'], 'outs': ['runs/__heuristics/${item.dataset}/${item.heuristic_group}/heuristic_matrix.pkl'], 'metrics': [], 'always_changed': False}}}, {'ComputeSingleHeuristicMetrics#name': {'cmd': 'bohr-internal compute-single-heuristic-metric name', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/id.test.jsonl', 'cloned-bohr/heuristics//heuristic1', 'runs/__heuristics/id.test//heuristic1/heuristic_matrix.pkl', 'cloned-bohr/heuristics//heuristic2', 'runs/__heuristics/id.test//heuristic2/heuristic_matrix.pkl'], 'outs': [{'runs/__single_heuristic_metrics/name/id.test//heuristic1/metrics.txt': {'cache': False}}, {'runs/__single_heuristic_metrics/name/id.test//heuristic2/metrics.txt': {'cache': False}}], 'metrics': [], 'always_changed': False}}, {'FetchMultipleHeuristicOutputs': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr-internal combine-heuristics "${item.exp}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['experiments.${item.exp}.heuristics_classifier']}, {'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/__heuristics/${item.dataset}'], 'outs': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl'], 'metrics': [], 'always_changed': False}}}, {'TrainModel#exp': {'cmd': 'bohr-internal train-model exp', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/name/exp/id.train/heuristic_matrix.pkl'], 'outs': ['runs/name/exp/label_model.pkl', 'runs/name/exp/label_model_weights.csv'], 'metrics': [], 'always_changed': False}}, {'PrepareDataset': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr-internal prepare-dataset "${item.exp}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl', 'runs/${item.task}/${item.exp}/label_model.pkl'], 'outs': ['runs/${item.task}/${item.exp}/${item.dataset}/labeled.csv'], 'metrics': [], 'always_changed': False}}}, {'ComputeRandomModelMetrics': {'foreach': {'name__id.test': {'dataset': 'id.test', 'task': 'name'}}, 'do': {'cmd': 'bohr-internal compute-random-model-metrics "${item.task}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/${item.dataset}.jsonl'], 'outs': [], 'metrics': [{'runs/${item.task}/random_model/${item.dataset}/metrics.txt': {'cache': False}}], 'always_changed': False}}}, {'ComputeZeroModelMetrics': {'foreach': {'name__id.test': {'dataset': 'id.test', 'task': 'name'}}, 'do': {'cmd': 'bohr-internal compute-zero-model-metrics "${item.task}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cached-datasets/${item.dataset}.jsonl'], 'outs': [], 'metrics': [{'runs/${item.task}/zero_model/${item.dataset}/metrics.txt': {'cache': False}}], 'always_changed': False}}}, {'CalculateMetrics': {'foreach': {'exp__id.test': {'dataset': 'id.test', 'exp': 'exp', 'task': 'name'}, 'exp__id.train': {'dataset': 'id.train', 'exp': 'exp', 'task': 'name'}}, 'do': {'cmd': 'bohr-internal run-metrics-and-analysis "${item.exp}" "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['runs/${item.task}/${item.exp}/${item.dataset}/heuristic_matrix.pkl', 'runs/${item.task}/${item.exp}/label_model.pkl', 'cached-datasets/${item.dataset}.jsonl'], 'outs': [{'runs/${item.task}/${item.exp}/${item.dataset}/analysis.json': {'cache': False}}, {'runs/${item.task}/${item.exp}/${item.dataset}/analysis.csv': {'cache': False}}], 'metrics': [{'runs/${item.task}/${item.exp}/${item.dataset}/metrics.txt': {'cache': False}}], 'always_changed': False}}}]
    """
    if len(workspace.experiments) == 0:
        raise ValueError("At least of task should be specified")

    all_tasks = sorted(
        {exp.task for exp in workspace.experiments}, key=lambda t: t.name
    )

    stages: List[Stage] = [
        LoadDatasetsStage(storage_engine, workspace),
        ApplyHeuristicsStage(storage_engine, workspace),
        CompoundStage([
            ComputeSingleHeuristicMetricsStage(storage_engine, task)
            for task in all_tasks
        ]),
        FetchMultipleHeuristicOutputsStage(storage_engine, workspace),
        CompoundStage([
            TrainModelStage(storage_engine, exp)
            for exp in sorted(workspace.experiments, key=lambda x: x.name)
        ]),
        PrepareDatasetStage(storage_engine, workspace),
        ComputeRandomModelMetricsStage(storage_engine, workspace),
        ComputeZeroModelMetricsStage(
            storage_engine, workspace
        ),  # TODO compute zero and random metric could be one stage (baselines?)
        CalculateMetricsStage(storage_engine, workspace),
    ]
    return stages


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


def get_params(workspace: BohrConfig, storage_engine: StorageEngine) -> Dict:
    params = {"bohr_runtime_version": workspace.bohr_runtime_version, "experiments": {}}
    heuristic_subfs = storage_engine.heuristics_subfs()
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
