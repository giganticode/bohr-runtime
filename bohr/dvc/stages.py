import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template
from pandas.io.formats.style import jinja2

from bohr.config.pathconfig import PathConfig
from bohr.datamodel.bohrrepo import BohrRepo, load_bohr_repo
from bohr.datamodel.dataset import Dataset
from bohr.datamodel.task import Task
from bohr.util.paths import AbsolutePath

logger = logging.getLogger(__name__)


class DvcCommand(ABC):
    def __init__(
        self,
        template: str,
        path_config: PathConfig,
        task: Task,
        execute_immediately: bool = False,
        transient_stage: bool = False,
    ):
        self.template = template
        self.path_config = path_config
        self.task = task
        self.execute_immediately = execute_immediately
        self.transient_stage = transient_stage

    def render_stage_template(self, template: Template) -> str:
        return template.render(
            stage_name=self.get_name(), task=self.task, path_config=self.path_config
        )

    def to_string(self) -> List[str]:
        template = jinja2.Environment().from_string(self.template)
        command = self.render_stage_template(template)
        if_exec = "" if self.execute_immediately else "--no-exec "
        command = f"dvc run -v {if_exec}--force {command}"
        command_array = list(filter(None, re.split("[\n ]", command)))
        return command_array

    @abstractmethod
    def summary(self) -> str:
        pass

    def run(self) -> subprocess.CompletedProcess:
        command = self.to_string()
        logger.debug(f"Adding stage: {self.summary()}")
        return subprocess.run(
            command, cwd=self.path_config.project_root, capture_output=True
        )

    @abstractmethod
    def get_name(self) -> str:
        pass


class ParseLabelsCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
        -d {{path_config.labels}}
        -O labels.py
        bohr porcelain parse-labels
    """

    def __init__(self, path_config: PathConfig, execute_immediately: bool = False):
        super().__init__(
            ParseLabelsCommand.TEMPLATE, path_config, None, execute_immediately
        )

    def summary(self) -> str:
        return "parse labels"

    def get_name(self) -> str:
        return "parse_labels"


class ApplyHeuristicsCommand(DvcCommand):

    TEMPLATE = """
        -n {{stage_name}}
        
        -d labels.py
        
        -d {{heuristic_group|replace('.', '/')}}.py
        {% for dataset in datasets %}
        -d {{task.all_affected_datasets[dataset].path_preprocessed}}
        {% endfor %}
        
        -p bohr.json:bohr_framework_version
        
        -o {{path_config.generated_dir}}/{{task.name}}/{{heuristic_group}}/heuristic_matrix_{{datasets[0]}}.pkl
        -M {{path_config.metrics_dir}}/{{task.name}}/{{heuristic_group}}/heuristic_metrics_{{datasets[0]}}.json
        
        bohr porcelain apply-heuristics {{task.name}} --heuristic-group {{heuristic_group}} --dataset {{datasets[0]}}
    """

    def __init__(
        self,
        path_config: PathConfig,
        task: Task,
        heuristic_group: str,
        datasets: List[str],
        execute_immediately: bool = False,
    ):
        super().__init__(
            ApplyHeuristicsCommand.TEMPLATE, path_config, task, execute_immediately
        )
        self.heuristic_group = heuristic_group
        self.datasets = datasets

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(),
            task=self.task,
            path_config=self.path_config,
            heuristic_group=self.heuristic_group,
            datasets=self.datasets,
        )

    def summary(self) -> str:
        return f"[{self.task.name}] apply heuristics (group: {self.heuristic_group}) to {self.datasets[0]}"

    def get_name(self) -> str:
        return f"{self.task.name}_apply_heuristics__{self.heuristic_group.replace('.', '_')}__{self.datasets[0].replace('.', '_')}"


class CombineHeuristicsCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
    
        {% for heuristic_group in task.heuristic_groups %}
        {% for dataset in task.datasets %} -d {{path_config.generated_dir}}/{{task.name}}/{{heuristic_group}}/heuristic_matrix_{{dataset}}.pkl {% endfor %}
        {% endfor %}
        
        -p bohr.json:bohr_framework_version
        
        {% for dataset in task.datasets %} -o {{path_config.generated_dir}}/{{task.name}}/heuristic_matrix_{{dataset}}.pkl {% endfor %}
        {% for dataset in task.datasets %} -O {{path_config.generated_dir}}/{{task.name}}/analysis_{{dataset}}.csv{% endfor %}
        
        {% for dataset in task.datasets %} -M {{path_config.metrics_dir}}/{{task.name}}/analysis_{{dataset}}.json{% endfor %}
        {% for dataset in task.datasets %} -M {{path_config.metrics_dir}}/{{task.name}}/heuristic_metrics_{{dataset}}.json{% endfor %}
        
        bohr porcelain apply-heuristics {{task.name}}`
    """

    def __init__(
        self, path_config: PathConfig, task: Task, execute_immediately: bool = False
    ):
        super().__init__(
            CombineHeuristicsCommand.TEMPLATE, path_config, task, execute_immediately
        )

    def summary(self) -> str:
        return f"[{self.task.name}] combine heuristics"

    def get_name(self) -> str:
        return f"{self.task.name}_combine_heuristics"


class TrainLabelModelCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
        
        -d {{path_config.generated_dir}}/{{task.name}}/heuristic_matrix_{{target_dataset}}.pkl
        {% for test_dataset in task.test_datasets %} -d {{path_config.generated_dir}}/{{task.name}}/heuristic_matrix_{{test_dataset}}.pkl {% endfor %}
        {% for datapath in task.test_datapaths %} -d {{datapath}}{% endfor %}
        
        -p bohr.json:bohr_framework_version
        
        -o {{path_config.generated_dir}}/{{task.name}}/label_model.pkl
        -O {{path_config.generated_dir}}/{{task.name}}/label_model_weights.csv
        
        -M {{path_config.metrics_dir}}/{{task.name}}/label_model_metrics.json
        
        bohr porcelain train-label-model {{task.name}} {{target_dataset}}
    """

    def __init__(
        self, path_config: PathConfig, task: Task, execute_immediately: bool = False
    ):
        super().__init__(
            TrainLabelModelCommand.TEMPLATE, path_config, task, execute_immediately
        )

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(),
            task=self.task,
            path_config=self.path_config,
            target_dataset=next(iter(self.task.train_datasets.keys())),
        )

    def summary(self) -> str:
        return f"[{self.task.name}] train label model on {next(iter(self.task.train_datasets.keys()))} dataset"

    def get_name(self) -> str:
        return f"{self.task.name}_train_label_model"


class LabelDatasetCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
    
        -d {{path_config.generated_dir}}/{{task.name}}/heuristic_matrix_{{dataset}}.pkl
        -d {{path_config.generated_dir}}/{{task.name}}/label_model.pkl
        -d {{task.datasets[dataset].path_preprocessed}}
        
        -p bohr.json:bohr_framework_version
        
        -o {{path_config.labeled_data_dir}}/{{dataset}}.labeled.csv
        
        bohr porcelain label-dataset {{task.name}} {{dataset}}
    """

    def __init__(
        self,
        path_config: PathConfig,
        task: Task,
        dataset: str,
        execute_immediately: bool = False,
    ):
        super().__init__(
            LabelDatasetCommand.TEMPLATE, path_config, task, execute_immediately
        )
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(),
            task=self.task,
            path_config=self.path_config,
            dataset=self.dataset,
        )

    def summary(self) -> str:
        return f"[{self.task.name}] label dataset: {self.dataset}"

    def get_name(self) -> str:
        return f"{self.task.name}_label_dataset_{self.dataset.replace('.', '_')}"


class PreprocessCopyCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
    
        -d {{dataset.path_dist}}
        -O {{dataset.path_preprocessed}}
        
        cp {{dataset.path_dist}} {{data_dir}} &&
        echo "{{dataset.path_preprocessed}}" >> .gitignore &&
        git add .gitignore

    """

    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            PreprocessCopyCommand.TEMPLATE,
            path_config,
            None,
            execute_immediately,
            transient_stage=True,
        )
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(),
            dataset=self.dataset,
            data_dir=self.path_config.data_dir,
        )

    def summary(self) -> str:
        return f"Pre-processing (copying): {self.dataset.name}"

    def get_name(self) -> str:
        return f"preprocess_{self.dataset.name}"


class Preprocess7zCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
        
        -d {{dataset.path_dist}}
        -O {{dataset.path_preprocessed}}
        
        7z x {{dataset.path_dist}} -o{{data_dir}} &&
        echo "{{dataset.path_preprocessed}}" >> .gitignore &&
        git add .gitignore
    """

    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            Preprocess7zCommand.TEMPLATE,
            path_config,
            None,
            execute_immediately,
            transient_stage=True,
        )
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(),
            dataset=self.dataset,
            data_dir=self.path_config.data_dir,
        )

    def summary(self) -> str:
        return f"Pre-processing (extracting): {self.dataset.name}"

    def get_name(self) -> str:
        return f"preprocess_{self.dataset.name}"


class PreprocessShellCommand(DvcCommand):
    TEMPLATE = """
        -n {{stage_name}}
    
        -d {{dataset.path_dist}}
        -d {{dataset.preprocessor}}
        -o {{dataset.path_preprocessed}}
        
        {{dataset.preprocessor}}
    """

    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            PreprocessShellCommand.TEMPLATE, path_config, None, execute_immediately
        )
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(stage_name=self.get_name(), dataset=self.dataset)

    def summary(self) -> str:
        return f"Pre-processing (shell): {self.dataset.name}"

    def get_name(self) -> str:
        return f"preprocess_{self.dataset.name}"


class ManualCommand(DvcCommand):
    def __init__(
        self,
        path_config: PathConfig,
        path_to_template: AbsolutePath,
        execute_immediately: bool = False,
    ):
        super().__init__("empty.template", path_config, None, execute_immediately)
        self.path_to_template = path_to_template

    def render_stage_template(self, template) -> str:
        with open(self.path_to_template) as f:
            return f.read()

    def summary(self) -> str:
        return f"Command from: {self.path_to_template}"

    def get_name(self) -> str:
        return "manual_command"

    def get_template(self) -> str:
        return ApplyHeuristicsCommand.TEMPLATE


def create_directories_if_necessary(bohr_repo: Optional[BohrRepo] = None) -> None:
    bohr_repo = bohr_repo or load_bohr_repo()
    path_config = PathConfig.load()
    for task in bohr_repo.tasks.values():
        for heuristic_group in task.heuristic_groups:
            (path_config.generated / task.name / heuristic_group).mkdir(
                exist_ok=True, parents=True
            )
            (path_config.metrics / task.name / heuristic_group).mkdir(
                exist_ok=True, parents=True
            )
    path_config.labeled_data.mkdir(exist_ok=True, parents=True)


def save_transient_stages_to_config(
    transient_stages: List[str], path_config: Optional[PathConfig] = None
) -> None:
    path_config = path_config or PathConfig.load()
    conf_dir = path_config.project_root / ".bohr"
    if not conf_dir.exists():
        conf_dir.mkdir()
    transient_stages_file = conf_dir / "transient_stages.json"
    with transient_stages_file.open("w") as f:
        json.dump(transient_stages, f)


def load_transient_stages(path_config: Optional[PathConfig] = None) -> List[str]:
    path_config = path_config or PathConfig.load()
    transient_stages_file = path_config.project_root / ".bohr" / "transient_stages.json"
    if not transient_stages_file.exists():
        return []
    with transient_stages_file.open() as f:
        return json.load(f)


def add_all_tasks_to_dvc_pipeline(
    bohr_repo: Optional[BohrRepo] = None, path_config: Optional[PathConfig] = None
) -> None:
    path_config = path_config or PathConfig.load()
    bohr_repo = bohr_repo or load_bohr_repo(path_config.project_root)

    create_directories_if_necessary(bohr_repo)
    all_tasks = sorted(bohr_repo.tasks.values(), key=lambda x: x.name)
    logger.info(
        f"Following tasks are added to the pipeline: {list(map(lambda x: x.name, all_tasks))}"
    )

    all_keys = set()
    for keys in map(lambda t: t.datasets.keys(), all_tasks):
        all_keys.update(keys)
    all_datasets_used_in_tasks = list(
        map(lambda key: bohr_repo.datasets[key], all_keys)
    )
    logger.info(f"Datasets used in tasks:")
    for dataset in all_datasets_used_in_tasks:
        linked_datasets = dataset.get_linked_datasets()
        logger.info(
            f"{dataset.name} {'-> ' + str(list(map(lambda d: d.name, linked_datasets))) if linked_datasets else ''}"
        )
    transient_stages = []
    commands: List[DvcCommand] = []
    for dataset_name, dataset in bohr_repo.datasets.items():
        if dataset.preprocessor == "copy":
            copy_command = PreprocessCopyCommand(path_config, dataset)
            commands.append(copy_command)
            transient_stages.append(copy_command.get_name())
        elif dataset.preprocessor == "7z":
            extract_command = Preprocess7zCommand(path_config, dataset)
            commands.append(extract_command)
            transient_stages.append(extract_command.get_name())
        else:
            commands.append(PreprocessShellCommand(path_config, dataset))
    commands.append(ParseLabelsCommand(path_config))
    for task in all_tasks:
        for heuristic_group in task.heuristic_groups:
            for dataset_name, dataset in task.datasets.items():
                datasets = [dataset_name] + list(
                    map(lambda d: d.name, dataset.get_linked_datasets())
                )
                commands.append(
                    ApplyHeuristicsCommand(path_config, task, heuristic_group, datasets)
                )
        commands.append(CombineHeuristicsCommand(path_config, task))
        commands.append(TrainLabelModelCommand(path_config, task))
        for dataset_name in task.datasets:
            commands.append(LabelDatasetCommand(path_config, task, dataset_name))
    if path_config.manual_stages.exists():
        root, dirs, files = next(os.walk(path_config.manual_stages))
        for file in files:
            commands.append(ManualCommand(path_config, Path(root) / file))
    for command in commands:
        completed_process = command.run()
        if completed_process.returncode != 0:
            print(completed_process.stderr.decode())
            break
    save_transient_stages_to_config(transient_stages, path_config)


if __name__ == "__main__":
    add_all_tasks_to_dvc_pipeline()
