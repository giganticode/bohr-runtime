import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from bohr.config import Config, load_config
from bohr.datamodel import Dataset, Task
from bohr.pathconfig import PathConfig

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

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(), task=self.task, path_config=self.path_config
        )

    def to_string(self) -> List[str]:
        env = Environment(
            loader=FileSystemLoader(Path(__file__).parent.parent),
            undefined=StrictUndefined,
        )
        template = env.get_template(f"resources/dvc_command_templates/{self.template}")
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
    def __init__(self, path_config: PathConfig, execute_immediately: bool = False):
        super().__init__(
            "parse_labels.template", path_config, None, execute_immediately
        )

    def summary(self) -> str:
        return "parse labels"

    def get_name(self) -> str:
        return "parse_labels"


class ApplyHeuristicsCommand(DvcCommand):
    def __init__(
        self,
        path_config: PathConfig,
        task: Task,
        heuristic_group: str,
        dataset: str,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "apply_heuristics.template", path_config, task, execute_immediately
        )
        self.heuristic_group = heuristic_group
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            stage_name=self.get_name(),
            task=self.task,
            path_config=self.path_config,
            heuristic_group=self.heuristic_group,
            dataset=self.dataset,
        )

    def summary(self) -> str:
        return f"[{self.task.name}] apply heuristics (group: {self.heuristic_group}) to {self.dataset}"

    def get_name(self) -> str:
        return f"{self.task.name}_apply_heuristics__{self.heuristic_group.replace('.', '_')}__{self.dataset.replace('.', '_')}"


class CombineHeuristicsCommand(DvcCommand):
    def __init__(
        self, path_config: PathConfig, task: Task, execute_immediately: bool = False
    ):
        super().__init__(
            "combine_heuristics.template", path_config, task, execute_immediately
        )

    def summary(self) -> str:
        return f"[{self.task.name}] combine heuristics"

    def get_name(self) -> str:
        return f"{self.task.name}_combine_heuristics"


class TrainLabelModelCommand(DvcCommand):
    def __init__(
        self, path_config: PathConfig, task: Task, execute_immediately: bool = False
    ):
        super().__init__(
            "train_label_model.template", path_config, task, execute_immediately
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
    def __init__(
        self,
        path_config: PathConfig,
        task: Task,
        dataset: str,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "label_dataset.template", path_config, task, execute_immediately
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
    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "preprocess_copy.template",
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
    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "preprocess_7z.template",
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
    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "preprocess_shell.template", path_config, None, execute_immediately
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
        path_to_template: Path,
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


def create_directories_if_necessary(config: Config) -> None:
    path_config = config.paths
    for task in config.tasks.values():
        for heuristic_group in task.heuristic_groups:
            (path_config.generated / task.name / heuristic_group).mkdir(
                exist_ok=True, parents=True
            )
            (path_config.metrics / task.name / heuristic_group).mkdir(
                exist_ok=True, parents=True
            )
    path_config.labeled_data.mkdir(exist_ok=True, parents=True)


def save_transient_stages_to_config(
    transient_stages: List[str], path_config: PathConfig
) -> None:
    conf_dir = path_config.project_root / ".bohr"
    if not conf_dir.exists():
        conf_dir.mkdir()
    transient_stages_file = conf_dir / "transient_stages.json"
    with transient_stages_file.open("w") as f:
        json.dump(transient_stages, f)


def load_transient_stages(path_config: PathConfig) -> List[str]:
    transient_stages_file = path_config.project_root / ".bohr" / "transient_stages.json"
    if not transient_stages_file.exists():
        return []
    with transient_stages_file.open() as f:
        return json.load(f)


def add_all_tasks_to_dvc_pipeline(config: Config) -> None:
    path_config = config.paths
    create_directories_if_necessary(config)
    all_tasks = sorted(config.tasks.values(), key=lambda x: x.name)
    logger.info(
        f"Following tasks are added to the pipeline: {list(map(lambda x: x.name, all_tasks))}"
    )
    transient_stages = []
    commands: List[DvcCommand] = []
    for dataset_name, dataset in config.datasets.items():
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
            for dataset_name in task.datasets:
                commands.append(
                    ApplyHeuristicsCommand(
                        path_config, task, heuristic_group, dataset_name
                    )
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
    config = load_config()
    add_all_tasks_to_dvc_pipeline(config)
