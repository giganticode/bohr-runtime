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
    ):
        self.template = template
        self.path_config = path_config
        self.task = task
        self.execute_immediately = execute_immediately

    def render_stage_template(self, template) -> str:
        return template.render(task=self.task, path_config=self.path_config)

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


class ParseLabelsCommand(DvcCommand):
    def __init__(self, path_config: PathConfig, execute_immediately: bool = False):
        super().__init__(
            "parse_labels.template", path_config, None, execute_immediately
        )

    def summary(self) -> str:
        return "parse labels"


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
            task=self.task,
            path_config=self.path_config,
            heuristic_group=self.heuristic_group,
            dataset=self.dataset,
        )

    def summary(self) -> str:
        return f"[{self.task.name}] apply heuristics (group: {self.heuristic_group}) to {self.dataset}"


class CombineHeuristicsCommand(DvcCommand):
    def __init__(
        self, path_config: PathConfig, task: Task, execute_immediately: bool = False
    ):
        super().__init__(
            "combine_heuristics.template", path_config, task, execute_immediately
        )

    def summary(self) -> str:
        return f"[{self.task.name}] combine heuristics"


class TrainLabelModelCommand(DvcCommand):
    def __init__(
        self, path_config: PathConfig, task: Task, execute_immediately: bool = False
    ):
        super().__init__(
            "train_label_model.template", path_config, task, execute_immediately
        )

    def render_stage_template(self, template) -> str:
        return template.render(
            task=self.task,
            path_config=self.path_config,
            target_dataset=next(iter(self.task.train_datasets.keys())),
        )

    def summary(self) -> str:
        return f"[{self.task.name}] train label model on {next(iter(self.task.train_datasets.keys()))} dataset"


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
            task=self.task, path_config=self.path_config, dataset=self.dataset
        )

    def summary(self) -> str:
        return f"[{self.task.name}] label dataset: {self.dataset}"


class PreprocessCopyCommand(DvcCommand):
    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "preprocess_copy.template", path_config, None, execute_immediately
        )
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            dataset=self.dataset,
            data_dir=self.path_config.data_dir,
        )

    def summary(self) -> str:
        return f"Pre-processing (copying): {self.dataset.name}"


class Preprocess7zCommand(DvcCommand):
    def __init__(
        self,
        path_config: PathConfig,
        dataset: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__(
            "preprocess_7z.template", path_config, None, execute_immediately
        )
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            dataset=self.dataset,
            data_dir=self.path_config.data_dir,
        )

    def summary(self) -> str:
        return f"Pre-processing (extracting): {self.dataset.name}"


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
        return template.render(
            dataset=self.dataset,
            data_dir=self.path_config.data_dir,
        )

    def summary(self) -> str:
        return f"Pre-processing (shell): {self.dataset.name}"


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


def add_all_tasks_to_dvc_pipeline(config: Config) -> None:
    path_config = config.paths
    create_directories_if_necessary(config)
    all_tasks = sorted(config.tasks.values(), key=lambda x: x.name)
    logger.info(
        f"Following tasks are added to the pipeline: {list(map(lambda x: x.name, all_tasks))}"
    )
    commands: List[DvcCommand] = []
    for dataset_name, dataset in config.datasets.items():
        if dataset.preprocessor == "copy":
            commands.append(PreprocessCopyCommand(path_config, dataset))
        elif dataset.preprocessor == "7z":
            commands.append(Preprocess7zCommand(path_config, dataset))
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
    root, dirs, files = next(os.walk(path_config.manual_stages))
    for file in files:
        commands.append(ManualCommand(path_config, Path(root) / file))
    for command in commands:
        completed_process = command.run()
        if completed_process.returncode != 0:
            print(completed_process.stderr.decode())
            break


if __name__ == "__main__":
    config = load_config()
    add_all_tasks_to_dvc_pipeline(config)
