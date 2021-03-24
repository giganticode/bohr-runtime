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

logger = logging.getLogger(__name__)


class DvcCommand(ABC):
    def __init__(
        self,
        template: str,
        config: Config,
        task: Task,
        execute_immediately: bool = False,
    ):
        self.template = template
        self.config = config
        self.task = task
        self.execute_immediately = execute_immediately

    def render_stage_template(self, template) -> str:
        return template.render(task=self.task, config=self.config)

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
            command, cwd=self.config.project_root, capture_output=True
        )


class ParseLabelsCommand(DvcCommand):
    def __init__(self, config: Config, execute_immediately: bool = False):
        super().__init__("parse_labels.template", config, None, execute_immediately)

    def summary(self) -> str:
        return "parse labels"


class ApplyHeuristicsCommand(DvcCommand):
    def __init__(
        self,
        config: Config,
        task: Task,
        heuristic_group: str,
        dataset: str,
        execute_immediately: bool = False,
    ):
        super().__init__("apply_heuristics.template", config, task, execute_immediately)
        self.heuristic_group = heuristic_group
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(
            task=self.task,
            config=self.config,
            heuristic_group=self.heuristic_group,
            dataset=self.dataset,
        )

    def summary(self) -> str:
        return f"[{self.task.name}] apply heuristics (group: {self.heuristic_group}) to {self.dataset}"


class CombineHeuristicsCommand(DvcCommand):
    def __init__(self, config: Config, task: Task, execute_immediately: bool = False):
        super().__init__(
            "combine_heuristics.template", config, task, execute_immediately
        )

    def summary(self) -> str:
        return f"[{self.task.name}] combine heuristics"


class TrainLabelModelCommand(DvcCommand):
    def __init__(self, config: Config, task: Task, execute_immediately: bool = False):
        super().__init__(
            "train_label_model.template", config, task, execute_immediately
        )

    def render_stage_template(self, template) -> str:
        return template.render(
            task=self.task,
            config=self.config,
            target_dataset=next(iter(self.task.train_datasets.keys())),
        )

    def summary(self) -> str:
        return f"[{self.task.name}] train label model on {next(iter(self.task.train_datasets.keys()))} dataset"


class LabelDatasetCommand(DvcCommand):
    def __init__(
        self,
        config: Config,
        task: Task,
        dataset: str,
        execute_immediately: bool = False,
    ):
        super().__init__("label_dataset.template", config, task, execute_immediately)
        self.dataset = dataset

    def render_stage_template(self, template) -> str:
        return template.render(task=self.task, config=self.config, dataset=self.dataset)

    def summary(self) -> str:
        return f"[{self.task.name}] label dataset: {self.dataset}"


class PreprocessCopyCommand(DvcCommand):
    def __init__(
        self,
        config: Config,
        dataset_name: str,
        dataloader: Dataset,
        execute_immediately: bool = False,
    ):
        super().__init__("preprocess_copy.template", config, None, False)
        self.dataset_name = dataset_name
        self.dataloader = dataloader

    def render_stage_template(self, template) -> str:
        return template.render(
            dataset_name=self.dataset_name,
            dataloader=self.dataloader,
            data_dir=self.config.paths.data_dir,
        )

    def summary(self) -> str:
        return f"Pre-processing: {self.dataset_name}"


class ManualCommand(DvcCommand):
    def __init__(
        self, config: Config, path_to_template: Path, execute_immediately: bool = False
    ):
        super().__init__("empty.template", config, None, execute_immediately)
        self.path_to_template = path_to_template

    def render_stage_template(self, template) -> str:
        with open(self.path_to_template) as f:
            return f.read()

    def summary(self) -> str:
        return f"Command from: {self.path_to_template}"


def create_directories_if_necessary(config: Config) -> None:
    for task in config.tasks.values():
        for heuristic_group in task.heuristic_groups:
            (config.paths.generated / task.name / heuristic_group).mkdir(
                exist_ok=True, parents=True
            )
            (config.paths.metrics / task.name / heuristic_group).mkdir(
                exist_ok=True, parents=True
            )
    config.paths.labeled_data.mkdir(exist_ok=True, parents=True)


def add_all_tasks_to_dvc_pipeline(config: Config) -> None:
    create_directories_if_necessary(config)
    all_tasks = sorted(config.tasks.values(), key=lambda x: x.name)
    logger.info(
        f"Following tasks are added to the pipeline: {list(map(lambda x: x.name, all_tasks))}"
    )
    commands = []
    for dataset_name, dataloader in config.dataloaders.items():
        if dataloader.format not in ["zip", "7z"]:
            commands.append(PreprocessCopyCommand(config, dataset_name, dataloader))
        else:
            raise NotImplementedError()
    commands.append(ParseLabelsCommand(config))
    for task in all_tasks:
        for heuristic_group in task.heuristic_groups:
            for dataset_name in task.datasets:
                commands.append(
                    ApplyHeuristicsCommand(config, task, heuristic_group, dataset_name)
                )
        commands.append(CombineHeuristicsCommand(config, task))
        commands.append(TrainLabelModelCommand(config, task))
        for dataset_name in task.datasets:
            commands.append(LabelDatasetCommand(config, task, dataset_name))
    root, dirs, files = next(os.walk(config.paths.manual_stages))
    for file in files:
        commands.append(ManualCommand(config, Path(root) / file))
    for command in commands:
        completed_process = command.run()
        if completed_process.returncode != 0:
            print(completed_process.stderr.decode())
            break


if __name__ == "__main__":
    config = load_config()

    add_all_tasks_to_dvc_pipeline(config)
