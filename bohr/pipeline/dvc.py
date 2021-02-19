import logging
import re
import subprocess
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from bohr.config import Config
from bohr.datamodel import Task

logger = logging.getLogger(__name__)

TEMPLATES = [
    "apply_heuristics.template",
    "train_label_model.template",
    "label_dataset.template",
]


def get_dvc_command(
    task: Task, template_name: str, config: Config, no_exec: bool = True
) -> List[str]:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent.parent), undefined=StrictUndefined
    )
    template = env.get_template(f"resources/dvc_command_templates/{template_name}")
    command = template.render(task=task, config=config)
    if no_exec:
        command = f"dvc run --no-exec --force {command}"
    command_array = list(filter(None, re.split("[\n ]", command)))
    return command_array


def add_all_tasks_to_dvc_pipeline(config: Config) -> None:
    all_tasks = sorted(config.tasks.values(), key=lambda x: x.name)
    logger.info(
        f"Following tasks are added to the pipeline: {list(map(lambda x: x.name, all_tasks))}"
    )
    for task in all_tasks:
        for template in TEMPLATES:
            command = get_dvc_command(task, template, config, no_exec=True)
            logger.info(f"Running {command}")
            subprocess.run(command, cwd=config.project_root)
