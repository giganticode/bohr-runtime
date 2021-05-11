import logging
from typing import Optional

import click

from bohr.config import load_config

logger = logging.getLogger(__name__)


@click.group()
def dataset():
    pass


@dataset.command()
@click.option("-t", "--task", type=str)
def ls(task: Optional[str]) -> None:
    config = load_config()
    if task:
        if task not in config.tasks:
            logger.error(
                f"Task not found in the config: {task}. \n"
                f"Defined tasks: {list(config.tasks.keys())}"
            )
            exit(404)
        datasets = list(config.tasks[task].datasets.keys())
    else:
        datasets = config.datasets
    for dataset in datasets:
        print(dataset)
