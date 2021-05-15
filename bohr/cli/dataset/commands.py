import logging
import textwrap
from pathlib import Path
from typing import Optional

import click
from tabulate import tabulate

from bohr import api
from bohr.config import load_config

logger = logging.getLogger(__name__)


@click.group()
def dataset():
    pass


@dataset.command()
@click.option("-t", "--task", type=str)
@click.option("-a", "--extended-list", is_flag=True)
def ls(task: Optional[str], extended_list: bool) -> None:
    config = load_config()
    if task:
        if task not in config.tasks:
            logger.error(
                f"Task not found in the config: {task}. \n"
                f"Defined tasks: {list(config.tasks.keys())}"
            )
            exit(404)
        datasets = config.tasks[task].datasets
    else:
        datasets = config.datasets
    if extended_list:
        print(
            tabulate(
                [
                    [dataset_name, textwrap.fill(dataset.description)]
                    for dataset_name, dataset in datasets.items()
                ],
                tablefmt="fancy_grid",
            )
        )
    else:
        for dataset in datasets:
            print(dataset)


@dataset.command()
@click.argument("path", type=str)
@click.option("-a", "--artifact", required=True)
@click.option("-t", "--test-set", is_flag=True)
def add(path: str, artifact: str, test_set: bool) -> None:
    dataset = api.add(Path(path), artifact, test_set)
    print(f"Dataset {dataset.name} is added.")
