import logging
import textwrap
from pathlib import Path
from typing import Optional

import click
from tabulate import tabulate

from bohr import api
from bohr.datamodel.bohrrepo import load_bohr_repo

logger = logging.getLogger(__name__)


@click.group()
def dataset():
    pass


@dataset.command()
@click.option("-t", "--task", type=str)
@click.option("-a", "--extended-list", is_flag=True)
def ls(task: Optional[str], extended_list: bool) -> None:
    bohr_repo = load_bohr_repo()
    if task:
        if task not in bohr_repo.tasks:
            logger.error(
                f"Task not found in the config: {task}. \n"
                f"Defined tasks: {list(bohr_repo.tasks.keys())}"
            )
            exit(404)
        datasets = bohr_repo.tasks[task].datasets
    else:
        datasets = bohr_repo.datasets
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
def add(path: str, artifact: str) -> None:
    dataset = api.add(Path(path), artifact)
    print(f"Dataset {dataset.name} is added.")
