import logging

import click

from bohr import api
from bohr.datamodel.bohrrepo import load_bohr_repo

logger = logging.getLogger(__name__)


@click.group()
def task():
    pass


@task.command()
@click.argument("task", type=str)
@click.argument("dataset", type=str)
@click.option("--repro", is_flag=True)
def add_dataset(task: str, dataset: str, repro: bool) -> None:
    bohr_repo = load_bohr_repo()
    if task not in bohr_repo.tasks:
        logger.error(f"Task {task} is not defined")
        exit(404)
    if dataset not in bohr_repo.datasets:
        logger.error(f"Dataset {dataset} is not defined")
        exit(404)
    dataset = api.add_dataset(
        bohr_repo.tasks[task], bohr_repo.datasets[dataset], bohr_repo
    )
    print(f"Dataset {dataset} is added to the task {task}.")
    if repro:
        logger.info("Re-running the pipeline ...")
        api.repro(task, bohr_repo=bohr_repo)
