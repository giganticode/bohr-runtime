import logging

import click

from bohr import api
from bohr.config import load_config

logger = logging.getLogger(__name__)


@click.group()
def task():
    pass


@task.command()
@click.argument("task", type=str)
@click.argument("dataset", type=str)
@click.option("--repro", is_flag=True)
def add_dataset(task: str, dataset: str, repro: bool) -> None:
    config = load_config()
    if task not in config.tasks:
        logger.error(f"Task {task} is not defined")
        exit(404)
    if dataset not in config.datasets:
        logger.error(f"Dataset {dataset} is not defined")
        exit(404)
    dataset = api.add_dataset(config.tasks[task], config.datasets[dataset], config)
    print(f"Dataset {dataset} is added to the task {task}.")
    if repro:
        logger.info("Re-running the pipeline ...")
        api.repro(task, config=config)
