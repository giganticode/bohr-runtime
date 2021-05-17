import json
from pprint import pprint
from typing import Optional

import click

from bohr import setup_loggers
from bohr.config.pathconfig import PathConfig
from bohr.datamodel.bohrrepo import load_bohr_repo
from bohr.util.profiler import Profiler


@click.group()
def porcelain():
    pass


@porcelain.command()
def parse_labels():
    from bohr.pipeline.parse_labels import parse_labels

    setup_loggers()
    parse_labels()


@porcelain.command()
@click.argument("task")
@click.argument("dataset")
@click.option("--debug", is_flag=True)
def label_dataset(task: str, dataset: str, debug: bool):
    from bohr.pipeline.label_dataset import label_dataset

    setup_loggers()
    bohr_repo = load_bohr_repo()
    task = bohr_repo.tasks[task]
    dataset = bohr_repo.datasets[dataset]
    label_dataset(task, dataset, debug=debug)


@porcelain.command()
@click.argument("task")
@click.option("--heuristic-group", type=str)
@click.option("--dataset", type=str)
@click.option("--profile", is_flag=True)
def apply_heuristics(
    task: str, heuristic_group: Optional[str], dataset: Optional[str], profile: bool
):
    from bohr.pipeline.apply_heuristics import apply_heuristics
    from bohr.pipeline.combine_heuristics import combine_applied_heuristics

    setup_loggers()
    bohr_repo = load_bohr_repo()

    task = bohr_repo.tasks[task]
    if heuristic_group:
        with Profiler(enabled=profile):
            dataset = bohr_repo.datasets[dataset]
            apply_heuristics(task, heuristic_group, dataset)
    else:
        combine_applied_heuristics(task)


@porcelain.command()
@click.argument("task")
@click.argument("target-dataset")
def train_label_model(task: str, target_dataset: str):
    from bohr.pipeline.train_label_model import train_label_model

    setup_loggers()
    bohr_repo = load_bohr_repo()
    path_config = PathConfig.load()
    task = bohr_repo.tasks[task]
    target_dataset = bohr_repo.datasets[target_dataset]
    stats = train_label_model(task, target_dataset, path_config)
    with open(path_config.metrics / task.name / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)
