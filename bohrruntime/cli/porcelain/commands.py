import json
from pprint import pprint
from typing import Optional

import click

from bohrruntime import setup_loggers
from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import load_workspace
from bohrruntime.util.profiler import Profiler


@click.group()
def porcelain():
    pass


@porcelain.command()
@click.argument("exp")
@click.argument("dataset")
@click.option("--debug", is_flag=True)
def label_dataset(exp: str, dataset: str, debug: bool):
    from bohrruntime.pipeline.label_dataset import label_dataset

    setup_loggers()
    workspace = load_workspace()
    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.task.get_dataset_by_id(dataset)
    label_dataset(exp, dataset, debug=debug)


@porcelain.command()
@click.option("--heuristic-group", type=str)
@click.option("--dataset", type=str)
@click.option("--profile", is_flag=True)
def apply_heuristics(
    heuristic_group: Optional[str], dataset: Optional[str], profile: bool
):
    from bohrruntime.pipeline.apply_heuristics import apply_heuristics_to_dataset

    setup_loggers()
    workspace = load_workspace()

    dataset = workspace.get_dataset_by_id(dataset)
    with Profiler(enabled=profile):
        apply_heuristics_to_dataset(heuristic_group, dataset)


@porcelain.command()
@click.option("--task", type=str)
@click.option("--heuristic-group", type=str)
@click.option("--dataset", type=str)
def compute_single_heuristic_metric(
    task: Optional[str], heuristic_group: Optional[str], dataset: Optional[str]
):
    from bohrruntime.pipeline.single_heuristic_metrics import (
        calculate_metrics_for_heuristic,
    )

    setup_loggers()
    path_config = PathConfig.load()
    workspace = load_workspace()
    task = workspace.get_task_by_name(task)
    dataset = workspace.get_dataset_by_id(dataset)
    calculate_metrics_for_heuristic(task, heuristic_group, dataset, path_config)


@porcelain.command()
@click.argument("exp")
@click.option("--dataset", type=str)
def combine_heuristics(exp: Optional[str], dataset: Optional[str]):
    from bohrruntime.pipeline.combine_heuristics import combine_applied_heuristics

    setup_loggers()
    workspace = load_workspace()

    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.task.get_dataset_by_id(dataset)
    combine_applied_heuristics(exp, dataset)


@porcelain.command()
@click.argument("dataset", type=str)
def load_dataset(dataset: str):
    from bohrruntime.core import load_dataset_from_explorer

    setup_loggers()
    workspace = load_workspace()

    dataset = workspace.get_dataset_by_id(dataset)

    load_dataset_from_explorer(dataset)


@porcelain.command()
@click.argument("exp")
def train_label_model(exp: str):
    from bohrruntime.pipeline.train_label_model import train_label_model

    # setup_loggers()
    workspace = load_workspace()
    path_config = PathConfig.load()
    exp = workspace.get_experiment_by_name(exp)
    stats = train_label_model(exp, exp.task.training_dataset, path_config)
    with open(path_config.exp_dir(exp) / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)
