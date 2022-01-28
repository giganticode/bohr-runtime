from typing import Optional

import click

from bohrruntime import setup_loggers
from bohrruntime.bohrfs import BohrFileSystem
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
    from bohrruntime.stages.label_dataset import label_dataset

    setup_loggers()
    workspace = load_workspace()
    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.get_dataset_by_id(dataset)
    label_dataset(exp, dataset, debug=debug)


@porcelain.command()
@click.option("--heuristic-group", type=str)
@click.option("--dataset", type=str)
@click.option("--profile", is_flag=True)
def apply_heuristics(
    heuristic_group: Optional[str], dataset: Optional[str], profile: bool
):
    from bohrruntime.stages.apply_heuristics import apply_heuristics_to_dataset

    setup_loggers()
    workspace = load_workspace()

    dataset = workspace.get_dataset_by_id(dataset)
    with Profiler(enabled=profile):
        apply_heuristics_to_dataset(heuristic_group, dataset)


@porcelain.command()
@click.argument("task", type=str)
def compute_single_heuristic_metric(task: Optional[str]):
    from bohrruntime.stages.single_heuristic_metrics import (
        calculate_metrics_for_heuristic,
    )

    setup_loggers()
    path_config = BohrFileSystem.init()
    workspace = load_workspace()
    task = workspace.get_task_by_name(task)
    calculate_metrics_for_heuristic(task, path_config)


@porcelain.command()
@click.argument("exp")
@click.option("--dataset", type=str)
def combine_heuristics(exp: Optional[str], dataset: Optional[str]):
    from bohrruntime.stages.combine_heuristics import combine_applied_heuristics

    setup_loggers()
    workspace = load_workspace()

    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.get_dataset_by_id(dataset)
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
    from bohrruntime.stages.train_label_model import train_label_model

    # setup_loggers()
    workspace = load_workspace()
    path_config = BohrFileSystem.init()
    exp = workspace.get_experiment_by_name(exp)
    train_label_model(exp, path_config)


@porcelain.command()
@click.argument("exp")
@click.argument("dataset")
def run_metrics_and_analysis(exp: str, dataset: str):
    from bohrruntime.stages.experiment_metrics import calculate_experiment_metrics

    setup_loggers()
    workspace = load_workspace()
    path_config = BohrFileSystem.init()
    exp = workspace.get_experiment_by_name(exp)
    dataset = workspace.get_dataset_by_id(dataset)
    calculate_experiment_metrics(exp, dataset, path_config)
