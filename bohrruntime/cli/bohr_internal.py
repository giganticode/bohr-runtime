import logging

from bohrruntime import __version__
from pathlib import Path
from typing import Optional

import click

from bohrruntime import setup_loggers
from bohrruntime.bohrconfigparser import load_workspace
from bohrruntime.datamodel.experiment import SynteticExperiment
from bohrruntime.heuristics import HeuristicURI
from bohrruntime.storageengine import StorageEngine
from bohrruntime.util.profiler import Profiler

"""
Implementation of internal cli commands used by the pipeline manager to execute stages
"""

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def bohr_internal():
    pass


@bohr_internal.command()
@click.argument("exp")
@click.argument("dataset")
@click.option("--debug", is_flag=True)
def prepare_dataset(exp: str, dataset: str, debug: bool):
    from bohrruntime.stages import prepare_dataset

    setup_loggers()
    fs = StorageEngine.init()
    workspace = load_workspace()
    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.get_dataset_by_id(dataset)
    prepare_dataset(exp, dataset, fs)


@bohr_internal.command()
@click.option("--heuristic-group", type=str)
@click.option("--dataset", type=str)
@click.option("--profile", is_flag=True)
def apply_heuristics(
    heuristic_group: Optional[str], dataset: Optional[str], profile: bool
):
    from bohrruntime.stages import apply_heuristics_to_dataset

    setup_loggers()
    fs = StorageEngine.init()
    workspace = load_workspace()

    dataset = workspace.get_dataset_by_id(dataset)
    with Profiler(enabled=profile):
        heuristic_applier = workspace.experiments[0].task.get_heuristic_applier()
        apply_heuristics_to_dataset(
            heuristic_applier,
            HeuristicURI(Path(heuristic_group), fs.heuristics_subfs()),
            dataset,
            fs,
        )


@bohr_internal.command()
@click.argument("task", type=str)
def compute_single_heuristic_metric(task: Optional[str]):
    from bohrruntime.stages import calculate_single_heuristic_metrics

    setup_loggers()
    fs = StorageEngine.init()
    workspace = load_workspace()
    task = workspace.get_task_by_name(task)
    calculate_single_heuristic_metrics(task, fs)


@bohr_internal.command()
@click.argument("exp")
@click.option("--dataset", type=str)
def combine_heuristics(exp: Optional[str], dataset: Optional[str]):
    from bohrruntime.stages import combine_applied_heuristics

    setup_loggers()
    fs = StorageEngine.init()
    workspace = load_workspace()

    exp = workspace.get_experiment_by_name(exp)
    dataset = exp.get_dataset_by_id(dataset)
    combine_applied_heuristics(exp, dataset, fs)


@bohr_internal.command()
@click.argument("dataset", type=str)
def load_dataset(dataset: str):
    from bohrruntime.stages import load_dataset

    setup_loggers()
    workspace = load_workspace()
    fs = StorageEngine.init()

    dataset = workspace.get_dataset_by_id(dataset)

    load_dataset(dataset, fs)


@bohr_internal.command()
@click.argument("exp")
def train_model(exp: str):
    from bohrruntime.stages import train_model

    # setup_loggers()
    workspace = load_workspace()
    fs = StorageEngine.init()
    exp = workspace.get_experiment_by_name(exp)

    train_model(exp, fs)


@bohr_internal.command()
@click.argument("exp")
@click.argument("dataset")
def run_metrics_and_analysis(exp: str, dataset: str):
    from bohrruntime.stages import calculate_experiment_metrics

    setup_loggers()
    workspace = load_workspace()
    fs = StorageEngine.init()
    exp = workspace.get_experiment_by_name(exp)
    dataset = workspace.get_dataset_by_id(dataset)
    calculate_experiment_metrics(exp, dataset, fs)


@bohr_internal.command()
@click.argument("task")
@click.argument("dataset")
def compute_random_model_metrics(task: str, dataset: str):
    from bohrruntime.stages import calculate_experiment_metrics

    setup_loggers()
    workspace = load_workspace()
    fs = StorageEngine.init()
    task = workspace.get_task_by_name(task)
    dataset = workspace.get_dataset_by_id(dataset)
    exp = SynteticExperiment("random_model", task, type="random")
    calculate_experiment_metrics(exp, dataset, fs)


@bohr_internal.command()
@click.argument("task")
@click.argument("dataset")
def compute_zero_model_metrics(task: str, dataset: str):
    from bohrruntime.stages import calculate_experiment_metrics

    setup_loggers()
    workspace = load_workspace()
    fs = StorageEngine.init()
    task = workspace.get_task_by_name(task)
    dataset = workspace.get_dataset_by_id(dataset)
    exp = SynteticExperiment("zero_model", task, type="zero")
    calculate_experiment_metrics(exp, dataset, fs)


if __name__ == '__main__':
    bohr_internal()