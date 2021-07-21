import logging
from typing import Optional

import click
import dvc.exceptions
import dvc.scm.base
import pandas as pd

from bohr import __version__, api, setup_loggers
from bohr.api import BohrDatasetNotFound, refresh_if_necessary
from bohr.cli.dataset.commands import dataset
from bohr.cli.porcelain.commands import porcelain
from bohr.cli.task.commands import task
from bohr.config.pathconfig import PathConfig, add_to_local_config
from bohr.datamodel.bohrrepo import load_bohr_repo
from bohr.formatting import tabulate_artifacts
from bohr.pipeline.apply_heuristics import apply_heuristic_to_dataset
from bohr.pipeline.data_analysis import calculate_metrics, get_fired_datapoints
from bohr.util.logging import verbosity

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def bohr():
    pass


@bohr.command()
@click.argument("task")
@click.argument("dataset")
@click.argument("old_rev", type=str, required=False, default="master")
@click.option("-i", "--datapoint", type=int, required=False, default=None)
@click.option("-h", "--heuristic", type=str, required=False, default=None)
@click.option(
    "-m",
    "--metric",
    type=click.Choice(["improvement", "certainty", "precision"]),
    required=False,
    default="improvement",
)
@click.option("-n", "--n-datapoints", type=int, required=False, default=None)
@click.option("-r", "--reverse", is_flag=True)
@click.option("--force-refresh", is_flag=True)
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def debug(
    task: str,
    dataset: str,
    old_rev: str,
    datapoint: Optional[int],
    heuristic: Optional[str],
    metric: str,
    n_datapoints: Optional[int],
    reverse: bool,
    force_refresh: bool,
    verbose: bool = False,
) -> None:
    from bohr.debugging import DataPointDebugger, DatasetDebugger

    setup_loggers(verbose)
    try:
        # TODO clean up everything
        dataset_debugger = DatasetDebugger(
            task, dataset, old_rev, force_update=force_refresh
        )
        if heuristic is not None:
            datapoint_debugger = DataPointDebugger(
                task, dataset, dataset_debugger, old_rev, force_update=force_refresh
            )
            ones, zeros, stats = get_fired_datapoints(
                datapoint_debugger.new_matrix,
                heuristic.split(","),
                dataset_debugger.combined_df,
            )
            print(
                "==================             0               ======================"
            )
            tabulate_artifacts(zeros.head(n_datapoints))
            print(
                "==================             1               ======================"
            )
            tabulate_artifacts(ones.head(n_datapoints))
            print(stats)
        elif datapoint is None:
            dataset_debugger.show_datapoints(
                metric, n_datapoints or 10, reverse=reverse
            )
        else:
            datapoint_debugger = DataPointDebugger(
                task, dataset, dataset_debugger, old_rev, force_update=force_refresh
            )
            datapoint_debugger.show_datapoint_info(datapoint)
    except dvc.scm.base.RevError:
        logger.error(f"Revision does not exist: {old_rev}")
        exit(23)
    except dvc.exceptions.PathMissingError:
        logger.error(f"Dataset {dataset} or task {task} does not exist.")
        exit(24)


@bohr.command()
@click.argument("task", required=False)
@click.option("--only-transient", is_flag=True)
@click.option("-f", "--force", is_flag=True, help="Force pipeline reproduction")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def repro(
    task: Optional[str],
    only_transient: bool,
    force: bool = False,
    verbose: bool = False,
):
    with verbosity(verbose):
        if only_transient and task:
            raise ValueError("Both --only-transient and task is not supported")
        api.repro(task, only_transient, force)


@bohr.command()
@click.argument("task")
@click.argument("dataset")
@click.option("-h", "--heuristic", type=str, required=True, default=None)
@click.option("-n", "--n-datapoints", type=int, required=False, default=None)
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def run(
    task: str,
    dataset: str,
    heuristic: str,
    n_datapoints: Optional[int],
    verbose: bool = False,
) -> None:

    setup_loggers(verbose)
    bohr_repo = load_bohr_repo()
    task = bohr_repo.tasks[task]
    dataset = bohr_repo.datasets[dataset]
    applied_matrix = apply_heuristic_to_dataset(task, heuristic, dataset)
    df = pd.DataFrame(applied_matrix, columns=[heuristic])
    artifact_df = dataset.load()
    ones, zeros, stats = get_fired_datapoints(df, [heuristic], artifact_df)

    print("==================             0               ======================")
    tabulate_artifacts(zeros.head(n_datapoints))
    print("==================             1               ======================")
    tabulate_artifacts(ones.head(n_datapoints))
    print(stats)
    label_series = (
        artifact_df[task.label_column_name]
        if task.label_column_name in artifact_df.columns
        else None
    )
    metrics = calculate_metrics(applied_matrix, [heuristic], label_series=label_series)
    print(metrics)


@bohr.command()
@click.argument("task")
@click.argument("target")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def pull(task: str, target: str, verbose: bool = False):
    try:
        with verbosity(verbose):
            path_config = PathConfig.load()
            refresh_if_necessary(path_config)
            path = api.pull(task, target, path_config=path_config)
            logger.info(
                f"The dataset is available at {path_config.project_root / path}"
            )
    except BohrDatasetNotFound as ex:
        logger.error(ex, exc_info=logger.getEffectiveLevel() == logging.DEBUG)
        exit(404)


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def status(verbose: bool = False):
    setup_loggers(verbose)
    print(api.status())


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def refresh(verbose: bool = False):
    setup_loggers(verbose)
    api.refresh()


@bohr.command()
@click.argument("key")
@click.argument("value")
def config(key: str, value: str):
    add_to_local_config(key, value)


bohr.add_command(dataset)
bohr.add_command(task)
bohr.add_command(porcelain)
