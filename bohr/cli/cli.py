import logging
from typing import Optional

import click
import dvc.exceptions
import dvc.scm.base

from bohr import __version__, api, setup_loggers
from bohr.api import BohrDatasetNotFound, refresh_if_necessary
from bohr.cli.dataset.commands import dataset
from bohr.cli.porcelain.commands import porcelain
from bohr.cli.task.commands import task
from bohr.config.pathconfig import PathConfig, add_to_local_config
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
    metric: str,
    n_datapoints: Optional[int],
    reverse: bool,
    force_refresh: bool,
    verbose: bool = False,
) -> None:
    from bohr.debugging import DataPointDebugger, DatasetDebugger

    setup_loggers(verbose)
    try:
        if datapoint is None:
            dataset_debugger = DatasetDebugger(
                task, dataset, old_rev, force_update=force_refresh
            )
            dataset_debugger.show_datapoints(
                metric, n_datapoints or 10, reverse=reverse
            )
        else:
            DataPointDebugger(
                task, dataset, old_rev, force_update=force_refresh
            ).show_datapoint_info(datapoint)
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
