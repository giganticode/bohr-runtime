import json
from pprint import pprint
from typing import Optional

import click
import dvc.exceptions
import dvc.scm.base

from bohr import __version__, api
from bohr.api import BohrDatasetNotFound, refresh_if_necessary
from bohr.cli.dataset.commands import dataset
from bohr.cli.task.commands import task
from bohr.config import load_config
from bohr.debugging import DataPointDebugger, DatasetDebugger
from bohr.pathconfig import AppConfig, add_to_local_config
from bohr.pipeline import stages
from bohr.pipeline.profiler import Profiler
from bohr.pipeline.stages.parse_labels import parse_label

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


import logging

logger = logging.getLogger(__name__)


class verbosity:
    def __init__(self, verbose: bool = True):
        self.current_verbosity = AppConfig.load().verbose
        self.verbose = verbose or self.current_verbosity

    def __enter__(self):
        add_to_local_config("core.verbose", str(self.verbose))
        setup_loggers(self.verbose)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        add_to_local_config("core.verbose", str(self.current_verbosity))


def setup_loggers(verbose: Optional[bool] = None):
    if verbose is None:
        verbose = AppConfig.load().verbose
    logging.captureWarnings(True)
    root = logging.root
    for (logger_name, logger) in root.manager.loggerDict.items():
        if logger_name != "bohr" and not logger_name.startswith("bohr."):
            logger.disabled = True
        else:
            if verbose:
                logging.getLogger("bohr").setLevel(logging.DEBUG)


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
        config = load_config()
        refresh_if_necessary(config)
        api.repro(task, only_transient, force, config)


@bohr.command()
@click.argument("target")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def pull(target: Optional[str], verbose: bool = False):
    try:
        with verbosity(verbose):
            config = load_config()
            refresh_if_necessary(config)
            path = api.pull(target, config)
            logger.info(
                f"The dataset is available at {config.paths.project_root / path}"
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


@bohr.command()
def parse_labels():
    setup_loggers()
    config = load_config()
    parse_label(config)


@bohr.command()
@click.argument("task")
@click.argument("dataset")
@click.option("--debug", is_flag=True)
def label_dataset(task: str, dataset: str, debug: bool):
    setup_loggers()
    config = load_config()
    task = config.tasks[task]
    dataset = config.datasets[dataset]
    stages.label_dataset(task, dataset, config, debug)


@bohr.command()
@click.argument("task")
@click.option("--heuristic-group", type=str)
@click.option("--dataset", type=str)
@click.option("--profile", is_flag=True)
def apply_heuristics(
    task: str, heuristic_group: Optional[str], dataset: Optional[str], profile: bool
):
    setup_loggers()
    config = load_config()
    path_config = config.paths

    task = config.tasks[task]
    if heuristic_group:
        with Profiler(enabled=profile):
            dataset = config.datasets[dataset]
            stages.apply_heuristics(task, path_config, heuristic_group, dataset)
    else:
        stages.combine_applied_heuristics(task, path_config)


@bohr.command()
@click.argument("task")
@click.argument("target-dataset")
def train_label_model(task: str, target_dataset: str):
    setup_loggers()
    config = load_config()
    path_config = config.paths
    task = config.tasks[task]
    target_dataset = config.datasets[target_dataset]
    stats = stages.train_label_model(task, target_dataset, path_config)
    with open(path_config.metrics / task.name / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)


bohr.add_command(dataset)
bohr.add_command(task)
