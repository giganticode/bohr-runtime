import logging
from typing import Optional

import click
import dvc.exceptions
import dvc.scm.base
import pandas as pd

from bohrruntime import __version__, api, setup_loggers
from bohrruntime.cli.porcelain.commands import porcelain
from bohrruntime.config.pathconfig import PathConfig, add_to_local_config
from bohrruntime.core import load_dataset, load_workspace
from bohrruntime.formatting import tabulate_artifacts
from bohrruntime.pipeline.apply_heuristics import apply_heuristic_to_dataset
from bohrruntime.pipeline.data_analysis import calculate_metrics, get_fired_datapoints
from bohrruntime.util.logging import verbosity

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def bohr():
    pass


@bohr.command()
@click.argument("exp")
@click.argument("dataset")
@click.argument("old_rev", type=str, required=False, default="master")
@click.option("-i", "--datapoint", type=int, required=False, default=None)
@click.option("-h", "--heuristic", type=str, required=False, default=None)
@click.option(
    "-m",
    "--metric",
    type=click.Choice(["improvement", "certainty", "precision", "no"]),
    required=False,
    default="improvement",
)
@click.option("-n", "--n-datapoints", type=int, required=False, default=None)
@click.option("-r", "--reverse", is_flag=True)
@click.option("--force-refresh", is_flag=True)
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def debug(
    exp: str,
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
    from bohrruntime.debugging import DataPointDebugger, DatasetDebugger

    setup_loggers(verbose)
    try:
        # TODO clean up everything
        dataset_debugger = DatasetDebugger(
            exp, dataset, old_rev, force_update=force_refresh
        )
        if heuristic is not None:
            datapoint_debugger = DataPointDebugger(
                exp, dataset, dataset_debugger, old_rev, force_update=force_refresh
            )
            ones, zeros, stats = get_fired_datapoints(
                datapoint_debugger.new_matrix,
                heuristic.split(","),
                dataset_debugger.result_df,
            )
            print(
                "==================             0               ======================"
            )
            if metric != "no":
                zeros.sort_values(by=metric, inplace=True, ascending=reverse)
            tabulate_artifacts(zeros.head(n_datapoints))
            print(
                "==================             1               ======================"
            )
            if metric != "no":
                ones.sort_values(by=metric, inplace=True, ascending=reverse)
            tabulate_artifacts(ones.head(n_datapoints))
            print(stats)
        elif datapoint is None:
            dataset_debugger.show_datapoints(
                metric, n_datapoints or 10, reverse=reverse
            )
        else:
            datapoint_debugger = DataPointDebugger(
                exp, dataset, dataset_debugger, old_rev, force_update=force_refresh
            )
            datapoint_debugger.show_datapoint_info(datapoint)
    except dvc.scm.base.RevError:
        logger.error(f"Revision does not exist: {old_rev}")
        exit(23)
    except dvc.exceptions.PathMissingError:
        logger.error(f"Dataset {dataset} or experiment {exp} does not exist.")
        exit(24)


@bohr.command()
@click.argument("task", required=False)
@click.option("-f", "--force", is_flag=True, help="Force pipeline reproduction")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
@click.option("--no-pull", is_flag=True, help="Do not pull from dvc remote")
def repro(
    task: Optional[str],
    force: bool = False,
    verbose: bool = False,
    no_pull: bool = False,
):
    with verbosity(verbose):
        api.repro(task, force, pull=not no_pull)


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
    workspace = load_workspace()
    task = workspace.get_task_by_id(task)
    dataset = task.get_dataset_by_id(dataset)
    applied_matrix = apply_heuristic_to_dataset(task, heuristic, dataset, n_datapoints)
    df = pd.DataFrame(applied_matrix, columns=[heuristic])
    artifact_df = load_dataset(dataset)
    ones, zeros, stats = get_fired_datapoints(df, [heuristic], artifact_df)

    print("==================             0               ======================")
    tabulate_artifacts(zeros)
    print("==================             1               ======================")
    tabulate_artifacts(ones)
    print(stats)
    label_series = (
        artifact_df[task.label_column_name]
        if task.label_column_name in artifact_df.columns
        else None
    )
    metrics = calculate_metrics(applied_matrix, [heuristic], label_series=label_series)
    print(metrics)


# @bohr.command()
# @click.argument("task")
# @click.argument("target")
# @click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
# def pull(task: str, target: str, verbose: bool = False):
#     try:
#         with verbosity(verbose):
#             path_config = PathConfig.load()
#             refresh(path_config)
#             path = api.pull(task, target, path_config=path_config)
#             logger.info(
#                 f"The dataset is available at {path_config.project_root / path}"
#             )
#     except BohrDatasetNotFound as ex:
#         logger.error(ex, exc_info=logger.getEffectiveLevel() == logging.DEBUG)
#         exit(404)


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def status(verbose: bool = False):
    setup_loggers(verbose)
    print(api.status())


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def refresh(verbose: bool = False):
    setup_loggers(verbose)
    api.refresh(load_workspace(), PathConfig.load())


@bohr.command()
@click.argument("key")
@click.argument("value")
def config(key: str, value: str):
    add_to_local_config(key, value)


bohr.add_command(porcelain)


if __name__ == "__main__":
    bohr()
