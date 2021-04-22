import json
import subprocess
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import click
import dvc.exceptions
import dvc.scm.base

from bohr import __version__, api
from bohr.api import refresh_if_necessary
from bohr.config import Config, load_config
from bohr.debugging import DataPointDebugger, DatasetDebugger
from bohr.pathconfig import add_to_local_config, load_path_config
from bohr.pipeline import stages
from bohr.pipeline.dvc import load_transient_stages
from bohr.pipeline.profiler import Profiler
from bohr.pipeline.stages.parse_labels import parse_label

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


import logging

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
@click.option("--top", type=int, required=False, default=None)
@click.option("--bottom", type=int, required=False, default=None)
@click.option("--force-refresh", is_flag=True)
def debug(
    task: str,
    dataset: str,
    old_rev: str,
    datapoint: Optional[int],
    top: Optional[int],
    bottom: Optional[int],
    force_refresh: bool,
) -> None:
    try:
        if datapoint is None:
            dataset_debugger = DatasetDebugger(task, dataset, old_rev, force_refresh)
            if bottom is None:
                dataset_debugger.show_worst_datapoints(top or 10)
            else:
                dataset_debugger.show_best_datapoints(bottom)
        else:
            DataPointDebugger(task, dataset, old_rev).show_datapoint_info(datapoint)
    except dvc.scm.base.RevError:
        logger.error(f"Revision does not exist: {old_rev}")
        exit(23)
    except dvc.exceptions.PathMissingError:
        logger.error(f"Dataset {dataset} or task {task} does not exist.")
        exit(24)


def run_dvc_commands(commands: List[List[str]], project_root: Path) -> None:
    for command in commands:
        logger.debug(f"Running command: {' '.join(command)}")
        completed_process = subprocess.run(command, cwd=project_root)
        completed_process.check_returncode()


def get_dvc_commands_to_repro(
    task: Optional[str], only_transient: bool, config: Config
) -> List[List[str]]:
    """
    # >>> import tempfile
    # >>> with tempfile.TemporaryDirectory() as tmpdirname:
    # ...     with open(Path(tmpdirname) / 'bohr.json', 'w') as f:
    # ...         print(f.write('{"bohr_framework_version": "0.3.9-rc", "tasks": {}, "datasets": {}}'))
    # ...     get_dvc_commands_to_repro(None, False, load_config(Path(tmpdirname)))
    """
    commands: List[List[str]] = []
    paths_to_pull = [str(d.path_dist) for d in config.datasets.values()]
    paths_to_pull.extend(str(l.link_file) for l in config.linkers)
    if len(paths_to_pull) > 0:
        commands.append(["dvc", "pull"] + paths_to_pull)

    # TODO run only task-related transient stages if task is passed:
    transient_stages = load_transient_stages(config.paths)
    if len(transient_stages) > 0:
        commands.append(["dvc", "repro"] + transient_stages)

    if not only_transient:
        cmd = ["dvc", "repro", "--pull"]
        if task:
            if task not in config.tasks:
                raise ValueError(f"Task {task} not found in bohr.json")
            cmd.extend(["--glob", f"{task}_*"])
        commands.append(cmd)
    return commands


@bohr.command()
@click.argument("task", required=False)
@click.option("--only-transient", is_flag=True)
def repro(task: Optional[str], only_transient: bool):
    if only_transient and task:
        raise ValueError("Both --only-transient and task is not supported")
    config = load_config()
    refresh_if_necessary(config.paths)
    commands = get_dvc_commands_to_repro(task, only_transient, config)
    run_dvc_commands(commands, config.paths.project_root)


@bohr.command()
def status():
    path_config = load_path_config()
    refresh_if_necessary(path_config)
    subprocess.run(["dvc", "status"], cwd=path_config.project_root)


@bohr.command()
def refresh():
    api.refresh()


@bohr.command()
@click.argument("key")
@click.argument("value")
def config(key: str, value: str):
    add_to_local_config("core", key, value)


@bohr.command()
def parse_labels():
    config = load_config()
    parse_label(config)


@bohr.command()
@click.argument("task")
@click.argument("dataset")
@click.option("--debug", is_flag=True)
def label_dataset(task: str, dataset: str, debug: bool):
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

    config = load_config()
    path_config = config.paths
    task = config.tasks[task]
    target_dataset = config.datasets[target_dataset]
    stats = stages.train_label_model(task, target_dataset, path_config)
    with open(path_config.metrics / task.name / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)
