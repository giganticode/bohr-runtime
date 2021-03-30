import json
import subprocess
from pprint import pprint
from typing import Optional

import click

from bohr import __version__, api
from bohr.api import refresh_if_necessary
from bohr.config import load_config
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
@click.argument("task", required=False)
@click.option("--only-transient", is_flag=True)
def repro(task: Optional[str], only_transient: bool):
    if only_transient and task:
        raise ValueError("Both --only-transient and task is not supported")
    config = load_config()
    refresh_if_necessary(config.paths)
    paths_to_pull = [str(d.path_dist) for d in config.datasets.values()]
    cm = ["dvc", "pull"] + paths_to_pull
    logger.debug(f"Pulling datasets with command: {cm}")
    completed_process = subprocess.run(cm, cwd=config.paths.project_root)
    completed_process.check_returncode()
    cmd = ["dvc", "repro", "--pull"]
    if only_transient:
        cmd.extend(load_transient_stages(config.paths))
    if task:
        if task not in config.tasks:
            raise ValueError(f"Task {task} not found in bohr.json")
        cmd.extend(["--glob", f"{task}_*"])
    logger.debug(f"Running command: {cmd}")
    completed_process = subprocess.run(cmd, cwd=config.paths.project_root)
    completed_process.check_returncode()


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
    path_config = load_path_config()

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
    path_config = load_path_config()
    task = config.tasks[task]
    target_dataset = config.datasets[target_dataset]
    stats = stages.train_label_model(task, target_dataset, path_config)
    with open(path_config.metrics / task.name / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)
