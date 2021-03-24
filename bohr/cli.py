import json
import subprocess
from pprint import pprint
from typing import Optional

import click

from bohr import __version__, api
from bohr.api import refresh_if_necessary
from bohr.config import add_to_local_config, load_config
from bohr.lock import bohr_up_to_date, update_lock
from bohr.pipeline import stages
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
def repro(task: Optional[str]):
    config = load_config()
    refresh_if_necessary(config)
    cmd = ["dvc", "repro", "--pull"]
    if task:
        if task not in config.tasks:
            raise ValueError(f"Task {task} not found in bohr.json")
        cmd.extend(["--glob", f"{task}_*"])
    subprocess.run(cmd, cwd=config.project_root)


@bohr.command()
def status():
    config = load_config()
    refresh_if_necessary(config)
    subprocess.run(["dvc", "status"], cwd=config.project_root)


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

    if heuristic_group:
        with Profiler(enabled=profile):
            stages.apply_heuristics(task, config, heuristic_group, dataset)
    else:
        stages.combine_applied_heuristics(task, config)


@bohr.command()
@click.argument("task")
@click.argument("target-dataset")
def train_label_model(task: str, target_dataset: str):

    config = load_config()

    stats = stages.train_label_model(task, target_dataset, config)
    with open(config.paths.metrics / task / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)
