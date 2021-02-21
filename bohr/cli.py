import json
import subprocess
from pprint import pprint

import click

from bohr import pipeline
from bohr.config import add_to_local_config, load_config
from bohr.pipeline.dvc import add_all_tasks_to_dvc_pipeline
from bohr.pipeline.parse_labels import parse_label
from bohr.pipeline.profiler import Profiler

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def bohr():
    pass


@bohr.command()
def repro():
    config = load_config()
    add_all_tasks_to_dvc_pipeline(config)
    subprocess.run(["dvc", "repro"], cwd=config.project_root)


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
    pipeline.label_dataset(task, dataset, config, debug)


@bohr.command()
@click.argument("task")
@click.option("--n-workers", type=int, default=1)
@click.option("--profile", is_flag=True)
def apply_heuristics(task: str, n_workers: int, profile: bool):
    config = load_config()

    with Profiler(enabled=profile):
        pipeline.apply_heuristics(task, n_workers, config)


@bohr.command()
@click.argument("task")
def train_label_model(task: str):

    config = load_config()

    stats = pipeline.train_label_model(task, config)
    with open(config.paths.metrics / task / "label_model_metrics.json", "w") as f:
        json.dump(stats, f)
    pprint(stats)
