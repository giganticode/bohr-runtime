import logging
from typing import Optional

import click

from bohrruntime import __version__, api, setup_loggers
from bohrruntime.appconfig import add_to_local_config
from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.cli.porcelain.commands import porcelain
from bohrruntime.core import load_workspace
from bohrruntime.util.logging import verbosity

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def bohr():
    pass


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
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def status(verbose: bool = False):
    setup_loggers(verbose)
    print(api.status())


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def refresh(verbose: bool = False):
    setup_loggers(verbose)
    api.refresh(load_workspace(), BohrFileSystem.init())


@bohr.command()
@click.argument("key")
@click.argument("value")
def config(key: str, value: str):
    add_to_local_config(key, value)


bohr.add_command(porcelain)


if __name__ == "__main__":
    bohr()
