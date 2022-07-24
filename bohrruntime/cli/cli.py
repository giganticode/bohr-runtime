import logging
from typing import Optional

import click

import bohrruntime.commands as api
from bohrruntime import __version__, setup_loggers
from bohrruntime.appconfig import add_to_local_config
from bohrruntime.bohrconfigparser import load_workspace
from bohrruntime.cli.porcelain.commands import porcelain
from bohrruntime.cli.remote.commands import remote
from bohrruntime.storageengine import StorageEngine
from bohrruntime.util.logging import verbosity

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def bohr():
    pass


@bohr.command()
def push():
    api.push()


@bohr.command()
@click.argument("path")
@click.option("-r", "--rev", help="Revision of BOHR with needed version of task config")
def clone(
    path: str,
    rev: Optional[str] = None,
):
    api.clone(path, rev)


@bohr.command()
@click.option("-f", "--force", is_flag=True, help="Force pipeline reproduction")
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def repro(force: bool = False, verbose: bool = False):
    with verbosity(verbose):
        api.repro(force=force)


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def status(verbose: bool = False):
    setup_loggers(verbose)
    print(api.status())


@bohr.command()
@click.option("-v", "--verbose", is_flag=True, help="Enables verbose mode")
def refresh(verbose: bool = False):
    setup_loggers(verbose)
    api.refresh_pipeline_config(load_workspace(), StorageEngine.init())


@bohr.command()
@click.argument("key")
@click.argument("value")
def config(key: str, value: str):
    storage_engine = StorageEngine.init()
    add_to_local_config(storage_engine.fs, key, value)


bohr.add_command(porcelain)
bohr.add_command(remote)

# TODO bohr init - clone template repo.


if __name__ == "__main__":
    bohr()
