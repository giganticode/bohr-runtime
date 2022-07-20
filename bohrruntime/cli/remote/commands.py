import click

import bohrruntime.appconfig as c
from bohrruntime.storageengine import StorageEngine


@click.group()
def remote():
    pass


@remote.command()
@click.argument("url")
def set_read_url(url: str):
    storage_engine = StorageEngine.init()
    c.set_remote_read_url(storage_engine.fs, url)


@remote.command()
@click.argument("url")
def set_write_url(url: str):
    storage_engine = StorageEngine.init()
    c.set_remote_write_url(storage_engine.fs, url)


@remote.command()
@click.argument("user")
@click.argument("password")
def set_write_credentials(user: str, password: str):
    storage_engine = StorageEngine.init()
    c.set_remote_write_credentials(storage_engine.fs, user, password)
