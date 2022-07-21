from dataclasses import dataclass
from typing import Optional

from configobj import ConfigObj
from fs.base import FS

from bohrruntime.storageengine import StorageEngine
from bohrruntime.util.paths import create_fs


@dataclass(frozen=True)
class AppConfig:
    verbose: bool
    storage_engine: StorageEngine

    @staticmethod
    def load(fs: Optional[FS] = None) -> "AppConfig":
        fs = fs or create_fs()
        with fs.open(".bohr/local.config", "rb") as f:
            config_dict = ConfigObj(f).dict()
        try:
            verbose_str = config_dict["core"]["verbose"]
            verbose = verbose_str == "true" or verbose_str == "True"
        except KeyError:
            verbose = False
        return AppConfig(verbose, StorageEngine.init(fs))


def add_to_local_config(fs: FS, key: str, value: str) -> None:
    """
    >>> from fs.memoryfs import MemoryFS
    >>> fs = MemoryFS()
    >>> add_to_local_config(fs, "section.key", "value")
    >>> fs.readtext('.bohr/config')
    '[section]\\n    key = value\\n'
    >>> add_to_local_config(fs, "key", "value2")
    >>> fs.readtext('.bohr/config')
    '[section]\\n    key = value\\n[core]\\n    key = value2\\n'
    """
    if "." not in key:
        key = "core." + key
    section, section_key = key.rsplit(".", maxsplit=1)
    set_config_value(fs, ".bohr/config", section, section_key, value)


LOCAL_CONFIG_FILE_NAME = "local.config"
BOHR_CONFIG_DIR = ".bohr"


def set_config_value(fs: FS, file: str, section: str, key: str, value: str):
    """
    >>> from fs.memoryfs import MemoryFS
    >>> fs = MemoryFS()
    >>> set_config_value(fs, '.bohr/config', 'section', 'key', 'value')
    >>> fs.readtext('.bohr/config')
    '[section]\\n    key = value\\n'
    >>> set_config_value(fs, '.bohr/config', 'section', 'key', 'new_value')
    >>> fs.readtext('.bohr/config')
    '[section]\\n    key = new_value\\n'
    >>> set_config_value(fs, '.bohr/config', 'section2', 'key', 'new_value')
    >>> fs.readtext('.bohr/config')
    '[section]\\n    key = new_value\\n[section2]\\n    key = new_value\\n'
    """
    if not fs.exists(file):
        *dir, _ = file.split("/")
        if dir:
            fs.makedirs("/".join(dir), recreate=True)
        fs.create(file)
    with fs.open(file, "rb") as f:
        dct = ConfigObj(f).dict()
    if section not in dct:
        dct[section] = {}
    dct[section][key] = value
    with fs.open(file, "wb") as f:
        c = ConfigObj(dct)
        c.write(f)


def set_remote_read_url(fs: FS, url: str):
    set_config_value(fs, ".dvc/config", 'remote "read"', "url", url)


def set_remote_write_url(fs: FS, url: str):
    set_config_value(fs, ".dvc/config", 'remote "write"', "url", url)


def set_remote_write_credentials(fs: FS, user: str, password: str):
    set_config_value(fs, ".dvc/config", f'remote "write"', "user", user)
    set_config_value(fs, ".dvc/config.local", f'remote "write"', "password", password)
    # TODO gitignore config.local just in case?
