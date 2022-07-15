from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import toml
from fs.base import FS

from bohrruntime.storageengine import StorageEngine
from bohrruntime.util.paths import AbsolutePath, create_fs, gitignore_file


@dataclass(frozen=True)
class AppConfig:
    verbose: bool
    fs: StorageEngine

    @staticmethod
    def load(fs: Optional[FS] = None) -> "AppConfig":
        fs = fs or create_fs()
        config_dict = load_config_dict_from_file(fs)
        try:
            verbose_str = config_dict["core"]["verbose"]
            verbose = verbose_str == "true" or verbose_str == "True"
        except KeyError:
            verbose = False
        return AppConfig(verbose, StorageEngine.init(fs))


def add_to_local_config(key: str, value: str) -> None:
    fs = create_fs()
    dct, local_config_path = load_config_dict_from_file(fs, return_path=True)
    if "." not in key:
        raise ValueError(f"The key must have format [section].[key] but is {key}")
    section, key = key.split(".", maxsplit=1)
    if section not in dct:
        dct[section] = {}
    dct[section][key] = value
    with fs.open(local_config_path, "w") as f:
        toml.dump(dct, f)


LOCAL_CONFIG_FILE_NAME = "local.config"


def load_config_dict_from_file(
    fs: FS, return_path: bool = False
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], str]]:
    config_dir_subfs = fs.opendir(".bohr")
    if not config_dir_subfs.exists(LOCAL_CONFIG_FILE_NAME):
        config_dir_subfs.touch(LOCAL_CONFIG_FILE_NAME)
        gitignore_file(config_dir_subfs, LOCAL_CONFIG_FILE_NAME)
    with config_dir_subfs.open("local.config") as f:
        dct = toml.load(f)
    return (dct, ".bohr/local.config") if return_path else dct
