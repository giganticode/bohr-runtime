from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import toml

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.fs import find_project_root, gitignore_file
from bohrruntime.util.paths import AbsolutePath


@dataclass(frozen=True)
class AppConfig:
    verbose: bool
    fs: BohrFileSystem

    @staticmethod
    def load(project_root: Optional[AbsolutePath] = None) -> "AppConfig":
        project_root = project_root or find_project_root()
        config_dict = load_config_dict_from_file(project_root)
        try:
            verbose_str = config_dict["core"]["verbose"]
            verbose = verbose_str == "true" or verbose_str == "True"
        except KeyError:
            verbose = False
        return AppConfig(verbose, BohrFileSystem.init())


def add_to_local_config(key: str, value: str) -> None:
    project_root = find_project_root()
    dct, local_config_path = load_config_dict_from_file(project_root, with_path=True)
    if "." not in key:
        raise ValueError(f"The key must have format [section].[key] but is {key}")
    section, key = key.split(".", maxsplit=1)
    if section not in dct:
        dct[section] = {}
    dct[section][key] = value
    with open(local_config_path, "w") as f:
        toml.dump(dct, f)


def load_config_dict_from_file(
    project_root: AbsolutePath, with_path: bool = False
) -> Union[Dict, Tuple[Dict, AbsolutePath]]:
    path_to_config_dir = project_root / ".bohr"
    path_to_config_dir.mkdir(exist_ok=True)
    path_to_local_config = path_to_config_dir / "local.config"
    if not path_to_local_config.exists():
        path_to_local_config.touch()
        gitignore_file(path_to_config_dir, "local.config")
    with open(path_to_local_config) as f:
        dct = toml.load(f)
    return (dct, path_to_local_config) if with_path else dct
