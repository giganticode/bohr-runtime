from dataclasses import dataclass
from typing import Optional

from bohr.config.pathconfig import PathConfig, load_config_dict_from_file
from bohr.fs import find_project_root
from bohr.util.paths import AbsolutePath


@dataclass(frozen=True)
class AppConfig:
    verbose: bool
    paths: PathConfig

    @staticmethod
    def load(project_root: Optional[AbsolutePath] = None) -> "AppConfig":
        project_root = project_root or find_project_root()
        config_dict = load_config_dict_from_file(project_root)
        try:
            verbose_str = config_dict["core"]["verbose"]
            verbose = verbose_str == "true" or verbose_str == "True"
        except KeyError:
            verbose = False
        return AppConfig(verbose, PathConfig.load())
