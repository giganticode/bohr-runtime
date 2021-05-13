import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import toml

from bohr.datamodel import AbsolutePath, RelativePath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    verbose: bool = False

    @staticmethod
    def load(project_root: Optional[AbsolutePath] = None) -> "AppConfig":
        project_root = project_root or find_project_root()
        config_dict = load_config(project_root)
        try:
            verbose_str = config_dict["core"]["verbose"]
            verbose = verbose_str == "true" or verbose_str == "True"
        except KeyError:
            verbose = False
        return AppConfig(verbose)


@dataclass(frozen=True)
class PathConfig:
    project_root: AbsolutePath
    software_path: AbsolutePath
    metrics_dir: RelativePath = Path("metrics")
    generated_dir: RelativePath = Path("generated")
    heuristics_dir: RelativePath = Path("heuristics")
    dataset_dir: RelativePath = Path("dataloaders")
    labeled_data_dir: RelativePath = Path("labeled-datasets")
    data_dir: RelativePath = Path("data")
    labels_dir: RelativePath = Path("labels")
    manual_stages_dir: RelativePath = Path("manual_stages")
    downloaded_data_dir: RelativePath = Path("downloaded-data")

    @property
    def metrics(self) -> AbsolutePath:
        return self.project_root / self.metrics_dir

    @property
    def generated(self) -> AbsolutePath:
        return self.project_root / self.generated_dir

    @property
    def heuristics(self) -> AbsolutePath:
        return self.project_root / self.heuristics_dir

    @property
    def dataset(self) -> AbsolutePath:
        return self.project_root / self.dataset_dir

    @property
    def labeled_data(self) -> AbsolutePath:
        return self.project_root / self.labeled_data_dir

    @property
    def data(self) -> AbsolutePath:
        return self.project_root / self.data_dir

    @property
    def downloaded_data(self) -> AbsolutePath:
        return self.project_root / self.downloaded_data_dir

    @property
    def labels(self) -> AbsolutePath:
        return self.project_root / self.labels_dir

    @property
    def manual_stages(self) -> AbsolutePath:
        return self.project_root / self.manual_stages_dir

    @staticmethod
    def deserialize(
        dct, cls, project_root: AbsolutePath, software_path: str, **kwargs
    ) -> "PathConfig":
        return PathConfig(project_root, Path(software_path), **dct)

    @staticmethod
    def load(project_root: Optional[AbsolutePath] = None) -> "PathConfig":
        project_root = project_root or find_project_root()
        config_dict = load_config(project_root)
        try:
            software_path = config_dict["paths"]["software_path"]
        except KeyError:
            logger.warning(
                f"Value not found in config: software_path, using default value."
            )
            software_path = str(project_root / "software")
        return PathConfig(project_root, Path(software_path))


def find_project_root() -> AbsolutePath:
    path = Path(".").resolve()
    current_path = path
    while True:
        lst = list(current_path.glob("bohr.json"))
        if len(lst) > 0 and lst[0].is_file():
            return current_path
        elif current_path == Path("/"):
            raise ValueError(
                f"Not a bohr directory: {path}. "
                f"Bohr config dir is not found in this or any parent directory"
            )
        else:
            current_path = current_path.parent


def gitignore_file(dir: AbsolutePath, filename: str):
    """
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     gitignore_file(Path(tmpdirname), 'file')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n']
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     (Path(tmpdirname) / '.gitignore').touch()
    ...     gitignore_file(Path(tmpdirname), 'file')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n']
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     with open(Path(tmpdirname) / '.gitignore', 'w') as f:
    ...         n = f.write("file\\n")
    ...     gitignore_file(Path(tmpdirname), 'file')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n']
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     with open(Path(tmpdirname) / '.gitignore', 'w') as f:
    ...         n = f.write('file\\n')
    ...     gitignore_file(Path(tmpdirname), 'file2')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n', 'file2\\n']
    """
    path_to_gitignore = dir / ".gitignore"
    path_to_gitignore.touch(exist_ok=True)
    with open(path_to_gitignore, "r") as f:
        lines = list(map(lambda l: l.rstrip("\n"), f.readlines()))
        if filename not in lines:
            with open(path_to_gitignore, "a") as a:
                a.write(f"{filename}\n")


def add_to_local_config(key: str, value: str) -> None:
    project_root = find_project_root()
    dct, local_config_path = load_config(project_root, with_path=True)
    if "." not in key:
        raise ValueError(f"The key must have format [section].[key] but is {key}")
    section, key = key.split(".", maxsplit=1)
    if section not in dct:
        dct[section] = {}
    dct[section][key] = value
    with open(local_config_path, "w") as f:
        toml.dump(dct, f)


def load_config(
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


def load_path_config(project_root: Optional[AbsolutePath] = None) -> PathConfig:
    return PathConfig.load(project_root)
