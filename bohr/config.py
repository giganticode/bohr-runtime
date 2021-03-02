import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type

import jsons
import toml

from bohr import version
from bohr.artifacts.core import Artifact
from bohr.datamodel import DatasetLoader, Task


def find_project_root() -> Path:
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


@dataclass(frozen=True)
class PathsConfig:
    """
    >>> jsons.loads('{}', PathsConfig, project_root=Path('/'), software_path='/software')
    PathsConfig(project_root=PosixPath('/'), software_path=PosixPath('/software'), metrics_dir='metrics', \
generated_dir='generated', heuristics_dir='heuristics', dataset_dir='dataloaders', labeled_data_dir='labeled-data', \
data_dir='data', labels_dir='labels')
    """

    project_root: Path
    software_path: Path
    metrics_dir: str = "metrics"
    generated_dir: str = "generated"
    heuristics_dir: str = "heuristics"
    dataset_dir: str = "dataloaders"
    labeled_data_dir: str = "labeled-data"
    data_dir: str = "data"
    labels_dir: str = "labels"

    @property
    def metrics(self) -> Path:
        return self.project_root / self.metrics_dir

    @property
    def generated(self) -> Path:
        return self.project_root / self.generated_dir

    @property
    def heuristics(self) -> Path:
        return self.project_root / self.heuristics_dir

    @property
    def dataset(self) -> Path:
        return self.project_root / self.dataset_dir

    @property
    def labeled_data(self) -> Path:
        return self.project_root / self.labeled_data_dir

    @property
    def data(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def labels(self) -> Path:
        return self.project_root / self.labels_dir

    @staticmethod
    def deserialize(
        dct, cls, project_root: Path, software_path: str, **kwargs
    ) -> "PathsConfig":
        return PathsConfig(project_root, Path(software_path), **dct)


def gitignore_file(dir: Path, filename: str):
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


def add_to_local_config(section: str, key: str, value: str) -> None:
    project_root = find_project_root()
    local_dir_path = project_root / ".bohr"
    local_dir_path.mkdir()
    LOCAL_CONFIG_FILE = "local.config"
    local_config_path = local_dir_path / LOCAL_CONFIG_FILE
    if local_config_path.exists():
        with open(local_config_path) as f:
            dct = toml.load(f)
    else:
        local_config_path.touch()
        dct = {}
    if section not in dct:
        dct[section] = {}
    dct[section][key] = value
    with open(local_config_path, "w") as f:
        toml.dump(dct, f)
    gitignore_file(local_dir_path, LOCAL_CONFIG_FILE)


@dataclass(frozen=True)
class Config:
    """
    >>> jsons.loads('{"bohr_framework_version": 0.1, "tasks": {}}', Config, project_root=Path('/'), \
software_path='/software')
    Config(project_root=PosixPath('/'), bohr_framework_version=0.1, tasks={}, \
paths=PathsConfig(project_root=PosixPath('/'), software_path=PosixPath('/software'), metrics_dir='metrics', \
generated_dir='generated', heuristics_dir='heuristics', dataset_dir='dataloaders', \
labeled_data_dir='labeled-data', data_dir='data', labels_dir='labels'))
    """

    project_root: Path
    bohr_framework_version: str
    tasks: Dict[str, Task]
    paths: PathsConfig

    @staticmethod
    def load(project_root: Path) -> "Config":
        with open(project_root / ".bohr" / "local.config") as f:
            software_path = toml.load(f)["core"]["software_path"]
        with open(project_root / "bohr.json") as f:
            return jsons.loads(
                f.read(), Config, project_root=project_root, software_path=software_path
            )


def get_version_from_config() -> str:
    config_file = find_project_root() / "bohr.json"
    with open(config_file, "r") as f:
        dct = jsons.loads(f.read())
        return str(dct["bohr-framework-version"])


def load_config() -> Config:
    project_root = find_project_root()
    config = Config.load(project_root)

    version_installed = version()
    if str(config.bohr_framework_version) != version_installed:
        raise EnvironmentError(
            f"Version of bohr framework from config: {config.bohr_framework_version}. "
            f"Version of bohr installed: {version_installed}"
        )
    return config


def get_dataset_loader(name: str) -> DatasetLoader:
    module = None
    try:
        module = importlib.import_module(name)
        all = module.__dict__["__all__"]
        if len(all) != 1 or not isinstance(all[0], DatasetLoader):
            raise SyntaxError(
                f"{DatasetLoader} object should be specified in __all__ list"
            )
        return all[0]
    except KeyError as e:
        raise ValueError(f"__all__ is not defined in module {module}")
    except ModuleNotFoundError as e:
        raise ValueError(f"Dataset {name} not defined.") from e


def load_artifact_by_name(artifact_name: str) -> Type["Artifact"]:
    *path, name = artifact_name.split(".")
    try:
        module = importlib.import_module(".".join(path))
    except ModuleNotFoundError as e:
        raise ValueError(f'Module {".".join(path)} not defined.') from e

    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ValueError(f"Artifact {name} not found in module {module}") from e


def deserialize_task(
    dct: Dict[str, Any],
    cls,
    project_root: Path,
    task_name: str,
    **kwargs,
) -> "Task":
    # """
    # >>> jsons.loads('{"top_artifact": "artifacts.commit.Commit", "test_dataset_names": [], "train_dataset_names": []}', Task, project_root='/', task_name="x")
    # """
    test_datasets = {name: get_dataset_loader(name) for name in dct["test_datasets"]}
    train_datasets = {name: get_dataset_loader(name) for name in dct["train_datasets"]}
    return Task(
        task_name,
        load_artifact_by_name(dct["top_artifact"]),
        dct["label_categories"],
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        label_column_name=dct["label_column_name"],
        project_root=project_root,
    )


def deserialize_config(
    dct, cls, project_root: Path, software_path: str, **kwargs
) -> Config:
    if not isinstance(project_root, Path):
        raise ValueError(
            f"Project root must be a path object but is: {type(project_root)}"
        )

    paths_json = dct["paths"] if "paths" in dct else {}
    paths: PathsConfig = jsons.load(
        paths_json, PathsConfig, project_root=project_root, software_path=software_path
    )
    tasks = dict()
    for task_name, task_json in dct["tasks"].items():
        tasks[task_name] = jsons.load(
            task_json, Task, project_root=project_root, task_name=task_name
        )
    return Config(
        project_root,
        dct["bohr_framework_version"],
        tasks,
        paths,
    )


jsons.set_deserializer(deserialize_task, Task)
jsons.set_deserializer(deserialize_config, Config)
jsons.set_deserializer(PathsConfig.deserialize, PathsConfig)
