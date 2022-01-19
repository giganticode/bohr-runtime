import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import toml
from bohrapi.core import Dataset, Experiment, Task

from bohrruntime.fs import find_project_root, gitignore_file
from bohrruntime.util.paths import AbsolutePath, RelativePath

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathConfig:
    project_root: AbsolutePath
    heuristics_dir: RelativePath = Path("heuristics")
    cloned_bohr_dir: RelativePath = Path("cloned-bohr")
    runs_dir: RelativePath = Path("runs")
    cached_dataset_dir: RelativePath = Path("cached-datasets")

    @property
    def cloned_bohr(self) -> AbsolutePath:
        return self.project_root / self.cloned_bohr_dir

    @property
    def heuristics(self) -> AbsolutePath:
        return self.cloned_bohr / self.heuristics_dir

    @property
    def runs(self) -> AbsolutePath:
        return self.project_root / self.runs_dir

    @property
    def cached_datasets(self) -> AbsolutePath:
        return self.project_root / self.cached_dataset_dir

    def exp_dir(self, exp: Experiment) -> AbsolutePath:
        return self.runs / exp.task.name / exp.name

    def exp_dataset_dir(self, exp: Experiment, dataset: Dataset) -> AbsolutePath:
        path = self.exp_dir(exp) / dataset.id
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def heuristic_group_dir(
        self, dataset: Dataset, heuristic_group: str
    ) -> AbsolutePath:
        path = (
            self.project_root / "runs" / "__heuristics" / dataset.id / heuristic_group
        )
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    def heuristic_matrix_file(
        self, dataset: Dataset, heuristic_group: str
    ) -> AbsolutePath:
        path = (
            self.project_root / "runs" / "__heuristics" / dataset.id / heuristic_group
        )
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path / "heuristic_matrix.pkl"

    def single_heuristic_metrics(
        self, task: Task, dataset: Dataset, heuristic_group: str
    ) -> AbsolutePath:
        path = (
            self.project_root
            / "runs"
            / "__single_heuristic_metrics"
            / task.name
            / dataset.id
            / heuristic_group
        )
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path / "heuristic_metrics.json"

    @staticmethod
    def deserialize(dct, cls, project_root: AbsolutePath, **kwargs) -> "PathConfig":
        return PathConfig(project_root, **dct)

    @staticmethod
    def load(project_root: Optional[AbsolutePath] = None) -> "PathConfig":
        project_root = project_root or find_project_root()
        return PathConfig(project_root)


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
