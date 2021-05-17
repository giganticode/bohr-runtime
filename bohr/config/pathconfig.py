import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import toml

from bohr.fs import find_project_root, gitignore_file
from bohr.util.paths import AbsolutePath, RelativePath

logger = logging.getLogger(__name__)


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
        config_dict = load_config_dict_from_file(project_root)
        try:
            software_path = config_dict["paths"]["software_path"]
        except KeyError:
            logger.warning(
                f"Value not found in config: software_path, using default value."
            )
            software_path = str(project_root / "software")
        return PathConfig(project_root, Path(software_path))


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
