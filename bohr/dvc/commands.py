import logging
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from bohr.config.pathconfig import PathConfig
from bohr.fs import find_project_root
from bohr.util.paths import AbsolutePath

logger = logging.getLogger(__name__)


def status(path_config: Optional[PathConfig] = None) -> str:
    path_config = path_config or PathConfig.load()
    command = ["dvc", "status"]
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(
        command, cwd=path_config.project_root, encoding="utf8"
    )


def add(path: AbsolutePath, project_root: Optional[Path] = None) -> str:
    project_root = project_root or find_project_root()
    command = ["dvc", "add", path.name]
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(
        command, cwd=project_root / path.parent, encoding="utf8"
    )


def pull(paths: List[str], path_config: Optional[PathConfig] = None) -> str:
    if not isinstance(paths, Iterable) or len(paths) < 1:
        raise ValueError(
            f"At least one path to be pulled needs to be specified but is passed: {paths}"
        )
    path_config = path_config or PathConfig.load()
    command = ["dvc", "pull"] + paths
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(
        command, cwd=path_config.project_root, encoding="utf8"
    )


def repro(
    stages: Optional[List[str]] = None,
    pull: bool = False,
    glob: Optional[str] = None,
    force: bool = False,
    path_config: Optional[PathConfig] = None,
) -> str:
    command = ["dvc", "repro"] + (stages or [])
    if pull:
        command.append("--pull")
    if glob:
        command.extend(["--glob", glob])
    if force:
        command.append("--force")
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(
        command, cwd=path_config.project_root, encoding="utf8"
    )
