import logging
import subprocess
from typing import Iterable, List, Optional

from bohr.pathconfig import PathConfig

logger = logging.getLogger(__name__)


def status(path_config: Optional[PathConfig] = None) -> str:
    path_config = path_config or PathConfig.load()
    command = ["dvc", "status"]
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(
        command, cwd=path_config.project_root, encoding="utf8"
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
