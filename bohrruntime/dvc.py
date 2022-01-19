import logging
import subprocess
from typing import List, Optional

from bohrruntime.config.pathconfig import PathConfig

logger = logging.getLogger(__name__)


def status(path_config: Optional[PathConfig] = None) -> str:
    path_config = path_config or PathConfig.load()
    command = ["dvc", "status"]
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
) -> None:
    command = ["dvc", "repro"] + (stages or [])
    if pull:
        command.append("--pull")
    if glob:
        command.extend(["--glob", glob])
    if force:
        command.append("--force")
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(
        command, cwd=path_config.project_root, encoding="utf8", shell=False
    )
    proc.communicate()


def pull(
    stages: Optional[List[str]] = None,
    path_config: Optional[PathConfig] = None,
) -> None:
    command = ["dvc", "pull"] + (stages or [])
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(
        command, cwd=path_config.project_root, encoding="utf8", shell=False
    )
    proc.communicate()
