import logging
import subprocess
from typing import List, Optional

from bohrruntime.bohrfs import BohrFileSystem

logger = logging.getLogger(__name__)


def status(fs: Optional[BohrFileSystem] = None) -> str:
    fs = fs or BohrFileSystem.init()
    command = ["dvc", "status"]
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(command, cwd=fs.root, encoding="utf8")


def repro(
    stages: Optional[List[str]] = None,
    pull: bool = False,
    glob: Optional[str] = None,
    force: bool = False,
    fs: Optional[BohrFileSystem] = None,
) -> None:
    command = ["dvc", "repro"] + (stages or [])
    if pull:
        command.append("--pull")
    if glob:
        command.extend(["--glob", glob])
    if force:
        command.append("--force")
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(command, cwd=fs.root, encoding="utf8", shell=False)
    proc.communicate()


def pull(
    stages: Optional[List[str]] = None,
    fs: Optional[BohrFileSystem] = None,
) -> None:
    command = ["dvc", "pull"] + (stages or [])
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(command, cwd=fs.root, encoding="utf8", shell=False)
    proc.communicate()
