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


class DvcRunFailed(Exception):
    pass


def init_dvc(fs: BohrFileSystem) -> None:
    command = ["dvc", "init"]
    if not (fs.root / ".git").exists():
        command.append("--no-scm")
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(
        command,
        cwd=fs.root,
        encoding="utf8",
        shell=False,
    )
    proc.communicate()
    if proc.returncode != 0:
        raise DvcRunFailed()


def repro(
    stages: Optional[List[str]] = None,
    pull: bool = False,
    glob: Optional[str] = None,
    force: bool = False,
    fs: BohrFileSystem = None,
) -> None:
    if not (fs.root / ".dvc").exists():
        init_dvc(fs)

    command = ["dvc", "repro"] + (stages or [])
    if pull:
        command.append("--pull")
    if glob:
        command.extend(["--glob", glob])
    if force:
        command.append("--force")
        command.append("-s")
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(command, cwd=fs.root, encoding="utf8", shell=False)
    proc.communicate()
    if proc.returncode != 0:
        raise DvcRunFailed()


def pull(
    stages: Optional[List[str]] = None,
    fs: Optional[BohrFileSystem] = None,
) -> None:
    if not (fs.root / ".dvc").exists():
        init_dvc(fs)

    command = ["dvc", "pull"] + (stages or [])
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(command, cwd=fs.root, encoding="utf8", shell=False)
    proc.communicate()
