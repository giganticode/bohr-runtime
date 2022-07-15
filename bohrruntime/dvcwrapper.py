import logging
import subprocess
from typing import List, Optional

from dvc.repo import Repo

from bohrruntime.storageengine import StorageEngine

logger = logging.getLogger(__name__)


def status(fs: Optional[StorageEngine] = None) -> str:
    fs = fs or StorageEngine.init()
    command = ["dvc", "status"]
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(command, cwd=fs.fs.getsyspath("."), encoding="utf8")


class DvcRunFailed(Exception):
    pass


def init_dvc(storage_engine: StorageEngine) -> None:
    command = ["dvc", "init"]
    if not storage_engine.fs.exists(".git"):
        command.append("--no-scm")
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(
        command,
        cwd=storage_engine.fs.getsyspath("."),
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
    storage_engine: StorageEngine = None,
) -> None:
    if not storage_engine.fs.exists(".dvc"):
        init_dvc(storage_engine)
    # Repo('.').reproduce() # TODO try this out
    command = ["dvc", "repro"] + (stages or [])
    if pull:
        command.append("--pull")
    if glob:
        command.extend(["--glob", glob])
    if force:
        command.append("--force")
        command.append("-s")
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(
        command, cwd=storage_engine.fs.getsyspath("."), encoding="utf8", shell=False
    )
    proc.communicate()
    if proc.returncode != 0:
        raise DvcRunFailed()


def pull(
    stages: Optional[List[str]] = None,
    storage_engine: Optional[StorageEngine] = None,
) -> None:
    if not storage_engine.fs.exists(".dvc"):
        init_dvc(storage_engine)

    command = ["dvc", "pull"] + (stages or [])
    logger.info(f"Running dvc command: {command}")
    proc = subprocess.Popen(
        command, cwd=storage_engine.fs.getsyspath("."), encoding="utf8", shell=False
    )
    proc.communicate()
