import logging
from typing import Optional

from bohrapi.core import Workspace

import bohrruntime.dvc as dvc
from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import load_workspace
from bohrruntime.pipeline import write_tasks_to_dvc_file

logger = logging.getLogger(__name__)


class BohrDatasetNotFound(Exception):
    pass


class NoTasksFound(Exception):
    pass


def repro(
    task: Optional[str] = None,
    force: bool = False,
    workspace: Optional[Workspace] = None,
    fs: Optional[BohrFileSystem] = None,
    pull: bool = True,
) -> None:
    fs = fs or BohrFileSystem.init()
    workspace = workspace or load_workspace()

    refresh(workspace, fs)

    glob = None
    if task:
        if task not in workspace.tasks:
            raise ValueError(f"Task {task} not found in bohr.json")
        glob = f"{task}_*"
    if pull:
        print("Pulling cache from DVC remote...")
        dvc.pull(fs=fs)
    print(dvc.repro(pull=False, glob=glob, force=force, fs=fs))


def refresh(workspace: Workspace, fs: BohrFileSystem) -> None:
    write_tasks_to_dvc_file(workspace, fs)


def status(
    work_space: Optional[Workspace] = None, fs: Optional[BohrFileSystem] = None
) -> str:
    work_space = work_space or load_workspace()
    fs = fs or BohrFileSystem.init()

    refresh(work_space, fs)
    return dvc.status(fs)
