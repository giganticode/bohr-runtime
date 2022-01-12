import logging
from typing import Optional

from bohrapi.core import Workspace

import bohrruntime.dvc.commands as dvc
from bohrruntime.config.pathconfig import PathConfig
from bohrruntime.core import load_workspace
from bohrruntime.dvc.stages import write_tasks_to_dvc_file

logger = logging.getLogger(__name__)


class BohrDatasetNotFound(Exception):
    pass


class NoTasksFound(Exception):
    pass


def repro(
    task: Optional[str] = None,
    force: bool = False,
    workspace: Optional[Workspace] = None,
    path_config: Optional[PathConfig] = None,
    pull: bool = True,
) -> None:
    path_config = path_config or PathConfig.load()
    workspace = workspace or load_workspace()

    refresh(workspace, path_config)

    glob = None
    if task:
        if task not in workspace.tasks:
            raise ValueError(f"Task {task} not found in bohr.json")
        glob = f"{task}_*"
    if pull:
        print("Pulling cache from DVC remote...")
        dvc.pull(path_config=path_config)
    print(dvc.repro(pull=False, glob=glob, force=force, path_config=path_config))


def refresh(workspace: Workspace, path_config: PathConfig) -> None:
    write_tasks_to_dvc_file(workspace, path_config)


def status(
    work_space: Optional[Workspace] = None, path_config: Optional[PathConfig] = None
) -> str:
    work_space = work_space or load_workspace()
    path_config = path_config or PathConfig.load()

    refresh(work_space, path_config)
    return dvc.status(path_config)
