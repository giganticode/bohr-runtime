import logging
from typing import Optional, Tuple, Type, Union

from bohrapi.core import HeuristicObj, Workspace

import bohrruntime.dvc as dvc
from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import load_workspace
from bohrruntime.heuristics import load_all_heuristics
from bohrruntime.pipeline import write_tasks_to_dvc_file
from bohrruntime.util.paths import AbsolutePath

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
        # if task not in {exp.task for exp in workspace.experiments}:
        #     raise ValueError(f"Task {task} not found in bohr.json")
        glob = f"{task}*"
    if pull:
        print("Pulling cache from DVC remote...")
        dvc.pull(fs=fs)
    dvc.repro(pull=False, glob=glob, force=force, fs=fs)


def refresh(workspace: Workspace, fs: BohrFileSystem) -> None:
    write_tasks_to_dvc_file(workspace, fs)


def status(
    work_space: Optional[Workspace] = None, fs: Optional[BohrFileSystem] = None
) -> str:
    work_space = work_space or load_workspace()
    fs = fs or BohrFileSystem.init()

    refresh(work_space, fs)
    return dvc.status(fs)


def load_heuristic_by_name(
    name: str,
    artifact_type: Type,
    heuristics_path: AbsolutePath,
    return_path: bool = False,
) -> Union[HeuristicObj, Tuple[HeuristicObj, str]]:
    for path, hs in load_all_heuristics(artifact_type, heuristics_path).items():
        for h in hs:
            if h.func.__name__ == name:
                return h if not return_path else (h, str(path))
    raise ValueError(f"Heuristic {name} does not exist")
