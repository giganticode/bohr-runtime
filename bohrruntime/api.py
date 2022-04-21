import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Type, Union

import requests
from bohrapi.core import HeuristicObj, Workspace
from git import Repo

import bohrruntime.dvcwrapper as dvc
from bohrruntime import __version__
from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import BOHR_ORGANIZATION, BOHR_REPO_NAME, load_workspace
from bohrruntime.heuristics import load_all_heuristics
from bohrruntime.pipeline import write_tasks_to_dvc_file
from bohrruntime.util.paths import AbsolutePath

logger = logging.getLogger(__name__)


class BohrDatasetNotFound(Exception):
    pass


class NoTasksFound(Exception):
    pass


def clone(task: str, path: str, revision: Optional[str] = None) -> None:
    """
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdirname: # doctest: +ELLIPSIS
    ...     clone('bugginess', Path(tmpdirname) / 'repo')
    ...     clone('bugginess', Path(tmpdirname) / 'repo')
    Traceback (most recent call last):
    ...
    RuntimeError: Path ... already exists and not empty.
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     clone('non-existent-task', Path(tmpdirname) / 'repo')
    Traceback (most recent call last):
    ...
    ValueError: Unknown task: non-existent-task
    >>> with tempfile.TemporaryDirectory() as tmpdirname: # doctest: +ELLIPSIS
    ...     clone('bugginess', Path(tmpdirname) / 'repo', '7711f36d5333aa8cd5d110a91fd58cfd2e7ee0d5')
    """
    if Path(path).exists() and len(os.listdir(path)) != 0:
        raise RuntimeError(f"Path {path} already exists and not empty.")
    elif not Path(path).exists():
        Path(path).mkdir()

    raw_task_config = f"https://raw.githubusercontent.com/{BOHR_ORGANIZATION}/{BOHR_REPO_NAME}/master/tasks.json"
    response = requests.get(raw_task_config)
    if response.status_code == 200:
        text = response.text
        tasks_json = json.loads(text)

        if task not in tasks_json:
            raise ValueError(f"Unknown task: {task}")

        available_revisions = next(iter(tasks_json[task].values()))
        if revision:
            if revision not in available_revisions:
                raise ValueError(f"Unknown revision: {revision}")
        else:
            revision = available_revisions[-1]

        github_repo = next(iter(tasks_json[task].keys()))

        repo_url = f"https://github.com/{github_repo}"
        repo = Repo.clone_from(repo_url, path, depth=1)
        repo.head.reset(revision, index=True, working_tree=True)
    elif response.status_code == 404:
        raise AssertionError(f"Could not find tasks config: {raw_task_config}")
    else:
        raise RuntimeError(f"Could not load task config:\n\n {response.text}")


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
