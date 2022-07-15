import logging
from pathlib import Path
from typing import Optional, Tuple, Type, Union

import yaml
from bohrapi.core import HeuristicObj
from git import Repo
from tqdm import tqdm

import bohrruntime.dvcwrapper as dvc
from bohrruntime.bohrconfig import load_workspace
from bohrruntime.datamodel.workspace import Workspace
from bohrruntime.heuristicuri import HeuristicURI
from bohrruntime.pipeline import (
    dvc_config_from_tasks,
    fetch_heuristics_if_needed,
    get_params,
    get_stages_list,
)
from bohrruntime.storageengine import (
    FileSystemHeuristicLoader,
    HeuristicLoader,
    StorageEngine,
)
from bohrruntime.util.paths import AbsolutePath, create_fs

logger = logging.getLogger(__name__)


def clone(url: str, revision: Optional[str] = None) -> None:
    repo_name = url[url.rfind("/") + 1 :]
    if (Path(".") / repo_name).exists():
        raise RuntimeError(f"Directory {repo_name} already exists")
    repo = Repo.clone_from(url, repo_name, depth=1)
    repo.head.reset(revision, index=True, working_tree=True)
    dvc_repo = dvc.Repo(repo_name)
    dvc_repo.pull()


def repro(
    force: bool = False,
    workspace: Optional[Workspace] = None,
    storage_engine: Optional[StorageEngine] = None,
):

    storage_engine = storage_engine or StorageEngine.init()
    workspace = workspace or load_workspace()
    refresh_pipeline_config(workspace, storage_engine)
    commands = get_stages_list(workspace, storage_engine)
    n_commands = len(commands)
    for i, command in enumerate(commands):
        if not isinstance(command, list):
            command = [command]
        print(
            f"===========    Executing stage: {command[0].summary()} [{i}/{n_commands}]"
        )
        for c in tqdm(command):
            repro_stage(c.stage_name(), storage_engine, force=force)


def repro_stage(
    stage_name: str, storage_engine: Optional[StorageEngine] = None, force: bool = False
):
    dvc.repro([stage_name], storage_engine=storage_engine, force=force)


def run_pipeline(
    task: Optional[str] = None,
    force: bool = False,
    workspace: Optional[Workspace] = None,
    storage_engine: Optional[StorageEngine] = None,
    pull: bool = True,
) -> None:
    storage_engine = storage_engine or StorageEngine.init()
    workspace = workspace or load_workspace()

    refresh_pipeline_config(workspace, storage_engine)

    glob = None
    if task:
        # if task not in {exp.task for exp in workspace.experiments}:
        #     raise ValueError(f"Task {task} not found in bohr.json")
        glob = f"{task}*"
    if pull:
        print("Pulling cache from DVC remote...")
        dvc.pull(storage_engine=storage_engine)
    dvc.repro(pull=False, glob=glob, force=force, storage_engine=storage_engine)


def refresh_pipeline_config(
    workspace: Workspace, storage_engine: StorageEngine
) -> None:
    fetch_heuristics_if_needed(
        workspace.experiments[0].revision,
        AbsolutePath(Path(storage_engine.cloned_bohr_subfs().getsyspath("."))),
    )

    stages = get_stages_list(workspace, storage_engine)
    dvc_config = dvc_config_from_tasks(stages)
    with storage_engine.fs.open("dvc.yaml", "w") as f:
        f.write(yaml.dump(dvc_config))

    params = get_params(workspace, storage_engine)
    with storage_engine.fs.open("bohr.lock", "w") as f:
        f.write(yaml.dump(params))


def status(
    work_space: Optional[Workspace] = None, fs: Optional[StorageEngine] = None
) -> str:
    work_space = work_space or load_workspace()
    fs = fs or StorageEngine.init()

    refresh_pipeline_config(work_space, fs)
    return dvc.status(fs)


def load_heuristic_by_name(
    name: str,
    artifact_type: Type = None,
    heuristic_loader: HeuristicLoader = None,
    return_path: bool = False,
) -> Union[HeuristicObj, Tuple[HeuristicObj, HeuristicURI]]:
    heuristic_loader = heuristic_loader or FileSystemHeuristicLoader(create_fs())
    for heuristic_uri, hs in heuristic_loader.load_all_heuristics(
        artifact_type
    ).items():
        for h in hs:
            if h.func.__name__ == name:
                return h if not return_path else (h, heuristic_uri)
    raise ValueError(f"Heuristic {name} does not exist")
