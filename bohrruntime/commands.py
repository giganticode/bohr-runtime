import logging
from pathlib import Path
from typing import Optional, Tuple, Type, Union

import yaml
from bohrapi.core import HeuristicObj
from git import Repo

import bohrruntime.dvcwrapper as pipeline_manager
from bohrruntime.bohrconfigparser import load_workspace
from bohrruntime.datamodel.bohrconfig import BohrConfig
from bohrruntime.heuristics import (
    FileSystemHeuristicLoader,
    HeuristicLoader,
    HeuristicURI,
)
from bohrruntime.pipeline import (
    fetch_heuristics_if_needed,
    get_params,
    get_stage_list,
)
from bohrruntime.dvcwrapper import save_stages_to_pipeline_config
from bohrruntime.storageengine import StorageEngine
from bohrruntime.util.paths import AbsolutePath, create_fs

"""
Implementation of CLI commands, e.g. `bohr repro`
"""

logger = logging.getLogger(__name__)


def clone(url: str, revision: Optional[str] = None) -> None:
    repo_name = url[url.rfind("/") + 1 :]
    if (Path(".") / repo_name).exists():
        raise RuntimeError(f"Directory {repo_name} already exists")
    repo = Repo.clone_from(url, repo_name, depth=1)
    repo.head.reset(revision, index=True, working_tree=True)
    dvc_repo = pipeline_manager.Repo(repo_name)
    dvc_repo.pull()


def repro(
    force: bool = False,
    no_pull: bool = False,
    only_cached_datasets: bool = False,
    workspace: Optional[BohrConfig] = None,
    storage_engine: Optional[StorageEngine] = None,
):

    storage_engine = storage_engine or StorageEngine.init()
    workspace = workspace or load_workspace()
    refresh_pipeline_config(workspace, storage_engine)
    stage = get_stage_list(workspace, storage_engine)
    n_stages = len(stage)
    if force:
        print("Forcing reproduction of all sub-stages ... ")
    for i, stage in enumerate(stage):
        print(f"===========    Executing stage: {stage.summary()} [{i+1}/{n_stages}]")
        pipeline_manager.repro(stage, storage_engine=storage_engine, force=force, no_pull=no_pull, only_cached_datasets=only_cached_datasets)


def refresh_pipeline_config(
    workspace: BohrConfig, storage_engine: StorageEngine
) -> None:
    fetch_heuristics_if_needed(
        workspace.experiments[0].revision,
        AbsolutePath(Path(storage_engine.cloned_bohr_subfs().getsyspath("."))),
    )

    stages = get_stage_list(workspace, storage_engine)
    save_stages_to_pipeline_config(stages, storage_engine)

    params = get_params(workspace, storage_engine)
    with storage_engine.fs.open("bohr.lock", "w") as f:
        f.write(yaml.dump(params))


def status(
    work_space: Optional[BohrConfig] = None, fs: Optional[StorageEngine] = None
) -> str:
    work_space = work_space or load_workspace()
    fs = fs or StorageEngine.init()

    refresh_pipeline_config(work_space, fs)
    return pipeline_manager.status(fs)


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


def push():
    pipeline_manager.Repo().push(remote="write")
