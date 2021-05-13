import logging
from pathlib import Path
from typing import Optional

import bohr.dvcwrapper as dvc
from bohr.config import Config, load_config
from bohr.datamodel import RelativePath
from bohr.lock import bohr_up_to_date, update_lock
from bohr.pathconfig import PathConfig, load_path_config
from bohr.pipeline.dvc import add_all_tasks_to_dvc_pipeline, load_transient_stages

logger = logging.getLogger(__name__)


class BohrDatasetNotFound(Exception):
    pass


def repro(
    task: Optional[str], only_transient: bool, force: bool, config: Config
) -> None:
    """
    # >>> import tempfile
    # >>> with tempfile.TemporaryDirectory() as tmpdirname:
    # ...     with open(Path(tmpdirname) / 'bohr.json', 'w') as f:
    # ...         print(f.write('{"bohr_framework_version": "0.3.9-rc", "tasks": {}, "datasets": {}}'))
    # ...     get_dvc_commands_to_repro(None, False, load_config(Path(tmpdirname)))
    """
    paths_to_pull = [str(d.path_dist) for d in config.datasets.values()]
    if len(paths_to_pull) > 0:
        logger.info(dvc.pull(paths_to_pull))

    # TODO run only task-related transient stages if task is passed:
    transient_stages = load_transient_stages(config.paths)
    if len(transient_stages) > 0:
        logger.info(dvc.repro(transient_stages, force=force, path_config=config.paths))

    if not only_transient:
        glob = None
        if task:
            if task not in config.tasks:
                raise ValueError(f"Task {task} not found in bohr.json")
            glob = f"{task}_*"
        logger.info(
            dvc.repro(pull=True, glob=glob, force=force, path_config=config.paths)
        )


def refresh(config: Optional[Config] = None) -> None:
    config = config or load_config()
    (config.paths.project_root / "dvc.yaml").unlink(missing_ok=True)
    add_all_tasks_to_dvc_pipeline(config)
    update_lock(config.paths)


def refresh_if_necessary(config: Optional[Config] = None) -> None:
    config = config or load_config()
    if not bohr_up_to_date(config.paths):
        logger.info("There are changes to the bohr config. Refreshing the workspace...")
        refresh(config)
    else:
        logger.info("Bohr config hasn't changed.")


def status() -> str:
    config = load_config()
    config.dump(config.paths.project_root)
    refresh_if_necessary(config)
    return dvc.status(config.paths)


def pull(target: str, config: Optional[Config] = None) -> RelativePath:
    config = config or load_config()
    if target in config.datasets:
        path = config.paths.labeled_data_dir / f"{target}.labeled.csv"
        logger.info(dvc.pull([str(path)]))
        return path
    else:
        raise BohrDatasetNotFound(
            f"Dataset {target} not found! Available datasets: {list(config.datasets.keys())}"
        )
