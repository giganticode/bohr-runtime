import logging

from bohr.config import load_config
from bohr.lock import bohr_up_to_date, update_lock
from bohr.pathconfig import PathConfig
from bohr.pipeline.dvc import add_all_tasks_to_dvc_pipeline

logger = logging.getLogger(__name__)


def refresh() -> None:
    config = load_config()
    (config.paths.project_root / "dvc.yaml").unlink(missing_ok=True)
    add_all_tasks_to_dvc_pipeline(config)
    update_lock(config.paths)


def refresh_if_necessary(path_config: PathConfig) -> None:
    if not bohr_up_to_date(path_config):
        logger.info("There are changes to the bohr config. Refreshing the workspace...")
        refresh()
    else:
        logger.info("Bohr config hasn't changed.")
