import logging

from bohr.config import Config, load_config
from bohr.lock import bohr_up_to_date, update_lock
from bohr.pipeline.dvc import add_all_tasks_to_dvc_pipeline

logger = logging.getLogger(__name__)


def refresh() -> None:
    config = load_config()
    (config.project_root / "dvc.yaml").unlink(missing_ok=True)
    add_all_tasks_to_dvc_pipeline(config)


def refresh_if_necessary(config: Config) -> None:
    if not bohr_up_to_date(config):
        logger.info("There are changes to the bohr config. Refreshing ...")
        refresh()
        update_lock(config)
