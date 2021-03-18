from bohr.config import load_config
from bohr.pipeline.dvc import add_all_tasks_to_dvc_pipeline


def refresh() -> None:
    config = load_config()
    (config.project_root / "dvc.yaml").unlink(missing_ok=True)
    add_all_tasks_to_dvc_pipeline(config)
