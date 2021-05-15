import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import bohr.dvcwrapper as dvc
from bohr.config import Config, get_mapper_by_name, get_preprocessed_path, load_config
from bohr.datamodel import Dataset, RelativePath, Task, relative_to_safe
from bohr.lock import bohr_up_to_date, update_lock
from bohr.pipeline.dvc import add_all_tasks_to_dvc_pipeline, load_transient_stages
from bohr.templates.dataloaders.from_csv import CsvDatasetLoader

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


def extract_preprocessor_from_file_name(file_name: str) -> Tuple[str, str]:
    """
    >>> extract_preprocessor_from_file_name("dataset.csv.7z")
    ('dataset.csv', '7z')
    >>> extract_preprocessor_from_file_name("dataset.zip")
    ('dataset', 'zip')
    >>> extract_preprocessor_from_file_name("dataset.csv")
    ('dataset.csv', 'copy')
    """
    *prefix, suffix = file_name.split(".")
    if suffix in ["zip", "7z"]:
        return ".".join(prefix), suffix
    else:
        return file_name, "copy"


def extract_format_from_file_name(file_name: str) -> Tuple[str, str]:
    *prefix, format = file_name.split(".")
    if format == "csv":
        return ".".join(prefix), format
    else:
        raise ValueError(f"Unrecognized file format: {format}")


def add(
    path: Path,
    artifact: str,
    name: Optional[str] = None,
    author: Optional[str] = None,
    description: Optional[str] = "",
    format: Optional[str] = None,
    preprocessor: Optional[str] = None,
    config: Optional[Config] = None,
) -> Dataset:
    config = config or load_config()
    destination_path = config.paths.downloaded_data / path.name
    logger.info(f"Copying {path.name} to {destination_path} ...")
    shutil.copy(path, destination_path)
    dvc_output = dvc.add(destination_path, config.paths.project_root)
    logger.info(dvc_output)
    file_name = path.name
    if preprocessor is None:
        file_name, preprocessor = extract_preprocessor_from_file_name(file_name)
    if format is None:
        file_name, format = extract_format_from_file_name(file_name)
    dataset_name = name or file_name
    if dataset_name in config.datasets:
        message = f"Dataset with name {dataset_name} already exists."
        if name is None:
            message += (
                "\nAre you trying to add the same dataset twice?\n"
                "If not, please specifying the `name` parameter explicitly."
            )
        raise ValueError(message)
    mapper = get_mapper_by_name(artifact)
    path_preprocessed: RelativePath = get_preprocessed_path(
        None,
        relative_to_safe(destination_path, config.paths.downloaded_data),
        config.paths.data_dir,
        preprocessor,
    )
    dataset = Dataset(
        dataset_name,
        author,
        description,
        path_preprocessed,
        config.paths.data_dir / path.name,
        CsvDatasetLoader(path_preprocessed, mapper()),
        preprocessor,
    )
    config.datasets[dataset.name] = dataset
    config.dump(config.paths.project_root)
    return dataset


def add_dataset(
    task: Task, dataset: Dataset, config: Optional[Config] = None
) -> Dataset:
    is_test_set = dataset.is_column_present(task.label_column_name)
    logger.info(
        f'Adding dataset {dataset.name} as a {"test" if is_test_set else "train"} set'
    )
    task.add_dataset(dataset, is_test_set)
    config.dump(config.paths.project_root)
    return dataset
