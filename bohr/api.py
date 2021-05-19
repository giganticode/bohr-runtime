import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import bohr.dvc.commands as dvc
from bohr.collection.artifacts import artifact_map
from bohr.collection.dataloaders.from_csv import CsvDatasetLoader
from bohr.collection.datamappers import default_mappers
from bohr.config.pathconfig import PathConfig
from bohr.datamodel.bohrrepo import BohrRepo, load_bohr_repo
from bohr.datamodel.dataset import Dataset
from bohr.datamodel.task import Task
from bohr.dvc.stages import add_all_tasks_to_dvc_pipeline, load_transient_stages
from bohr.fs import get_preprocessed_path
from bohr.util.paths import RelativePath, load_class_by_full_path, relative_to_safe
from bohr.workspace import bohr_up_to_date, update_lock

logger = logging.getLogger(__name__)


class BohrDatasetNotFound(Exception):
    pass


def repro(
    task: Optional[str],
    only_transient: bool = False,
    force: bool = False,
    bohr_repo: Optional[BohrRepo] = None,
    path_config: Optional[PathConfig] = None,
) -> None:
    """
    # >>> import tempfile
    # >>> with tempfile.TemporaryDirectory() as tmpdirname:
    # ...     with open(Path(tmpdirname) / 'bohr.json', 'w') as f:
    # ...         print(f.write('{"bohr_framework_version": "0.3.9-rc", "tasks": {}, "datasets": {}}'))
    # ...     get_dvc_commands_to_repro(None, False, load_config(Path(tmpdirname)))
    """
    path_config = path_config or PathConfig.load()
    bohr_repo = bohr_repo or load_bohr_repo(path_config.project_root)

    refresh_if_necessary(path_config)

    paths_to_pull = [str(d.path_dist) for d in bohr_repo.datasets.values()]
    if len(paths_to_pull) > 0:
        logger.info(dvc.pull(paths_to_pull))

    # TODO run only task-related transient stages if task is passed:
    transient_stages = load_transient_stages(path_config)
    if len(transient_stages) > 0:
        logger.info(dvc.repro(transient_stages, force=force, path_config=path_config))

    if not only_transient:
        glob = None
        if task:
            if task not in bohr_repo.tasks:
                raise ValueError(f"Task {task} not found in bohr.json")
            glob = f"{task}_*"
        logger.info(
            dvc.repro(pull=True, glob=glob, force=force, path_config=path_config)
        )


def refresh(path_config: Optional[PathConfig] = None) -> None:
    path_config = path_config or PathConfig.load()
    (path_config.project_root / "dvc.yaml").unlink(missing_ok=True)
    add_all_tasks_to_dvc_pipeline()
    update_lock(path_config)


def refresh_if_necessary(path_config: Optional[PathConfig] = None) -> None:
    path_config = path_config or PathConfig.load()
    if not bohr_up_to_date(path_config):
        logger.info("There are changes to the bohr config. Refreshing the workspace...")
        refresh(path_config)
    else:
        logger.info("Bohr config hasn't changed.")


def status(
    bohr_repo: Optional[BohrRepo] = None, path_config: Optional[PathConfig] = None
) -> str:
    path_config = path_config or PathConfig.load()
    bohr_repo = bohr_repo or load_bohr_repo(path_config.project_root)
    bohr_repo.dump(path_config.project_root)
    refresh_if_necessary(path_config)
    return dvc.status(path_config)


def pull(
    target: str,
    bohr_repo: Optional[BohrRepo] = None,
    path_config: Optional[PathConfig] = None,
) -> RelativePath:
    path_config = path_config or PathConfig.load()
    bohr_repo = bohr_repo or load_bohr_repo(path_config.project_root)

    if target in bohr_repo.datasets:
        path = path_config.labeled_data_dir / f"{target}.labeled.csv"
        logger.info(dvc.pull([str(path)]))
        return path
    else:
        raise BohrDatasetNotFound(
            f"Dataset {target} not found! Available datasets: {list(bohr_repo.datasets.keys())}"
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
    bohr_repo: Optional[BohrRepo] = None,
    path_config: Optional[PathConfig] = None,
) -> Dataset:
    path_config = path_config or PathConfig.load()
    bohr_repo = bohr_repo or load_bohr_repo(path_config.project_root)
    destination_path = path_config.downloaded_data / path.name
    logger.info(f"Copying {path.name} to {destination_path} ...")
    shutil.copy(path, destination_path)
    dvc_output = dvc.add(destination_path, path_config.project_root)
    logger.info(dvc_output)
    file_name = path.name
    if preprocessor is None:
        file_name, preprocessor = extract_preprocessor_from_file_name(file_name)
    if format is None:
        file_name, format = extract_format_from_file_name(file_name)
    dataset_name = name or file_name
    if dataset_name in bohr_repo.datasets:
        message = f"Dataset with name {dataset_name} already exists."
        if name is None:
            message += (
                "\nAre you trying to add the same dataset twice?\n"
                "If not, please specifying the `name` parameter explicitly."
            )
        raise ValueError(message)
    try:
        mapper = default_mappers[artifact_map[artifact]]
    except KeyError:
        mapper = load_class_by_full_path(artifact)
    path_preprocessed: RelativePath = get_preprocessed_path(
        None,
        relative_to_safe(destination_path, path_config.downloaded_data),
        path_config.data_dir,
        preprocessor,
    )
    dataset = Dataset(
        dataset_name,
        author,
        description,
        path_preprocessed,
        path_config.data_dir / path.name,
        CsvDatasetLoader(path_preprocessed, mapper()),
        preprocessor,
    )
    bohr_repo.datasets[dataset.name] = dataset
    bohr_repo.dump(path_config.project_root)
    return dataset


def add_dataset(
    task: Task,
    dataset: Dataset,
    bohr_repo: Optional[BohrRepo] = None,
    path_config: Optional[PathConfig] = None,
) -> Dataset:
    path_config = path_config or PathConfig.load()
    bohr_repo = bohr_repo or load_bohr_repo(path_config.project_root)

    is_test_set = dataset.is_column_present(task.label_column_name)
    logger.info(
        f'Adding dataset {dataset.name} as a {"test" if is_test_set else "train"} set'
    )
    task.add_dataset(dataset, is_test_set)
    bohr_repo.dump(path_config.project_root)
    return dataset
