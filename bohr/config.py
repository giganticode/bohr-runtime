import importlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type

import jsons

from bohr import version
from bohr.datamodel import (
    ArtifactType,
    Dataset,
    DatasetLinker,
    Heuristic,
    MapperType,
    Task,
)
from bohr.pathconfig import PathConfig, find_project_root, load_path_config
from bohr.templates.dataloaders.from_csv import CsvDatasetLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    bohr_framework_version: str
    tasks: Dict[str, Task]
    datasets: Dict[str, Dataset]
    linkers: List[DatasetLinker]
    paths: PathConfig

    def serealize(self, **kwargs) -> Dict[str, Any]:
        return {
            "bohr_framework_version": version(),
            "tasks": {name: jsons.dump(task) for name, task in self.tasks.items()},
            "datasets": {
                name: jsons.dump(dataset, **kwargs)
                for name, dataset in self.datasets.items()
            },
            "dataset-linkers": sorted(
                [jsons.dump(linker) for linker in self.linkers],
                key=lambda l: (l["from"], l["to"]),
            ),
        }

    @staticmethod
    def load(project_root: Path) -> "Config":
        with open(project_root / "bohr.json") as f:
            return jsons.loads(
                f.read(), Config, path_config=load_path_config(project_root)
            )

    def dump(self, project_root: Path) -> None:
        with open(project_root / "bohr.json", "w") as f:
            f.write(
                json.dumps(
                    jsons.dump(self, data_dir=load_path_config(project_root).data_dir),
                    indent=2,
                )
            )


def get_version_from_config() -> str:
    config_file = find_project_root() / "bohr.json"
    with open(config_file, "r") as f:
        dct = jsons.loads(f.read())
        return str(dct["bohr-framework-version"])


def load_config(project_root: Optional[Path] = None) -> Config:
    project_root = project_root or find_project_root()
    config = Config.load(project_root)

    version_installed = version()
    if str(config.bohr_framework_version) != version_installed:
        raise EnvironmentError(
            f"Version of bohr framework from config: {config.bohr_framework_version}. "
            f"Version of bohr installed: {version_installed}"
        )
    return config


def load_artifact_class(artifact_name: str) -> ArtifactType:
    *path, name = artifact_name.split(".")
    try:
        module = importlib.import_module(".".join(path))
    except ModuleNotFoundError as e:
        raise ValueError(f'Module {".".join(path)} not defined.') from e

    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ValueError(f"Artifact {name} not found in module {module}") from e


def load_mapper_type(path_to_mapper_obj: str) -> MapperType:
    # TODO deduplicate
    *path, name = path_to_mapper_obj.split(".")
    try:
        module = importlib.import_module(".".join(path))
    except ModuleNotFoundError as e:
        raise ValueError(f'Module {".".join(path)} not defined.') from e

    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ValueError(f"Mapper {name} not found in module {module}") from e


def get_mapper_by_name(name: str) -> str:
    return load_mapper_type(name)


def deserialize_task(
    dct: Dict[str, Any],
    cls,
    task_name: str,
    heuristic_path: Path,
    datasets: Dict[str, Dataset],
    **kwargs,
) -> "Task":
    # """
    # >>> jsons.loads('{"top_artifact": "artifacts.commit.Commit", "test_dataset_names": [], "train_dataset_names": []}', Task, project_root='/', task_name="x")
    # """
    test_datasets = {
        dataset_name: datasets[dataset_name] for dataset_name in dct["test_datasets"]
    }
    train_datasets = {
        dataset_name: datasets[dataset_name] for dataset_name in dct["train_datasets"]
    }

    artifact = load_artifact_class(dct["top_artifact"])
    heuristic_groups = get_heuristic_module_list(artifact, heuristic_path)
    return Task(
        task_name,
        dct["author"] if "author" in dct else None,
        dct["description"] if "description" in dct else "",
        artifact,
        dct["label_categories"],
        train_datasets=train_datasets,
        test_datasets=test_datasets,
        label_column_name=dct["label_column_name"],
        heuristic_groups=heuristic_groups,
    )


def deserialize_config(dct, cls, path_config: PathConfig, **kwargs) -> Config:
    """
    >>> jsons.loads('{"bohr_framework_version": 0.1, "tasks": {}, "datasets": {}, "dataset-linkers": {}}', Config, \
path_config={})
    Config(bohr_framework_version=0.1, tasks={}, datasets={}, linkers=[], paths={})
    """
    datasets: Dict[str, Dataset] = {}
    for dataset_name, dataset_object in dct["datasets"].items():
        datasets[dataset_name] = jsons.load(
            dataset_object,
            Dataset,
            dataset_name=dataset_name,
            downloaded_data_dir=path_config.downloaded_data_dir,
            data_dir=path_config.data_dir,
        )
    linkers = [
        jsons.load(
            dataset_linker_obj,
            DatasetLinker,
            datasets=datasets,
            data_dir=path_config.data_dir,
        )
        for dataset_linker_obj in dct["dataset-linkers"]
    ]

    for dataset_name, dataset in datasets.items():
        dataset.mapper.linkers = []

    for linker in linkers:
        linker.from_.mapper.linkers = linkers

    tasks = dict()
    for task_name, task_json in dct["tasks"].items():
        tasks[task_name] = jsons.load(
            task_json,
            Task,
            task_name=task_name,
            heuristic_path=path_config.heuristics,
            datasets=datasets,
        )
    return Config(
        dct["bohr_framework_version"],
        tasks,
        datasets,
        linkers,
        path_config,
    )


def desearialize_dataset(
    dct: Dict[str, Any],
    cls,
    dataset_name: str,
    downloaded_data_dir: Path,
    data_dir: Path,
    **kwargs,
) -> "Dataset":
    extra_args = {}
    if "mapper" in dct:
        mapper = load_mapper_type(dct["mapper"])
        extra_args["mapper"] = mapper()

    if dct["loader"] == "csv":
        if "n_rows" in dct:
            extra_args["n_rows"] = dct["n_rows"]
        if "sep" in dct:
            extra_args["sep"] = dct["sep"]
        if "keep_default_na" in dct:
            extra_args["keep_default_na"] = dct["keep_default_na"]
        if "dtype" in dct:
            extra_args["dtype"] = jsons.load(dct["dtype"])
        if "path_preprocessed" in dct:
            path_preprocessed = data_dir / dct["path_preprocessed"]
        elif dct["preprocessor"] in ["zip", "7z"]:
            *name, ext = dct["path"].split(".")
            path_preprocessed = data_dir / ".".join(name)
        else:
            path_preprocessed = data_dir / dct["path"]

        dataset_loader = CsvDatasetLoader(
            path_preprocessed=path_preprocessed, **extra_args
        )

        return Dataset(
            name=dataset_name,
            author=dct["author"] if "author" in dct else None,
            description=dct["description"] if "description" in dct else "",
            path_preprocessed=path_preprocessed,
            path_dist=downloaded_data_dir / dct["path"],
            dataloader=dataset_loader,
            test_set=dct["test_set"],
            preprocessor=dct["preprocessor"],
        )
    else:
        raise NotImplementedError()


def desearialize_linker(
    dct: Dict[str, Any],
    cls,
    datasets: Dict[str, Dataset],
    data_dir: Path,
    **kwargs,
) -> "DatasetLinker":
    extras = {}
    if "link" in dct:
        extras["link"] = datasets[dct["link"]]
    return DatasetLinker(from_=datasets[dct["from"]], to=datasets[dct["to"]], **extras)


jsons.set_deserializer(desearialize_linker, DatasetLinker)
jsons.set_serializer(DatasetLinker.serealize, DatasetLinker)
jsons.set_deserializer(desearialize_dataset, Dataset)
jsons.set_serializer(Dataset.serealize, Dataset)
jsons.set_deserializer(deserialize_task, Task)
jsons.set_serializer(Task.serealize, Task)
jsons.set_deserializer(deserialize_config, Config)
jsons.set_serializer(Config.serealize, Config)


def load_heuristics_from_module(
    artifact_type: Type, full_module_path: str
) -> List[Heuristic]:
    def is_heuristic_of_needed_type(obj):
        return (
            isinstance(obj, Heuristic) and obj.artifact_type_applied_to == artifact_type
        )

    heuristics: List[Heuristic] = []
    module = importlib.import_module(full_module_path)
    heuristics.extend(
        [
            obj
            for name, obj in inspect.getmembers(module)
            if is_heuristic_of_needed_type(obj)
        ]
    )
    for name, obj in inspect.getmembers(module):
        if (
            isinstance(obj, list)
            and len(obj) > 0
            and is_heuristic_of_needed_type(obj[0])
        ):
            heuristics.extend(obj)
    check_names_unique(heuristics)
    return heuristics


def check_names_unique(heuristics: List[Heuristic]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)


def get_heuristic_module_list(
    artifact_type: Type,
    heuristics_path: Path,
    limited_to_modules: Optional[Set[str]] = None,
) -> List[str]:
    modules: List[str] = []
    for root, dirs, files in os.walk(heuristics_path):
        for file in files:
            if (
                file.startswith("_")
                or root.endswith("__pycache__")
                or not file.endswith(".py")
            ):
                continue
            full_path = Path(root) / file
            relative_path = full_path.relative_to(heuristics_path.parent)
            heuristic_module_path = ".".join(
                str(relative_path).replace("/", ".").split(".")[:-1]
            )
            if (
                limited_to_modules is None
                or heuristic_module_path in limited_to_modules
            ):
                hs = load_heuristics_from_module(artifact_type, heuristic_module_path)
                if len(hs) > 0:
                    modules.append(heuristic_module_path)
    return sorted(modules)
