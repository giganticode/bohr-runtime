import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsons

from bohr import version
from bohr.config.pathconfig import PathConfig
from bohr.datamodel.dataset import Dataset
from bohr.datamodel.datasetlinker import DatasetLinker
from bohr.datamodel.task import Task
from bohr.fs import find_project_root
from bohr.util.paths import AbsolutePath


@dataclass(frozen=True)
class BohrRepo:
    bohr_framework_version: str
    tasks: Dict[str, Task]
    datasets: Dict[str, Dataset]
    linkers: List[DatasetLinker]

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
    def load(project_root: AbsolutePath) -> "BohrRepo":
        with open(project_root / "bohr.json") as f:
            return jsons.loads(f.read(), BohrRepo)

    def dump(self, project_root: AbsolutePath) -> None:
        with open(project_root / "bohr.json", "w") as f:
            f.write(
                json.dumps(
                    jsons.dump(self, data_dir=PathConfig.load(project_root).data_dir),
                    indent=2,
                )
            )


def deserialize_bohr_repo(
    dct, cls, path_config: Optional[PathConfig] = None, **kwargs
) -> BohrRepo:
    """
    >>> jsons.loads('{"bohr_framework_version": 0.1, "tasks": {}, "datasets": {}, "dataset-linkers": {}}', BohrRepo, \
path_config={'project_root': '/'})
    BohrRepo(bohr_framework_version=0.1, tasks={}, datasets={}, linkers=[])
    """
    path_config = path_config or PathConfig.load()
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
    return BohrRepo(
        dct["bohr_framework_version"],
        tasks,
        datasets,
        linkers,
    )


jsons.set_deserializer(deserialize_bohr_repo, BohrRepo)
jsons.set_serializer(BohrRepo.serealize, BohrRepo)


def load_bohr_repo(project_root: Optional[Path] = None) -> BohrRepo:
    project_root = project_root or find_project_root()
    bohr_repo = BohrRepo.load(project_root)

    version_installed = version()
    if str(bohr_repo.bohr_framework_version) != version_installed:
        raise EnvironmentError(
            f"Version of bohr framework from config: {bohr_repo.bohr_framework_version}. "
            f"Version of bohr installed: {version_installed}"
        )
    return bohr_repo
