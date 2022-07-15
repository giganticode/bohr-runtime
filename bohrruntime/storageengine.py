import inspect
import json
import logging
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from bohrapi.core import ArtifactType, HeuristicObj
from fs.base import FS
from fs.memoryfs import MemoryFS

from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment
from bohrruntime.datamodel.model import HeuristicOutputs, Model
from bohrruntime.datamodel.task import PreparedDataset, Task
from bohrruntime.heuristicuri import HeuristicURI, PathTemplate
from bohrruntime.util.paths import create_fs

logger = logging.getLogger(__name__)


@dataclass
class HeuristicLoader:
    heuristic_fs: FS

    def get_heuristic_uris(
        self,
        heuristic_uri: HeuristicURI = None,
        input_artifact_type: Optional[ArtifactType] = None,
        error_if_none_found: bool = True,
    ) -> List[HeuristicURI]:
        """
        >>> from fs.memoryfs import MemoryFS
        >>> hl = HeuristicLoader(MemoryFS())
        >>> hl.get_heuristic_uris(error_if_none_found=False)
        []
        >>>
        """
        heuristic_uri = heuristic_uri or HeuristicURI.from_path_and_fs(
            "/", self.heuristic_fs
        )
        heuristic_files: List[HeuristicURI] = []
        if heuristic_uri.is_heuristic_file():
            heuristic_files.append(heuristic_uri)
        else:
            for root, dirs, files in self.heuristic_fs.walk(str(heuristic_uri.path)):
                for file in files:
                    path_to_heuristic: HeuristicURI = HeuristicURI(
                        Path(f"{root}/{file.name}"), self.heuristic_fs
                    )
                    if path_to_heuristic.is_heuristic_file():
                        heuristic_files.append(path_to_heuristic)
        if input_artifact_type is not None:
            res: List[HeuristicURI] = []
            for h in heuristic_files:
                if self.load_heuristics_by_uri(h, input_artifact_type):
                    res.append(h)
            heuristic_files = res
        if error_if_none_found and len(heuristic_files) == 0:
            raise RuntimeError(
                f"No heuristic groups are found at path: {heuristic_uri}."
            )
        return sorted(heuristic_files)

    def load_all_heuristics(
        self, artifact_type: Type = None
    ) -> Dict[HeuristicURI, List[HeuristicObj]]:
        map: Dict[HeuristicURI, List[HeuristicObj]] = {}

        for heuristic_uri in self.get_heuristic_uris(artifact_type):
            hs = self.load_heuristics_by_uri(heuristic_uri, artifact_type)
            if len(hs) > 0:
                map[heuristic_uri] = hs
        return map

    @abstractmethod
    def load_heuristics_by_uri(
        self, heuristic_uri: HeuristicURI, artifact_type: Optional[Type] = None
    ) -> List[HeuristicObj]:
        pass


@dataclass
class BohrPathStructure:
    def __init__(
        self,
        heuristics: str = "cloned-bohr/heuristics",
        cloned_bohr: str = "cloned-bohr",
        runs: str = "runs",
        cached_datasets: str = "cached-datasets",
    ):
        self.heuristics = heuristics
        self.cloned_bohr = cloned_bohr
        self.runs = runs
        self.cached_datasets = cached_datasets

    def heuristic_dataset_dir(self, dataset: Dataset) -> str:
        return f"{self.runs}/__heuristics/{dataset.id}"

    def heuristic_group(
        self, heuristic_group: Union[PathTemplate, HeuristicURI]
    ) -> str:
        return f"{self.heuristics}/{(heuristic_group if isinstance(heuristic_group, str) else heuristic_group.path)}"

    def dataset(self, dataset_name: str) -> str:
        return f"{self.cached_datasets}/{dataset_name}.jsonl"

    def dataset_metadata(self, dataset_name: str) -> str:
        return f"{self.cached_datasets}/{dataset_name}.jsonl.metadata.json"

    def exp_dir(self, exp: Experiment) -> str:
        return f"{self.runs}/{exp.task.name}/{exp.name}"

    def label_model_weights(self, exp: Experiment) -> str:
        return f"{self.exp_dir(exp)}/label_model_weights.csv"

    def label_model(self, exp: Experiment) -> str:
        return f"{self.exp_dir(exp)}/label_model.pkl"

    def exp_dataset_dir(self, exp: Experiment, dataset: Dataset):
        return f"{self.exp_dir(exp)}/{dataset.id}"

    def experiment_label_matrix_file(self, exp: Experiment, dataset: Dataset) -> str:
        return f"{self.exp_dataset_dir(exp, dataset)}/heuristic_matrix.pkl"

    def experiment_metrics(self, exp: Experiment, dataset: Dataset) -> str:
        return f"{self.exp_dataset_dir(exp, dataset)}/metrics.txt"

    def analysis_json(self, exp: Experiment, dataset: Dataset) -> str:
        return f"{self.exp_dataset_dir(exp, dataset)}/analysis.json"

    def analysis_csv(self, exp: Experiment, dataset: Dataset) -> str:
        return f"{self.exp_dataset_dir(exp, dataset)}/analysis.csv"

    def labeled_dataset(self, exp: Experiment, dataset: Dataset) -> str:
        return f"{self.exp_dataset_dir(exp, dataset)}/labeled.csv"


@dataclass
class StorageEngine:
    fs: FS
    path_structure: BohrPathStructure

    @staticmethod
    def init(fs: Optional[FS] = None) -> "StorageEngine":
        fs = fs or create_fs()
        return StorageEngine(fs, BohrPathStructure())

    def get_workspace_root(self) -> str:
        return self.fs.root_path

    def heuristic_matrix_file(
        self, dataset: Dataset, heuristic_group: Union[PathTemplate, HeuristicURI]
    ) -> str:
        heuristic_group = (
            heuristic_group
            if isinstance(heuristic_group, str)
            else heuristic_group.path
        )
        path = f"{self.path_structure.heuristic_dataset_dir(dataset)}/{heuristic_group}"
        if not self.fs.exists(path) and not Path(path).name.startswith("$"):
            self.fs.makedirs(path)
        return f"{path}/heuristic_matrix.pkl"

    def exp_dataset_dir_make_sure_exists(self, exp: Experiment, dataset: Dataset) -> FS:
        path = self.path_structure.exp_dataset_dir(exp, dataset)
        if not self.fs.exists(path) and not Path(path).name.startswith(
            "$"
        ):  # FIXME this is hack
            self.fs.makedirs(path, recreate=True)
        return self.fs.opendir(path)

    def get_heuristic_module_paths(self, exp: Experiment) -> List[HeuristicURI]:
        heuristic_module_paths = []
        for heuristic_group in (
            exp.heuristic_groups if exp.heuristic_groups is not None else ["."]
        ):
            for heuristic_module_path in self.get_heuristic_loader().get_heuristic_uris(
                HeuristicURI.from_path_and_fs(heuristic_group, self.heuristics_subfs()),
                exp.task.heuristic_input_artifact_type,
            ):
                heuristic_module_paths.append(heuristic_module_path)
        return sorted(heuristic_module_paths)

    def cached_datasets_subfs(self) -> FS:
        return self.fs.makedir(self.path_structure.cached_datasets, recreate=True)

    def get_heuristic_loader(self) -> HeuristicLoader:
        return FileSystemHeuristicLoader(
            self.heuristics_subfs()
        )  # TODO heuristic loader type depends on fs

    def single_heuristic_metrics(
        self,
        task: Task,
        dataset: Dataset,
        heuristic_group: Union[PathTemplate, HeuristicURI],
    ) -> str:
        heuristic_group = (
            heuristic_group
            if isinstance(heuristic_group, str)
            else heuristic_group.path
        )
        path = f"{self.path_structure.runs}/__single_heuristic_metrics/{task.name}/{dataset.id}/{heuristic_group}"

        if not self.fs.exists(path) and not Path(path).name.startswith("$"):
            self.fs.makedirs(path, recreate=True)
        return f"{path}/metrics.txt"

    def load_heuristic_outputs(
        self, exp: Experiment, dataset: Dataset
    ) -> HeuristicOutputs:
        """
        >>> from bohrruntime.testtools import get_stub_experiment, get_stub_dataset, VirtualHeuristicLoader
        >>> import pickle
        >>> stub_fs = MemoryFS()
        >>> bohrfs = StorageEngine(stub_fs, BohrPathStructure('/root'))
        >>> sub_fs = stub_fs.makedirs('runs/stub-task/stub-exp/stub-dataset/')
        >>> df = pd.DataFrame()
        >>> with sub_fs.open('heuristic_matrix.pkl', 'wb', encoding='utf-8') as f:
        ...     pickle.dump(df, f)
        >>> exp = get_stub_experiment()
        >>> dataset = get_stub_dataset()
        >>> bohrfs.load_heuristic_outputs(exp, dataset)
        HeuristicOutputs(label_matrix=array([], shape=(0, 0), dtype=float64))
        """
        all_heuristics_file = self.path_structure.experiment_label_matrix_file(
            exp, dataset
        )
        with self.fs.open(all_heuristics_file, "rb") as f:
            heuristic_matrix = pd.read_pickle(f)
        label_matrix = heuristic_matrix
        return HeuristicOutputs(label_matrix)

    def load_model(self, exp: Experiment) -> Model:
        exp_dir = self.path_structure.exp_dir(exp)
        model = exp.load_model(self.fs, exp_dir)
        return model

    def save_model(self, model: Model, exp: Experiment) -> None:
        exp_dir = self.path_structure.exp_dir(exp)
        exp.save_model(model, self.fs, exp_dir)

    def save_heuristic_outputs(
        self, heuristics_output: HeuristicOutputs, exp: Experiment, dataset: Dataset
    ) -> None:
        exp_dataset_subfs = self.exp_dataset_dir_make_sure_exists(exp, dataset)
        with exp_dataset_subfs.open("heuristic_matrix.pkl", "wb") as f:
            pickle.dump(heuristics_output.label_matrix, f)

    def save_single_heuristic_outputs(
        self,
        heuristics_output: HeuristicOutputs,
        dataset: Dataset,
        heuristic_uri: HeuristicURI,
    ) -> None:
        save_to_matrix = self.heuristic_matrix_file(dataset, heuristic_uri)
        print(save_to_matrix)
        heuristics_output.label_matrix.to_pickle(save_to_matrix)

    def load_single_heuristic_output(
        self, dataset: Dataset, heuristic_uri: HeuristicURI
    ) -> HeuristicOutputs:
        partial_heuristics_file = self.heuristic_matrix_file(dataset, heuristic_uri)
        with self.fs.open(partial_heuristics_file, "rb") as f:
            matrix = pd.read_pickle(f)
        return HeuristicOutputs(matrix)

    def save_single_heuristic_metrics(
        self,
        metrics: Dict[str, Any],
        task: Task,
        dataset: Dataset,
        heuristic_uri: HeuristicURI,
    ) -> None:
        save_to_metrics = self.single_heuristic_metrics(task, dataset, heuristic_uri)
        with self.fs.open(save_to_metrics, "w") as f:
            json.dump(metrics, f)

    def save_experiment_metrics(
        self, metrics: Dict[str, Any], exp: Experiment, dataset: Dataset
    ) -> None:
        self.exp_dataset_dir_make_sure_exists(exp, dataset)
        save_metrics_to = self.path_structure.experiment_metrics(exp, dataset)
        with self.fs.open(save_metrics_to, "w") as f:
            for metric_key, metric_value in metrics.items():
                f.write(f"{metric_key} = {metric_value}\n")

    def save_prepared_dataset(
        self, labeled_dataset: PreparedDataset, exp: Experiment, dataset: Dataset
    ):
        exp_dataset_subfs = self.exp_dataset_dir_make_sure_exists(exp, dataset)
        labeled_dataset.save(exp_dataset_subfs)

    def heuristics_subfs(self) -> FS:
        return self.fs.makedir(self.path_structure.heuristics, recreate=True)

    def cloned_bohr_subfs(self) -> FS:
        return self.fs.makedir(self.path_structure.cloned_bohr, recreate=True)

    def exp_dataset_fs(self, exp: Experiment, dataset: Dataset) -> FS:
        return self.fs.makedir(
            self.path_structure.exp_dataset_dir(exp, dataset), recreate=True
        )


@dataclass
class FileSystemHeuristicLoader(HeuristicLoader):
    def load_heuristics_by_uri(
        self, heuristic_uri: HeuristicURI, artifact_type: Optional[Type] = None
    ) -> List[HeuristicObj]:
        import importlib.util

        heuristic_file_abs_path = heuristic_uri.to_filesystem_path()
        spec = importlib.util.spec_from_file_location(
            "heuristic.module", heuristic_file_abs_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def is_heuristic_of_needed_type(obj, artifact_type):
            return isinstance(obj, HeuristicObj) and (
                obj.artifact_type_applied_to == artifact_type or artifact_type is None
            )

        heuristics: List[HeuristicObj] = []
        for name, obj in inspect.getmembers(module):
            if is_heuristic_of_needed_type(obj, artifact_type):
                if name != (filename := heuristic_file_abs_path.stem):
                    raise ValueError(
                        f"For consistency, file and heuristic name must be the same.\n"
                        f"Hovewer, filename is {filename}, heuristic name is {name}."
                    )
                heuristics.append(obj)
        for name, obj in inspect.getmembers(module):
            if (
                isinstance(obj, list)
                and len(obj) > 0
                and (
                    is_heuristic_of_needed_type(obj[0], artifact_type)
                    or artifact_type is None
                )
            ):
                heuristics.extend(obj)
        check_names_unique(heuristics)
        return heuristics


def check_names_unique(heuristics: List[HeuristicObj]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)
