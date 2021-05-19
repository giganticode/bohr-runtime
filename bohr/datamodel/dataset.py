from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsons

from bohr.collection.artifacts import artifact_map
from bohr.collection.dataloaders.from_csv import CsvDatasetLoader
from bohr.collection.datamappers import default_mappers
from bohr.datamodel.artifact import ArtifactType
from bohr.datamodel.datasetloader import DatasetLoader
from bohr.fs import get_preprocessed_path
from bohr.util.paths import RelativePath, load_class_by_full_path, relative_to_safe


@dataclass
class Dataset(ABC):
    name: str
    author: str
    description: Optional[str]
    path_preprocessed: RelativePath
    path_dist: RelativePath
    dataloader: DatasetLoader
    preprocessor: str

    def serealize(self, **kwargs) -> Dict[str, Any]:
        dct = {
            "author": self.author,
            "description": self.description,
            "path": self.path_dist.name,
            "path_preprocessed": str(
                relative_to_safe(self.dataloader.path_preprocessed, kwargs["data_dir"])
            ),
            "preprocessor": self.preprocessor,
            "loader": "csv",
        }
        if type(self.mapper).__name__ != "DummyMapper":
            dct["mapper"] = ".".join(
                [type(self.mapper).__module__, type(self.mapper).__name__]
            )
        return {**dct, **self.dataloader.get_extra_params()}

    def load(self):
        return self.dataloader.load()

    def is_column_present(self, column: str) -> bool:
        return self.dataloader.is_column_present(column)

    def get_linked_datasets(self) -> List["Dataset"]:
        return list(map(lambda l: l.to, self.mapper.linkers))

    @property
    def mapper(self) -> "ArtifactMapperSubclass":
        return self.dataloader.mapper

    @property
    def primary_key(self) -> List[str]:
        return self.mapper.primary_key

    @property
    def foreign_key(self):
        return self.mapper.foreign_key

    @property
    def artifact_type(self) -> ArtifactType:
        return self.dataloader.artifact_type

    def __str__(self):
        return f"name: {self.name}, path: {self.path_preprocessed}"


def desearialize_dataset(
    dct: Dict[str, Any],
    cls,
    dataset_name: str,
    downloaded_data_dir: RelativePath,
    data_dir: RelativePath,
    **kwargs,
) -> "Dataset":
    extra_args = {}
    if "mapper" in dct:
        try:
            mapper = default_mappers[artifact_map[dct["mapper"]]]
        except KeyError:
            mapper = load_class_by_full_path(dct["mapper"])
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
        path_preprocessed = get_preprocessed_path(
            Path(dct.get("path_preprocessed")),
            Path(dct["path"]),
            data_dir,
            dct["preprocessor"],
        )

        dataset_loader = CsvDatasetLoader(
            path_preprocessed=path_preprocessed, **extra_args
        )

        return Dataset(
            name=dataset_name,
            author=dct.get("author"),
            description=dct.get("description", ""),
            path_preprocessed=path_preprocessed,
            path_dist=downloaded_data_dir / dct["path"],
            dataloader=dataset_loader,
            preprocessor=dct["preprocessor"],
        )
    else:
        raise NotImplementedError()


jsons.set_deserializer(desearialize_dataset, Dataset)
jsons.set_serializer(Dataset.serealize, Dataset)


def get_all_linked_datasets(datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
    total = datasets.copy()
    for dataset_name, dataset in datasets.items():
        total.update({d.name: d for d in dataset.get_linked_datasets()})
    return total


def train_and_test(
    all_datasets: Dict[str, Dataset], label_column: str
) -> Tuple[Dict[str, Dataset], Dict[str, Dataset]]:
    train, test = {}, {}
    for name, dataset in all_datasets.items():
        dct = test if dataset.is_column_present(label_column) else train
        dct[name] = dataset
    return train, test
