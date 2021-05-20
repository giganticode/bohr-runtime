from pathlib import Path

from bohr.collection.artifacts import Commit
from bohr.collection.dataloaders.from_csv import CsvDatasetLoader
from bohr.collection.datamappers import CommitMapper
from bohr.datamodel.dataset import Dataset
from bohr.datamodel.task import Task

stub_commit_mapper = CommitMapper()
stub_commit_mapper._linkers = []
stub_dataset1 = Dataset(
    "dataset1",
    "hlib",
    None,
    Path("prep_path/dataset1"),
    Path("dist_path/dataset1"),
    CsvDatasetLoader(Path("prep_path/dataset1"), stub_commit_mapper),
    preprocessor="copy",
)
stub_dataset2 = Dataset(
    "dataset2",
    "hlib",
    None,
    Path("prep_path/dataset2"),
    Path("dist_path/dataset2"),
    CsvDatasetLoader(Path("prep_path/dataset2"), stub_commit_mapper),
    preprocessor="copy",
)
stub_task = Task(
    "bugginess",
    "hlib",
    description=None,
    top_artifact=Commit,
    labels=["bug", "nonbug"],
    _train_datasets={"dataset1": stub_dataset1},
    _test_datasets={"dataset2": stub_dataset2},
    label_column_name="bug",
    heuristic_groups=["group.1", "group.2"],
)
