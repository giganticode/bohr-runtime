from bohr.templates.dataloaders.from_csv import CsvDatasetLoader
from bohr.templates.datamappers.commit import CommitMapper

dataset_loader = CsvDatasetLoader(
    "bugginess-train-1k",
    path_to_file="data/bugginess/train/bug_sample.csv",
    mapper=CommitMapper(),
    n_rows=1000,
    test_set=False,
    additional_data_files=[
        "data/bugginess/train/bug_sample_issues.csv",
        "data/bugginess/train/bug_sample_files.csv",
    ],
)
