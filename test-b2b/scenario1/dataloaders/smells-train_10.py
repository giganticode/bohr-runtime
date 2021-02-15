from bohr.templates.dataloaders.from_csv import CsvDatasetLoader
from bohr.templates.datamappers.method import MethodMapper

dataset_loader = CsvDatasetLoader(
    "smells-train_10",
    path_to_file="data/smells/train.csv",
    mapper=MethodMapper(),
    sep=";",
    n_rows=10,
    test_set=False,
)
