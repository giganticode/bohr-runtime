from bohr.templates.dataloaders.from_csv import CsvDatasetLoader
from bohr.templates.datamappers.method import MethodMapper

dataset_loader = CsvDatasetLoader(
    "smells-test_10",
    path_to_file="data/smells/test.csv",
    mapper=MethodMapper(),
    sep=";",
    n_rows=10,
    test_set=True,
)
