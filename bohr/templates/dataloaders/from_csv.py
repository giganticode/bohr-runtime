from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from bohr.datamodel import DatasetLoader
from bohr.pathconfig import load_path_config

DEFAULT_SEP = ","


@dataclass
class CsvDatasetLoader(DatasetLoader):
    n_rows: Optional[int] = None
    sep: str = DEFAULT_SEP
    dtype: Dict[str, str] = None
    keep_default_na: bool = True

    def get_extra_params(self) -> Dict[str, Any]:
        dct = {}
        if self.n_rows is not None:
            dct["n_rows"] = self.n_rows
        if self.dtype is not None:
            dct["dtype"] = self.dtype
        if not self.keep_default_na:
            dct["keep_default_na"] = self.keep_default_na
        if self.sep != DEFAULT_SEP:
            dct["sep"] = self.sep
        return dct

    def is_column_present(self, column: str) -> bool:
        path = load_path_config().project_root / self.path_preprocessed
        with open(path) as f:
            columns = f.readline()
        return column in map(lambda s: s.strip(), columns.split(self.sep))

    def load(self) -> pd.DataFrame:
        absolute_path = load_path_config().project_root / self.path_preprocessed
        mapper = self.mapper
        index = (
            mapper.foreign_key if mapper.foreign_key is not None else mapper.primary_key
        )
        try:
            artifact_df = pd.read_csv(
                absolute_path,
                nrows=self.n_rows,
                sep=self.sep,
                dtype=self.dtype,
                keep_default_na=self.keep_default_na,
                index_col=index,
            )
            if artifact_df.empty:
                raise ValueError(
                    f"Dataframe is empty, path: {absolute_path}, n_rows={self.n_rows}"
                )
            return artifact_df
        except ValueError as ex:
            raise ValueError(f"Problem loading dataset: {self}") from ex
