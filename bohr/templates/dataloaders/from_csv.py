from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from bohr.datamodel import DatasetLoader
from bohr.pathconfig import load_path_config


@dataclass
class CsvDatasetLoader(DatasetLoader):
    n_rows: Optional[int] = None
    sep: str = ","
    dtype: Dict[str, str] = None
    keep_default_na: bool = True

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
