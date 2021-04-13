from dataclasses import dataclass
from typing import Optional

import pandas as pd

from bohr.datamodel import DatasetLoader
from bohr.pathconfig import load_path_config


@dataclass
class CsvDatasetLoader(DatasetLoader):
    n_rows: Optional[int] = None
    sep: str = ","

    def load(self) -> pd.DataFrame:
        absolute_path = load_path_config().project_root / self.path_preprocessed
        artifact_df = pd.read_csv(absolute_path, nrows=self.n_rows, sep=self.sep)
        if artifact_df.empty:
            raise ValueError(
                f"Dataframe is empty, path: {absolute_path}, n_rows={self.n_rows}"
            )
        return artifact_df
