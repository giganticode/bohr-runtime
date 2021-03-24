from pathlib import Path
from typing import List, Optional

import pandas as pd

from bohr.core import ArtifactMapper
from bohr.datamodel import DatasetLoader


class CsvDatasetLoader(DatasetLoader):
    def __init__(
        self,
        path_to_file: str,
        mapper: ArtifactMapper,
        path_to_raw_file: Optional[str] = None,
        n_rows: Optional[int] = None,
        sep: str = ",",
        test_set: bool = False,
        additional_data_files: List[str] = None,
        format: str = "csv",
        from_file: bool = True,
    ):
        super().__init__(test_set, mapper, format, from_file)
        self.path_to_file = path_to_file
        self.path_to_raw_file = path_to_raw_file
        self.n_rows = n_rows
        self.sep = sep
        self.additional_data_files = additional_data_files or []

    def load(self, project_root: Path) -> pd.DataFrame:
        artifact_df = pd.read_csv(
            project_root / self.path_to_file, nrows=self.n_rows, sep=self.sep
        )
        if artifact_df.empty:
            raise ValueError(
                f"Dataframe is empty, path: {project_root / self.path_to_file}, n_rows={self.n_rows}"
            )
        return artifact_df

    def get_paths(self, project_root: Path) -> List[Path]:
        return [project_root / self.path_to_file] + list(
            map(lambda x: project_root / x, self.additional_data_files)
        )

    def get_relative_paths(self) -> List[str]:
        return [self.path_to_file] + self.additional_data_files
