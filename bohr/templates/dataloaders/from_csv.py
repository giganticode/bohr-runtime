from pathlib import Path
from typing import List, Optional

import pandas as pd

from bohr.core import ArtifactMapper, DatasetLoader


class CsvDatasetLoader(DatasetLoader):
    def __init__(
        self,
        name: str,
        path_to_file: str,
        mapper: ArtifactMapper,
        n_rows: Optional[int] = None,
        sep: str = ",",
        test_set: bool = False,
        additional_data_files: List[str] = None,
    ):
        super().__init__(name, test_set, mapper)
        self.path_to_file = path_to_file
        self.n_rows = n_rows
        self.sep = sep
        self.additional_data_files = additional_data_files or []

    def load(self, project_root: Path) -> pd.DataFrame:
        artifact_df = pd.read_csv(project_root / self.path_to_file, nrows=self.n_rows, sep=self.sep)
        if artifact_df.empty:
            raise ValueError(
                f"Dataframe is empty, path: {project_root / self.path_to_file}, n_rows={self.n_rows}"
            )
        return artifact_df

    def get_paths(self, project_root: Path) -> List[Path]:
        return [project_root / self.path_to_file] + list(map(lambda x: project_root / x, self.additional_data_files))
