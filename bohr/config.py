from pathlib import Path
from typing import Dict

DEFAULT_HEURISTICS_DIR = "heuristics"
DEFAULT_DATASET_DIR = "dataloaders"
DEFAULT_TASK_DIR = "tasks"
DEFAULT_METRICS_DIR = "metrics"
DEFAULT_GENERATED_DIR = "generated"
DEFAULT_LABELED_DATA_DIR = "labeled-data"
DEFAULT_DATA_DIR = "data"
DEFAULT_LABELS_DIR = "labels"
DEFAULT_SOFTWARE_DIR = '../bohr-software'


DEFAULT_ARTIFACT_PACKAGE = "bohr.artifacts"
DEFAULT_HEURISTICS_PACKAGE = "heuristics"
DEFAULT_DATASET_PACKAGE = "dataloaders"
DEFAULT_TASK_PACKAGE = "tasks"


def find_project_root() -> Path:
    current_path = Path('.').resolve()
    while True:
        lst = list(current_path.glob('.bohr'))
        if len(lst) > 0 and lst[0].is_dir():
            return current_path
        elif current_path == Path('/'):
            raise ValueError('Not a bohr directory')
        else:
            current_path = current_path.parent


def read_project_config(project_root) -> Dict[str, str]:
    #TODO this is a stub
    return {'software_path': '/Users/hlib/dev/bohr-software'}

class Config:
    def __init__(self):
        self.project_root = find_project_root()
        project_config = read_project_config(self.project_root)

        self.metrics_dir = DEFAULT_METRICS_DIR
        self.generated_dir = DEFAULT_GENERATED_DIR
        self.heuristics_dir = DEFAULT_HEURISTICS_DIR
        self.dataset_dir = DEFAULT_DATASET_DIR
        self.task_dir = DEFAULT_TASK_DIR
        self.labeled_data_dir = DEFAULT_LABELED_DATA_DIR
        self.data_dir = DEFAULT_DATA_DIR
        self.labels_dir = DEFAULT_LABELS_DIR
        self.software_dir = DEFAULT_SOFTWARE_DIR

        self.metrics_path = self.project_root / self.metrics_dir
        self.generated_path = self.project_root / self.generated_dir
        self.heuristics_path = self.project_root / self.heuristics_dir
        self.dataset_path = self.project_root / self.dataset_dir
        self.task_path = self.project_root / self.task_dir
        self.labeled_data_path = self.project_root / self.labeled_data_dir
        self.data_path = self.project_root / self.data_dir
        self.labels_path = self.project_root / self.labels_dir
        self.heuristics_package = DEFAULT_HEURISTICS_PACKAGE
        self.artifact_package = DEFAULT_ARTIFACT_PACKAGE
        self.dataset_package = DEFAULT_DATASET_PACKAGE
        self.task_package = DEFAULT_TASK_PACKAGE
        self.software_path = project_config['software_path']


def load_config() -> Config:
    return Config()