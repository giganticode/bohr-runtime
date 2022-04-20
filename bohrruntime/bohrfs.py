import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

from bohrapi.core import Dataset, Experiment, Task

from bohrruntime.fs import find_project_root
from bohrruntime.util.paths import AbsolutePath

logger = logging.getLogger(__name__)


class BohrFsPath:
    def __init__(self, path: Union[str, Path], anchor: AbsolutePath):
        self.path = Path(path)
        self.anchor = anchor

    @staticmethod
    def from_absolute_path(path: AbsolutePath, base: AbsolutePath) -> "BohrFsPath":
        return BohrFsPath(path.relative_to(base), base)

    def to_absolute_path(self) -> AbsolutePath:
        return self.anchor / self.path

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f'{type(self).__name__}: anchor="{self.anchor}" path="{self.path}"'

    def __truediv__(self, other) -> "BohrFsPath":
        if isinstance(other, str) or isinstance(other, Path):
            return BohrFsPath(self.path / other, self.anchor)
        else:
            raise ValueError(
                f"Cannot concatenate {BohrFsPath.__name__} and {type(other).__name__}"
            )

    def __lt__(self, other) -> bool:
        if isinstance(other, BohrFsPath):
            return self.to_absolute_path() < other.to_absolute_path()
        else:
            raise ValueError(
                f"Cannot compare {BohrFsPath.__name__} and {type(other).__name__}"
            )

    def is_heuristic_file(self):
        abs_path = self.to_absolute_path()
        return (
            abs_path.is_file()
            and not str(abs_path.name).startswith("_")
            and not str(abs_path.parent).endswith("__pycache__")
            and str(abs_path.name).endswith(".py")
        )

    def with_anchor(self, with_anchor: AbsolutePath):
        """
        >>> BohrFsPath('a/b', '/root').with_anchor('/root/a')
        BohrFsPath: anchor="/root/a" path="b"
         >>> BohrFsPath('a/b', '/root').with_anchor('random')
         Traceback (most recent call last):
         ...
         ValueError: '/root/a/b' does not start with 'random'
        """
        return BohrFsPath.from_absolute_path(self.to_absolute_path(), with_anchor)


TASK_TEMPLATE = SimpleNamespace(name="${item.task}")
EXPERIMENT_TEMPLATE = SimpleNamespace(name="${item.exp}", task=TASK_TEMPLATE)
DATASET_TEMPLATE = SimpleNamespace(id="${item.dataset}")


class BohrFileSystem:
    def __init__(self, root: AbsolutePath):
        self.root: AbsolutePath = root
        self.heuristics_dir = "heuristics"
        self.cloned_bohr_dir = "cloned-bohr"
        self.runs_dir = "runs"
        self.cached_dataset_dir = "cached-datasets"

    @staticmethod
    def init(root: Optional[AbsolutePath] = None) -> "BohrFileSystem":
        root = root or find_project_root()
        return BohrFileSystem(root)

    @staticmethod
    def deserialize(dct, cls, project_root: AbsolutePath, **kwargs) -> "BohrFileSystem":
        return BohrFileSystem(project_root, **dct)

    @property
    def cloned_bohr(self) -> BohrFsPath:
        return BohrFsPath(self.cloned_bohr_dir, self.root)

    @property
    def heuristics(self) -> BohrFsPath:
        return self.cloned_bohr / self.heuristics_dir

    def heuristic_group(self, heuristic_group: Union[str, BohrFsPath]) -> BohrFsPath:
        return self.heuristics / str(heuristic_group)

    @property
    def runs(self) -> BohrFsPath:
        return BohrFsPath(self.runs_dir, self.root)

    @property
    def cached_datasets(self) -> BohrFsPath:
        return BohrFsPath(self.cached_dataset_dir, self.root)

    def dataset(self, dataset_name: str) -> BohrFsPath:
        return self.cached_datasets / f"{dataset_name}.jsonl"

    def dataset_metadata(self, dataset_name: str) -> BohrFsPath:
        return self.cached_datasets / f"{dataset_name}.jsonl.metadata.json"

    def exp_dir(self, exp: Experiment) -> BohrFsPath:
        return self.runs / exp.task.name / exp.name

    def label_model(self, exp: Experiment) -> BohrFsPath:
        return self.exp_dir(exp) / "label_model.pkl"

    def label_model_weights(self, exp: Experiment) -> BohrFsPath:
        return self.exp_dir(exp) / "label_model_weights.csv"

    def exp_dataset_dir(self, exp: Experiment, dataset: Dataset) -> BohrFsPath:
        path = self.exp_dir(exp) / dataset.id
        abs_path = path.to_absolute_path()
        if not abs_path.exists() and not abs_path.name.startswith("$"):  # TODO hack
            abs_path.mkdir(parents=True, exist_ok=True)
        return path

    def experiment_label_matrix_file(
        self, exp: Experiment, dataset: Dataset
    ) -> BohrFsPath:
        return self.exp_dataset_dir(exp, dataset) / "heuristic_matrix.pkl"

    def experiment_metrics(self, exp: Experiment, dataset: Dataset) -> BohrFsPath:
        return self.exp_dataset_dir(exp, dataset) / "metrics.txt"

    def analysis_json(self, exp: Experiment, dataset: Dataset) -> BohrFsPath:
        return self.exp_dataset_dir(exp, dataset) / "analysis.json"

    def analysis_csv(self, exp: Experiment, dataset: Dataset) -> BohrFsPath:
        return self.exp_dataset_dir(exp, dataset) / "analysis.csv"

    def labeled_dataset(self, exp: Experiment, dataset: Dataset) -> BohrFsPath:
        return self.exp_dataset_dir(exp, dataset) / "labeled.csv"

    def heuristic_matrix_file(
        self, dataset: Dataset, heuristic_group: Union[str, BohrFsPath]
    ) -> BohrFsPath:
        path = self.heuristic_dataset_dir(dataset) / str(heuristic_group)
        abs_path = path.to_absolute_path()
        if not abs_path.exists() and not abs_path.name.startswith("$"):
            abs_path.mkdir(parents=True, exist_ok=True)
        return path / "heuristic_matrix.pkl"

    def single_heuristic_metrics(
        self, task: Task, dataset: Dataset, heuristic_group: Union[str, BohrFsPath]
    ) -> BohrFsPath:
        path = (
            self.runs
            / "__single_heuristic_metrics"
            / task.name
            / dataset.id
            / str(heuristic_group)
        )
        abs_path = path.to_absolute_path()
        if not abs_path.exists() and not abs_path.name.startswith("$"):
            abs_path.mkdir(parents=True, exist_ok=True)
        return path / "metrics.txt"

    def heuristic_dataset_dir(self, dataset: Dataset) -> BohrFsPath:
        return self.runs / "__heuristics" / dataset.id
