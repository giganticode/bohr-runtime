import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type, TypeVar

from bohr import datamodel
from bohr.artifacts.commit import Commit
from bohr.artifacts.core import Artifact
from bohr.decorators import Heuristic
from bohr.labels.labelset import Labels
from bohr.nlp_utils import camel_case_to_snake_case
from bohr.pathconfig import PathConfig, load_path_config

logger = logging.getLogger(__name__)


class Tool(ABC):
    def __init__(self, path_config: PathConfig):
        self.path_config = path_config

    @abstractmethod
    def check_installed(self):
        pass

    @abstractmethod
    def run(self, artifact: Artifact) -> Any:
        pass


ToolSubclass = TypeVar("ToolSubclass", bound="Tool")
ToolType = Type[ToolSubclass]


class ToolOutputHeuristic(Heuristic):
    def __init__(
        self,
        artifact_type_applied_to: Type[Artifact],
        tool: ToolType,
        resources=None,
    ):
        super().__init__(artifact_type_applied_to)
        self.tool_class = tool
        self.resources = resources

    def __call__(
        self, f: Callable[[Commit, Tool], Optional[Labels]]
    ) -> datamodel.Heuristic:

        safe_func = self.get_artifact_safe_func(f)
        path_config = load_path_config()
        tool = self.tool_class(path_config)
        tool.check_installed()
        heuristic = datamodel.Heuristic(
            safe_func,
            artifact_type_applied_to=self.artifact_type_applied_to,
            resources={camel_case_to_snake_case(type(tool).__name__): tool},
        )

        return heuristic
