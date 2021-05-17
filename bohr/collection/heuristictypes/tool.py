import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type, TypeVar

from bohr.collection.artifacts.commit import Commit
from bohr.core import Heuristic
from bohr.datamodel.artifact import Artifact
from bohr.datamodel.heuristic import HeuristicObj
from bohr.labeling.labelset import Labels
from bohr.util.nlp import camel_case_to_snake_case

logger = logging.getLogger(__name__)


class Tool(ABC):
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

    def __call__(self, f: Callable[[Commit, Tool], Optional[Labels]]) -> HeuristicObj:

        safe_func = self.get_artifact_safe_func(f)
        tool = self.tool_class()
        tool.check_installed()
        heuristic = HeuristicObj(
            safe_func,
            artifact_type_applied_to=self.artifact_type_applied_to,
            resources={camel_case_to_snake_case(type(tool).__name__): tool},
        )

        return heuristic
