import functools
from subprocess import CalledProcessError
from typing import Callable, Type

from bohr import datamodel
from bohr.artifacts.core import Artifact
from bohr.datamodel import HeuristicFunction, logger
from bohr.labels.labelset import Label


class Heuristic:
    def __init__(self, artifact_type_applied_to: Type[Artifact]):
        self.artifact_type_applied_to = artifact_type_applied_to

    def get_artifact_safe_func(self, f: HeuristicFunction) -> HeuristicFunction:
        def func(artifact, *args, **kwargs):
            if not isinstance(artifact, self.artifact_type_applied_to):
                raise ValueError("Not right artifact")
            try:
                return f(artifact, *args, **kwargs)
            except (
                ValueError,
                KeyError,
                AttributeError,
                IndexError,
                TypeError,
                CalledProcessError,
            ):
                logger.exception(
                    "Exception thrown while applying heuristic, "
                    "skipping the heuristic for this datapoint ..."
                )
                return None

        return functools.wraps(f)(func)

    def __call__(self, f: Callable[..., Label]) -> datamodel.Heuristic:
        safe_func = self.get_artifact_safe_func(f)
        return datamodel.Heuristic(safe_func, self.artifact_type_applied_to)
