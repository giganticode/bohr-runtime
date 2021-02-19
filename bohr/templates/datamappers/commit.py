from pathlib import Path
from typing import Optional, Type

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.artifacts.commit import Commit
from bohr.core import ArtifactMapper


class CommitMapper(ArtifactMapper):

    cache = LRUCache(512)

    def __init__(self, project_root: Optional[Path]) -> None:
        super().__init__("CommitMapper", [], memoize=False)
        self.project_root = project_root

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        key = (x.owner, x.repository, x.sha)
        if key in self.cache:
            return self.cache[key]

        commit = Commit(x.owner, x.repository, x.sha, str(x.message), self.project_root)
        self.cache[key] = commit

        return commit

    def get_artifact(self) -> Type:
        return Commit
