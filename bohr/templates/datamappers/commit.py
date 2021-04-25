from typing import Optional

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.artifacts.commit import Commit
from bohr.artifacts.core import Artifact
from bohr.core import ArtifactMapper


class CommitMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(Commit, "commit_id")

    cache = LRUCache(512)

    def map(self, x: DataPoint) -> Optional[Artifact]:
        return Commit(x.owner, x.repository, x.sha, str(x.message))
