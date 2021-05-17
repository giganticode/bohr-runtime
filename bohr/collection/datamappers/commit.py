from typing import Optional

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.collection.artifacts.commit import Commit
from bohr.datamodel.artifact import Artifact
from bohr.datamodel.artifactmapper import ArtifactMapper


class CommitMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(Commit, "commit_id")

    cache = LRUCache(512)

    def map(self, x: DataPoint) -> Optional[Artifact]:
        return Commit(x.owner, x.repository, x.sha, str(x.message))
