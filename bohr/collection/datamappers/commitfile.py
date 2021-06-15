from typing import Optional

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.collection.artifacts.commit_file import CommitFile
from bohr.datamodel.artifactmapper import ArtifactMapper


class CommitFileMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(CommitFile, primary_key="file_id", foreign_key="commit_id")

    cache = LRUCache(512)

    def map(self, x: DataPoint) -> Optional[CommitFile]:
        return CommitFile(
            x.filename,
            x.status,
            x.patch if not isinstance(x.patch, float) else None,
            x.change if not isinstance(x.change, float) else None,
        )
