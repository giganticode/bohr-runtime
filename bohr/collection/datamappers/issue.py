from typing import Optional

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.collection.artifacts.issue import Issue
from bohr.datamodel.artifactmapper import ArtifactMapper


class IssueMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(Issue, "issue_id")

    cache = LRUCache(512)

    def map(self, x: DataPoint) -> Optional[Issue]:
        labels = list(filter(None, x.labels.split(", ")))
        return Issue(x.title, x.body, labels)
