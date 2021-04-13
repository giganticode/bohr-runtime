from typing import Optional

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.artifacts.issue import Issue
from bohr.core import ArtifactMapper
from bohr.datamodel import ArtifactDependencies


class IssueMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(Issue, ["issue_id"])

    cache = LRUCache(512)

    def map(self, x: DataPoint, dependencies: ArtifactDependencies) -> Optional[Issue]:
        labels = list(filter(None, x.labels.split(", ")))
        return Issue(x.title, x.body, labels)
