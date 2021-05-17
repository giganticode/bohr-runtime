import importlib
from typing import Optional

from cachetools import LRUCache
from snorkel.types import DataPoint

from bohr.collection.artifacts.issue import Issue
from bohr.datamodel.artifactmapper import ArtifactMapper
from bohr.labeling.labelset import Label


class ManualLabelMapper(ArtifactMapper):
    def __init__(self):
        super().__init__(Label, foreign_key="commit_id")

    cache = LRUCache(512)

    def map(self, x: DataPoint) -> Optional[Issue]:
        class_name, obj = x.label.split('.')
        module = importlib.import_module("labels")
        clz = getattr(module, class_name)
        return clz[obj]
