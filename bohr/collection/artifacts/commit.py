import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Set

from bohr.collection.artifacts.commit_file import CommitFile
from bohr.collection.artifacts.commit_message import CommitMessage
from bohr.collection.artifacts.issue import Issue
from bohr.datamodel.artifact import Artifact
from bohr.labeling.labelset import Label
from bohr.util.misc import NgramSet

logger = logging.getLogger(__name__)


@dataclass
class Commit(Artifact):

    owner: str
    repository: str
    sha: str
    raw_message: str
    message: CommitMessage = field(init=False)

    @cached_property
    def issues(self) -> List[Issue]:
        return self.linked("issues")

    @cached_property
    def commit_files(self) -> List[CommitFile]:
        return self.linked("commit_files")

    @cached_property
    def labels(self) -> List[Label]:
        return self.linked("labels")

    def __post_init__(self):
        self.message = CommitMessage(self.raw_message)

    def __hash__(self):
        return hash((self.owner, self.repository, self.sha))

    def issues_match_label(self, stemmed_labels: Set[str]) -> bool:
        for issue in self.issues:
            if not issue.stemmed_labels.isdisjoint(stemmed_labels):
                return True
        return False

    def issues_match_ngrams(self, stemmed_keywords: NgramSet) -> bool:
        for issue in self.issues:
            if not issue.stemmed_ngrams.isdisjoint(stemmed_keywords):
                return True
        return False
