import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from bohr.artifacts.commit_file import CommitFile
from bohr.artifacts.commit_message import CommitMessage
from bohr.artifacts.core import Artifact
from bohr.artifacts.issue import Issue
from bohr.labels.labelset import Label
from bohr.nlp_utils import NgramSet

logger = logging.getLogger(__name__)


@dataclass
class Commit(Artifact):

    owner: str
    repository: str
    sha: str
    raw_message: str
    issues: List[Issue] = field(default_factory=list)
    commit_files: List[CommitFile] = field(default_factory=list)
    labels: List[Label] = field(default_factory=list)
    message: CommitMessage = field(init=False)

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
