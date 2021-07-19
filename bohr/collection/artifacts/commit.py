import logging
from dataclasses import dataclass, field
from functools import cached_property
from time import sleep
from typing import Dict, List, Optional, Set

import jsons
import requests
from requests import Response

from bohr.collection.artifacts.commit_file import CommitFile
from bohr.collection.artifacts.commit_message import CommitMessage
from bohr.collection.artifacts.issue import Issue
from bohr.datamodel.artifact import Artifact
from bohr.labeling.labelset import Label
from bohr.util.misc import NgramSet

logger = logging.getLogger(__name__)


# TODO use existing solution for retries https://stackoverflow.com/questions/15431044/can-i-set-max-retries-for-requests-request
def query_commit_explorer_ironspeed(sha: str) -> Response:
    time_to_sleep = 1
    total_attempts = 7
    for i in range(total_attempts):
        try:
            return requests.get(f"http://10.10.20.160/{sha[:2]}/{sha[2:4]}/{sha[4:]}")
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            print(
                f"ConnectionError, attempt {i+1}/{total_attempts}; waiting {time_to_sleep} seconds before retrying ..."
            )
            sleep(time_to_sleep)
            time_to_sleep *= 5


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

    @cached_property
    def commit_explorer_data(self) -> Optional[Dict]:
        response = query_commit_explorer_ironspeed(self.sha)
        return jsons.loads(response.text) if response.status_code == 200 else None

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
