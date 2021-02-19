import logging
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from bohr.artifacts.commit_file import CommitFile
from bohr.artifacts.commit_message import CommitMessage
from bohr.artifacts.core import Artifact
from bohr.artifacts.issue import Issue
from bohr.nlp_utils import NgramSet

logger = logging.getLogger(__name__)


class Cache:
    def __init__(self):
        self.project_root = None

    @property
    def bugginess_train(self):
        return self.project_root / "data" / "bugginess" / "train"

    @property
    def issues_file(self):
        return self.bugginess_train / "bug_sample_issues.csv"

    @property
    def changes_file(self):
        return self.bugginess_train / "bug_sample_files.csv"

    @property
    def commits_file(self):
        return self.bugginess_train / "bug_sample.csv"

    @lru_cache(maxsize=8)
    def __load_df(self, type: str, owner: str, repository: str):
        path = self.bugginess_train / type / owner / f"{repository}.csv"
        if path.is_file():
            return pd.read_csv(
                path,
                index_col=["sha"],
                keep_default_na=False,
                dtype={"labels": "str"},
            )
        else:
            return None

    @cached_property
    def __issues_df(self):
        logger.debug("Reading and caching bug reports...")
        return pd.read_csv(
            self.issues_file,
            index_col=["owner", "repository", "sha"],
            keep_default_na=False,
            dtype={"labels": "str"},
        )

    @cached_property
    def __files_df(self):
        logger.debug("Reading and caching commit files...")
        return pd.read_csv(self.changes_file, index_col=["owner", "repository", "sha"])

    def get_resources_from_file(self, type: str, owner: str, repository: str, sha: str):
        if type == "issues":
            df = self.__issues_df
        elif type == "files":
            df = self.__files_df
        else:
            raise ValueError("invalid resources type")

        try:
            return df.loc[[(owner, repository, sha)]]
        except KeyError:
            return None

    def get_files(self, owner: str, repository: str, sha: str):
        return self.get_resources_from_file("files", owner, repository, sha)

    def get_issues(self, owner: str, repository: str, sha: str):
        return self.get_resources_from_file("issues", owner, repository, sha)


@dataclass
class Commit(Artifact):

    owner: str
    repository: str
    sha: str
    raw_message: str
    project_root: Optional[Path] = None
    message: CommitMessage = field(init=False)

    def __post_init__(self):
        self.message = CommitMessage(self.raw_message)
        self._cache.project_root = self.project_root

    _cache = Cache()

    def __hash__(self):
        return hash((self.owner, self.repository, self.sha))

    @cached_property
    def files(self) -> List[CommitFile]:
        df = self._cache.get_files(self.owner, self.repository, self.sha)
        files = []

        if df is not None:
            for file in df.itertuples(index=False):
                files.append(
                    CommitFile(
                        file.filename,
                        file.status,
                        file.patch if not isinstance(file.patch, float) else None,
                        file.change if not isinstance(file.change, float) else None,
                    )
                )
        return files

    @cached_property
    def issues(self) -> List[Issue]:
        df = self._cache.get_issues(self.owner, self.repository, self.sha)
        issues = []

        if df is not None:
            for issue in df.itertuples():
                labels = issue.labels
                labels = list(filter(None, labels.split(", ")))

                issues.append(Issue(issue.title, issue.body, labels))

        return issues

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
