import re
from typing import List, Optional

from labels import *

from bohr.collection.artifacts.commit import Commit
from bohr.collection.heuristictypes.keywords import KeywordHeuristics
from bohr.core import Heuristic
from bohr.labeling.labelset import Labels
from bohr.util.nlp import NgramSet


@KeywordHeuristics(
    Commit,
    keywords=[
        "bad",
        ["bug", "bugg"],
        "defect",
        ["fail", "failur", "fault"],
        ["bugfix", "fix", "hotfix", "quickfix", "small fix"],
        "not work",
        ["outofbound", "of bound"],
    ],
    name_pattern="bug_message_keywords",
    lf_per_key_word=False,
)
def bug_keywords_lookup_in_message(
    commit: Commit, keywords: List[NgramSet]
) -> Optional[Labels]:
    for keyword in keywords:
        if commit.message.match_ngrams(keyword):
            return CommitLabel.BugFix
    return None


@KeywordHeuristics(
    Commit,
    keywords=["add", ["doc", "document", "javadoc"]],
    name_pattern="bugless_message_keyword_%1",
)
def bugless_keywords_lookup_in_message(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.message.match_ngrams(keywords):
        return CommitLabel.NonBugFix
    return None


@KeywordHeuristics(
    Commit,
    keywords=["refactor"],
    name_pattern="refactoring_message_keyword",
    lf_per_key_word=False,
)
def refactoring_keywords_lookup_in_message(
    commit: Commit,
    keywords: List[NgramSet],
) -> Optional[Labels]:
    if commit.message.match_ngrams(keywords[0]):
        return CommitLabel.NonBugFix
    return None


GITHUB_REF_RE = re.compile(r"gh(-|\s)\d+", flags=re.I)
VERSION_RE = re.compile(r"v\d+.*", flags=re.I)


@Heuristic(Commit)
def github_ref_in_message(commit: Commit) -> Optional[Labels]:
    return CommitLabel.BugFix if GITHUB_REF_RE.search(commit.message.raw) else None


@Heuristic(Commit)
def version_in_message(commit: Commit) -> Optional[Labels]:
    return CommitLabel.NonBugFix if VERSION_RE.search(commit.message.raw) else None
