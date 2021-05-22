from typing import List, Optional

from labels import CommitLabel

from bohr.collection.artifacts.commit import Commit
from bohr.collection.heuristictypes.keywords import KeywordHeuristics
from bohr.labeling.labelset import Labels
from bohr.util.misc import NgramSet


@KeywordHeuristics(
    Commit,
    keywords=["bug", "fixed", "fix", "error"],
    name_pattern="bug_issue_label_keyword_%1",
)
def bug_keywords_lookup_in_issue_label(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_label(keywords):
        return CommitLabel.BugFix
    return None


@KeywordHeuristics(
    Commit,
    keywords=["enhancement", "feature", "request", "refactor", "renovate", "new"],
    name_pattern="bugless_issue_label_keyword_%1",
)
def bugless_keywords_lookup_in_issue_label(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_label(keywords):
        return CommitLabel.NonBugFix
    return None


@KeywordHeuristics(
    Commit,
    keywords=[
        "bad",
        "broken",
        ["bug", "bugg"],
        ["fail", "failur", "fault"],
        ["bugfix", "fix", "hotfix", "quickfix", "small fix"],
        "minor",
        ["nullpointer", "npe", "null pointer"],
        "not work",
    ],
    name_pattern="bug_issue_body_keywords",
    lf_per_key_word=False,
)
def bug_keywords_lookup_in_issue_body(
    commit: Commit, keywords: List[NgramSet]
) -> Optional[Labels]:
    for keyword_group in keywords:
        if commit.issues_match_ngrams(keyword_group):
            return CommitLabel.BugFix
    return None


@KeywordHeuristics(
    Commit,
    keywords=[
        "add",
        ["clean", "cleanup"],
        ["doc", "document", "javadoc"],
    ],
    name_pattern="bugless_issue_body_keyword_%1",
)
def bugless_keywords_lookup_in_issue_body(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_ngrams(keywords):
        return CommitLabel.NonBugFix
    return None
