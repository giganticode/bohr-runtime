from typing import Optional

from labels import CommitLabel

from bohr.artifacts.commit import Commit
from bohr.labels.labelset import Labels
from bohr.nlp_utils import NgramSet
from bohr.templates.heuristics.keywords import KeywordHeuristics


@KeywordHeuristics(Commit, "bug.issue_label", name_pattern="bug_issue_label_keyword_%1")
def bug_keywords_lookup_in_issue_label(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_label(keywords):
        return CommitLabel.BugFix
    return None


@KeywordHeuristics(
    Commit, "bugless.issue_label", name_pattern="bugless_issue_label_keyword_%1"
)
def bugless_keywords_lookup_in_issue_label(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_label(keywords):
        return CommitLabel.NonBugFix
    return None


@KeywordHeuristics(Commit, "bug", name_pattern="bug_issue_body_keyword_%1")
def bug_keywords_lookup_in_issue_body(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_ngrams(keywords):
        return CommitLabel.BugFix
    return None


@KeywordHeuristics(Commit, "bugless", name_pattern="bugless_issue_body_keyword_%1")
def bugless_keywords_lookup_in_issue_body(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.issues_match_ngrams(keywords):
        return CommitLabel.NonBugFix
    return None
