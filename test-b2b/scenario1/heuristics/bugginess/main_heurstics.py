import re
from typing import Optional

from labels import *

from bohr.artifacts.commit import Commit
from bohr.decorators import Heuristic
from bohr.labels.labelset import Label, Labels
from bohr.nlp_utils import NgramSet
from bohr.templates.heuristics.keywords import KeywordHeuristics


@KeywordHeuristics(Commit, "bug", name_pattern="bug_message_keyword_%1")
def bug_keywords_lookup_in_message(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.message.match_ngrams(keywords):
        return CommitLabel.BugFix
    return None


@KeywordHeuristics(Commit, "bugless", name_pattern="bugless_message_keyword_%1")
def bugless_keywords_lookup_in_message(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.message.match_ngrams(keywords):
        return CommitLabel.NonBugFix
    return None


# @keyword_labeling_functions('bogusbugs', name_pattern='bogusbugs_message_keyword_%1')
def bogus_fix_keyword_in_message(commit: Commit, keywords: NgramSet) -> Optional[Label]:
    if "fix" in commit.message.stemmed_ngrams or "bug" in commit.message.stemmed_ngrams:
        if commit.message.match_ngrams(keywords):
            return CommitLabel.NonBugFix
        else:
            return CommitLabel.BugFix
    return None


GITHUB_REF_RE = re.compile(r"gh(-|\s)\d+", flags=re.I)
VERSION_RE = re.compile(r"v\d+.*", flags=re.I)


@Heuristic(Commit)
def github_ref_in_message(commit: Commit) -> Optional[Labels]:
    return CommitLabel.BugFix if GITHUB_REF_RE.search(commit.message.raw) else None


@Heuristic(Commit)
def version_in_message(commit: Commit) -> Optional[Labels]:
    return CommitLabel.NonBugFix if VERSION_RE.search(commit.message.raw) else None
