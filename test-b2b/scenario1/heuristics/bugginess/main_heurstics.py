import re
from typing import List, Optional

from labels import *

from bohr.artifacts.commit import Commit
from bohr.decorators import Heuristic
from bohr.labels.labelset import Label, Labels
from bohr.nlp_utils import NgramSet
from bohr.templates.heuristics.keywords import KeywordHeuristics


@KeywordHeuristics(
    Commit,
    keywords=[
        "bad",
        "broken",
        ["bug", "bugg"],
        "close",
        "concurr",
        ["correct", "correctli"],
        "corrupt",
        "crash",
        ["deadlock", "dead lock"],
        "defect",
        "disabl",
        "endless",
        "ensur",
        "error",
        "except",
        ["fail", "failur", "fault"],
        ["bugfix", "fix", "hotfix", "quickfix", "small fix"],
        "garbag",
        "handl",
        "incomplet",
        "inconsist",
        "incorrect",
        "infinit",
        "invalid",
        "issue",
        "leak",
        "loop",
        "minor",
        "mistak",
        ["nullpointer", "npe", "null pointer"],
        "not work",
        "not return",
        ["outofbound", "of bound"],
        "patch",
        "prevent",
        "problem",
        "properli",
        "race condit",
        "repair",
        ["resolv", "solv"],
        ["threw", "throw"],
        "timeout",
        "unabl",
        "unclos",
        "unexpect",
        "unknown",
        "unsynchron",
        "wrong",
    ],
    name_pattern="bug_message_keyword_%1",
)
def bug_keywords_lookup_in_message(
    commit: Commit, keywords: NgramSet
) -> Optional[Labels]:
    if commit.message.match_ngrams(keywords):
        return CommitLabel.BugFix
    return None


@KeywordHeuristics(
    Commit,
    keywords=[
        "abil",
        "ad",
        "add",
        "addit",
        "allow",
        "analysi",
        "avoid",
        "baselin",
        "beautification",
        "benchmark",
        "better",
        "bump",
        "chang log",
        ["clean", "cleanup"],
        "comment",
        "complet",
        "configur chang",
        "consolid",
        "convert",
        "coverag",
        "create",
        "deprec",
        "develop",
        ["doc", "document", "javadoc"],
        "drop",
        "enhanc",
        "exampl",
        "exclud",
        "expand",
        "extendgener",
        "featur",
        "forget",
        "format",
        "gitignor",
        "idea",
        "implement",
        "improv",
        "includ",
        "info",
        "intorduc",
        "limit",
        "log",
        "migrat",
        "minim",
        "modif",
        "move",
        "new",
        "note",
        "opinion",
        ["optim", "optimis"],
        "pass test",
        "perf test",
        "perfom test",
        "perform",
        "plugin",
        "polish",
        "possibl",
        "prepar",
        "propos",
        "provid",
        "publish",
        "readm",
        "reduc",
        "refin",
        "reformat",
        "regress test",
        "reimplement",
        "release",
        "remov",
        "renam",
        "reorgan",
        "replac",
        "restrict",
        "restructur",
        "review",
        "rewrit",
        "rid",
        "set up",
        "simplif",
        "simplifi",
        ["speedup", "speed up"],
        "stage",
        "stat",
        "statist",
        "support",
        "switch",
        "test",
        "test coverag",
        "test pass",
        "todo",
        "tweak",
        "unit",
        "unnecessari",
        "updat",
        "upgrad",
        "version",
    ],
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


# @KeywordHeuristics(
#     Commit,
#     keywords=[
#         "ad",
#         "add",
#         "build",
#         "chang",
#         "doc",
#         "document",
#         "javadoc",
#         "junit",
#         "messag",
#         "report",
#         "test",
#         "typo",
#         "unit",
#         "warn",
#     ],
#     name_pattern="bogusbugs_message_keyword_%1",
# )
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
