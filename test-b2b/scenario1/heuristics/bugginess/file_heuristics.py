from typing import Optional

from labels import CommitLabel

from bohr.collection.artifacts.commit import Commit
from bohr.core import Heuristic
from bohr.labeling.labelset import Labels


@Heuristic(Commit)
def no_files_have_modified_status(commit: Commit) -> Optional[Labels]:
    for file in commit.commit_files:
        if file.status == "modified":
            return None
    return CommitLabel.NonBugFix


@Heuristic(Commit)
def bug_if_only_changed_lines_in_one_file(commit: Commit) -> Optional[Labels]:
    if (
        len(commit.commit_files) == 1
        and commit.commit_files[0].status == "modified"
        and commit.commit_files[0].changes
        and commit.commit_files[0].no_added_lines()
        and commit.commit_files[0].no_removed_lines()
    ):
        return CommitLabel.BugFix
    return None


@Heuristic(Commit)
def bugless_if_many_files_changes(commit: Commit) -> Optional[Labels]:
    if len(commit.commit_files) > 6:
        return CommitLabel.NonBugFix
    else:
        return None
