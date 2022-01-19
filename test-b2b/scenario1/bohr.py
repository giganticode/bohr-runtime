from bohrapi.artifacts import Commit
from bohrapi.core import Dataset, Experiment, Task, Workspace
from bohrlabels.labels import CommitLabel

HEURISTIC_REVISION = "0888f28b0c1619c3c8ea4378887ff2633f065691"
BOHR_FRAMEWORK_VERSION = "0.5.0rc2"

berger = Dataset(
    id="manual_labels.berger",
    top_artifact=Commit,
)

herzig = Dataset(id="manual_labels.herzig", top_artifact=Commit)


bugginess = Task(
    name="bugginess",
    author="hlib",
    description="bug or not",
    top_artifact=Commit,
    labels=[CommitLabel.NonBugFix, CommitLabel.BugFix],
    test_datasets={
        herzig: lambda c: (
            CommitLabel.BugFix
            if c.raw_data["manual_labels"]["herzig"]["bug"] == 1
            else CommitLabel.NonBugFix
        )
    },
)

keywords_in_message = Experiment(
    "keywords_in_message",
    bugginess,
    berger,
    f"bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{HEURISTIC_REVISION}",
)

w = Workspace(BOHR_FRAMEWORK_VERSION, [keywords_in_message])
