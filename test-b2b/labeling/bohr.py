from bohrapi.artifacts import Commit
from bohrapi.core import Dataset, Experiment, LabelingTask, Workspace
from bohrlabels.core import LabelSet
from bohrlabels.labels import CommitLabel

HEURISTIC_REVISION = "5e101b0d8dc87680d6f6b7af012feffbc55834be"
BOHR_FRAMEWORK_VERSION = "0.7.0"

berger = Dataset(id="manual_labels.berger", heuristic_input_artifact_type=Commit)

herzig = Dataset(id="manual_labels.herzig", heuristic_input_artifact_type=Commit)


bugginess = LabelingTask(
    name="bugginess",
    author="hlib",
    description="bug or not",
    heuristic_input_artifact_type=Commit,
    test_datasets={
        herzig: lambda c: (
            CommitLabel.BugFix
            if c.raw_data["manual_labels"]["herzig"]["bug"] == 1
            else CommitLabel.NonBugFix
        )
    },
    labels=(CommitLabel.NonBugFix, CommitLabel.BugFix),
)

refactoring = LabelingTask(
    name="refactoring",
    author="hlib",
    description="refactoring or not",
    heuristic_input_artifact_type=Commit,
    labels=(
        ~LabelSet.of(CommitLabel.Refactoring),
        CommitLabel.Refactoring,
    ),
    test_datasets={
        herzig: lambda c: (
            CommitLabel.Refactoring
            if c.raw_data["manual_labels"]["herzig"]["CLASSIFIED"] == "REFACTORING"
            else CommitLabel.CommitLabel & ~CommitLabel.Refactoring
        )
    },
)

keywords_in_message = Experiment(
    "keywords_in_message",
    bugginess,
    berger,
    f"bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{HEURISTIC_REVISION}",
)

keywords_in_message_refactoring = Experiment(
    "keywords_in_message_refactoring",
    refactoring,
    berger,
    f"bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@{HEURISTIC_REVISION}",
)

w = Workspace(
    BOHR_FRAMEWORK_VERSION, [keywords_in_message, keywords_in_message_refactoring]
)
