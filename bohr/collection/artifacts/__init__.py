from bohr.collection.artifacts.commit import Commit
from bohr.collection.artifacts.commit_file import CommitFile
from bohr.collection.artifacts.commit_message import CommitMessage
from bohr.collection.artifacts.issue import Issue
from bohr.collection.artifacts.method import Method
from bohr.labeling.labelset import Label

artifact_map = {
    "commit": Commit,
    "commit_file": CommitFile,
    "commit_message": CommitMessage,
    "issue": Issue,
    "method": Method,
    "label": Label,
}
