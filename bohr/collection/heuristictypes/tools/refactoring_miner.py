import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List

import jsons

from bohr.collection.artifacts.commit import Commit
from bohr.collection.heuristictypes.tool import Tool
from bohr.config.pathconfig import PathConfig

logger = logging.getLogger(__name__)


@dataclass
class RefactoringMinerCommit:
    repository: str
    sha1: str
    url: str
    refactorings: List


@dataclass
class RefactoringMinerOutput:
    commits: List[RefactoringMinerCommit]


# TODO do we have such method anywhere
def get_full_github_url(author: str, repo: str) -> str:
    return f"https://github.com/{author}/{repo}"


# this class is only for illustration purposes.
# We are not going to use this anymore, we use CommitExplorer with pre-computed refactoring miner outputs instead
class RefactoringMiner(Tool):
    def __init__(self):
        super().__init__()
        path_config = PathConfig.load()
        refactoring_miner_dir = os.listdir(path_config.software_path)[0]
        logger.debug(f"Using RefactoringMiner version {refactoring_miner_dir}")
        self.path = path_config.software_path / refactoring_miner_dir / "bin"

    def check_installed(self):
        pass

    def run(self, commit: Commit) -> RefactoringMinerOutput:
        url = get_full_github_url(commit.owner, commit.repository)
        cmd = ["./RefactoringMiner", "-gc", url, commit.sha, "1000"]

        result = subprocess.run(cmd, cwd=self.path, capture_output=True, check=True)

        output = result.stdout.decode()
        return jsons.loads(output, cls=RefactoringMinerOutput)
