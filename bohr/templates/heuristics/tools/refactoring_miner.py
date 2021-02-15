import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

import jsons

from bohr.artifacts.commit import Commit
from bohr.config import Config
from bohr.templates.heuristics.tool import Tool


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


class RefactoringMiner(Tool):
    def __init__(self, config: Config):
        super().__init__(config)

    def check_installed(self):
        pass

    def run(self, commit: Commit) -> RefactoringMinerOutput:
        url = get_full_github_url(commit.owner, commit.repository)
        cmd = ["./RefactoringMiner", "-gc", url, commit.sha, "1000"]

        refactoring_miner_path = Path(self.config.software_path) / "RefactoringMiner-2.0.3" / "bin"
        result = subprocess.run(
            cmd, cwd=refactoring_miner_path, capture_output=True, check=True
        )

        output = result.stdout.decode()
        return jsons.loads(output, cls=RefactoringMinerOutput)
