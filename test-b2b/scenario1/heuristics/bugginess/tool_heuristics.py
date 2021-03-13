from typing import Optional

from labels import CommitLabel

from bohr.artifacts.commit import Commit
from bohr.labels.labelset import Labels
from bohr.templates.heuristics.tool import ToolOutputHeuristic
from bohr.templates.heuristics.tools.refactoring_miner import RefactoringMiner


@ToolOutputHeuristic(Commit, tool=RefactoringMiner)
def refactorings_detected(
    commit: Commit, refactoring_miner: RefactoringMiner
) -> Optional[Labels]:
    if commit.sha.endswith("f"):  # running on 1/16 of commits for now to make it faster
        refactoring_miner_output = refactoring_miner.run(commit)
        if len(refactoring_miner_output.commits[0].refactorings) > 0:
            return CommitLabel.Refactoring
    return None
