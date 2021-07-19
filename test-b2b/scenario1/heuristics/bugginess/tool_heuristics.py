from typing import Optional

from labels import CommitLabel

from bohr.collection.artifacts.commit import Commit
from bohr.collection.heuristictypes.tools.refactoring_miner import RefactoringMiner
from bohr.labeling.labelset import Labels


# this class is only for illustration purposes.
# We are not going to use this anymore, we use CommitExplorer with pre-computed refactoring miner outputs instead
# @ToolOutputHeuristic(Commit, tool=RefactoringMiner)
def refactorings_detected(
    commit: Commit, refactoring_miner: RefactoringMiner
) -> Optional[Labels]:
    if commit.sha.endswith("f"):  # running on 1/16 of commits for now to make it faster
        refactoring_miner_output = refactoring_miner.run(commit)
        if len(refactoring_miner_output.commits[0].refactorings) > 0:
            return CommitLabel.Refactoring
    return None
