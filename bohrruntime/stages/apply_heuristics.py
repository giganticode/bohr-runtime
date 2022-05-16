from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bohrapi.core import Artifact, HeuristicObj
from bohrlabels.core import Label, Labels
from snorkel.labeling import LabelingFunction
from snorkel.labeling.apply.core import LFApplier

from bohrruntime.bohrfs import BohrFileSystem
from bohrruntime.core import to_labeling_functions
from bohrruntime.dataset import Dataset
from bohrruntime.heuristics import load_heuristics_from_file


def apply_lfs_to_dataset(
    lfs: List[LabelingFunction], artifacts: List[Dict]
) -> np.ndarray:
    applier = LFApplier(lfs=lfs)
    applied_lf_matrix = applier.apply(artifacts)
    return applied_lf_matrix


def apply_heuristics_to_artifacts(
    heuristics: List[HeuristicObj],
    artifacts: Union[List[Artifact], List[Tuple[Artifact, Artifact]]],
) -> pd.DataFrame:
    """
    >>> from bohrapi.core import Heuristic, Artifact
    >>> from enum import auto
    >>> class TestArtifact(Artifact): pass
    >>> class TestLabel(Label): Yes = auto(); No = auto()

    >>> @Heuristic(TestArtifact)
    ... def heuristic_yes(artifact: TestArtifact) -> Optional[Labels]:
    ...     if artifact.raw_data['name'] == 'yes':
    ...         return TestLabel.Yes

    >>> @Heuristic(TestArtifact)
    ... def heuristic_no(artifact: TestArtifact) -> Optional[Labels]:
    ...     if artifact.raw_data['name'] == 'no':
    ...         return TestLabel.No

    >>> apply_heuristics_to_artifacts([heuristic_yes, heuristic_no], [TestArtifact({'name': 'yes'}), TestArtifact({'name': 'no'}), TestArtifact({'name': 'maybe'})])
       heuristic_yes  heuristic_no
    0              1            -1
    1             -1             2
    2             -1            -1
    """
    labeling_functions = to_labeling_functions(heuristics)
    applied_lf_matrix = apply_lfs_to_dataset(labeling_functions, artifacts)
    df = pd.DataFrame(applied_lf_matrix, columns=[lf.name for lf in labeling_functions])
    return df


def apply_heuristics_to_dataset(
    heuristic_group: str, dataset: Dataset, fs: BohrFileSystem
) -> None:
    save_to_matrix = fs.heuristic_matrix_file(
        dataset, heuristic_group
    ).to_absolute_path()

    heuristic_file = fs.heuristics / heuristic_group
    heuristics = load_heuristics_from_file(heuristic_file)
    if not heuristics:
        raise AssertionError(f"No labeling functions in {heuristic_file}")
    artifacts = dataset.load_artifacts()
    df = apply_heuristics_to_artifacts(heuristics, artifacts)
    df.to_pickle(str(save_to_matrix))
