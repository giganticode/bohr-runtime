from pathlib import Path

from bohrapi.artifacts.identity import Identity
from bohrapi.core import Experiment, GroupingDataset, GroupingTask, Workspace

HEURISTIC_REVISION = "@5e101b0d8dc87680d6f6b7af012feffbc55834be"
BOHR_FRAMEWORK_VERSION = "0.7.0"

apache_train = GroupingDataset(
    id="apache_train",
    path="/Users/hlib/dev/bohr-runtime/test-b2b/grouping/local-datasets/apache_train.jsonl",
    heuristic_input_artifact_type=Identity,
)

apache_test = GroupingDataset(
    id="apache_test",
    path="/Users/hlib/dev/bohr-runtime/test-b2b/grouping/local-datasets/apache_test.jsonl",
    heuristic_input_artifact_type=Identity,
)

identities = GroupingTask(
    name="identities",
    author="hlib",
    description="merging online identities",
    heuristic_input_artifact_type=Identity,
    test_datasets={apache_test: lambda c: c.raw_data["group"]},
    # (matches := (3 - 1)/ (18. - 1), 1 - matches), # if n clusters k identities per cluster -> k-1 match out of n-1
)

trivial_experiment = Experiment(
    "trivial",
    identities,
    apache_train,
    HEURISTIC_REVISION,
)

w = Workspace(BOHR_FRAMEWORK_VERSION, [trivial_experiment])
