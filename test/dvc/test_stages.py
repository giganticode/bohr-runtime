from pathlib import Path
from test.testutils import stub_commit_mapper, stub_task

from bohr.config.pathconfig import PathConfig
from bohr.dvc.stages import ApplyHeuristicsCommand, ParseLabelsCommand


def test_parse_labels_command():
    command = ParseLabelsCommand(
        PathConfig(Path("/project_root"), Path("/software_root"))
    )
    assert command.summary() == "parse labels"
    assert command.get_name() == "parse_labels"
    assert command.to_string() == [
        "dvc",
        "run",
        "-v",
        "--no-exec",
        "--force",
        "-n",
        "parse_labels",
        "-d",
        "/project_root/labels",
        "-O",
        "labels.py",
        "bohr",
        "porcelain",
        "parse-labels",
    ]


def test_apply_heuristics_command():
    command = ApplyHeuristicsCommand(
        PathConfig(Path("/project_root"), Path("/software_root")),
        stub_task,
        "group.1",
        datasets=["dataset1"],
        execute_immediately=True,
    )
    assert (
        command.summary() == "[bugginess] apply heuristics (group: group.1) to dataset1"
    )
    assert command.get_name() == "bugginess_apply_heuristics__group_1__dataset1"
    assert command.to_string() == [
        "dvc",
        "run",
        "-v",
        "--force",
        "-n",
        "bugginess_apply_heuristics__group_1__dataset1",
        "-d",
        "labels.py",
        "-d",
        "group/1.py",
        "-d",
        "prep_path/dataset1",
        "-p",
        "bohr.json:bohr_framework_version",
        "-o",
        "generated/bugginess/group.1/heuristic_matrix_dataset1.pkl",
        "-M",
        "metrics/bugginess/group.1/heuristic_metrics_dataset1.json",
        "bohr",
        "porcelain",
        "apply-heuristics",
        "bugginess",
        "--heuristic-group",
        "group.1",
        "--dataset",
        "dataset1",
    ]
