from pathlib import Path
from unittest import mock

import pytest

from bohr.config.pathconfig import PathConfig
from bohr.dvc.commands import add, pull, repro, status


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_status(subprocess_mock):
    status(PathConfig(Path("/project_root"), Path("/software-root")))
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "status"], cwd=Path("/project_root"), encoding="utf8"
    )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_add(subprocess_mock):
    add(Path("path/to/dataset.csv"), Path("/project_root"))
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "add", "dataset.csv"],
        cwd=Path("/project_root/path/to"),
        encoding="utf8",
    )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_pull_paths_not_iterable(subprocess_mock):
    with pytest.raises(ValueError):
        pull(
            Path("path/to/dataset.csv"),
            PathConfig(Path("/project_root"), Path("/software-path")),
        )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_pull(subprocess_mock):
    pull(
        ["path/to/dataset1.csv", "path/to/dataset2.csv"],
        PathConfig(Path("/project_root"), Path("/software-path")),
    )
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "pull", "path/to/dataset1.csv", "path/to/dataset2.csv"],
        cwd=Path("/project_root"),
        encoding="utf8",
    )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_repro(subprocess_mock):
    repro(path_config=PathConfig(Path("/project_root"), Path("/software-path")))
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "repro"], cwd=Path("/project_root"), encoding="utf8"
    )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_repro_with_stages(subprocess_mock):
    repro(
        ["stage1", "stage2"],
        path_config=PathConfig(Path("/project_root"), Path("/software-path")),
    )
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "repro", "stage1", "stage2"], cwd=Path("/project_root"), encoding="utf8"
    )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_repro_pull(subprocess_mock):
    repro(
        pull=True, path_config=PathConfig(Path("/project_root"), Path("/software-path"))
    )
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "repro", "--pull"], cwd=Path("/project_root"), encoding="utf8"
    )


@mock.patch("bohr.dvc.commands.subprocess", autospec=True)
def test_repro_force(subprocess_mock):
    repro(
        force=True,
        path_config=PathConfig(Path("/project_root"), Path("/software-path")),
    )
    subprocess_mock.check_output.assert_called_with(
        ["dvc", "repro", "--force"], cwd=Path("/project_root"), encoding="utf8"
    )
