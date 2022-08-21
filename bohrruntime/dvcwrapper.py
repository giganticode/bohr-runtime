import logging
import subprocess
from pprint import pprint
from typing import Dict, List, Optional, Union

import yaml
from dvc.exceptions import ReproductionError, CheckoutError
from dvc.repo import Repo
from tqdm import tqdm

from bohrapi.artifacts import Commit
from bohrruntime.datamodel.bohrconfig import BohrConfig
from bohrruntime.datamodel.dataset import Dataset
from bohrruntime.datamodel.experiment import Experiment

from bohrruntime.pipeline import MultiStage, Stage, LoadDatasetsStage
from bohrruntime.storageengine import StorageEngine

logger = logging.getLogger(__name__)

"""
This class encapsulates usage of DVC as a pipeline manager. 
If you want to use another pipeline manager, for example, a self-implemented one,
you need to re-implement methods in this file.
"""


def status(storage_engine: Optional[StorageEngine] = None) -> str:
    storage_engine = storage_engine or StorageEngine.init()
    command = ["dvc", "status"]
    logger.debug(f"Running dvc command: {command}")
    return subprocess.check_output(
        command, cwd=storage_engine.fs.getsyspath("."), encoding="utf8"
    )


def init_dvc(storage_engine: StorageEngine) -> None:
    no_scm = True if not storage_engine.fs.exists(".git") else False
    return Repo.init(storage_engine.fs.getsyspath("."), no_scm=no_scm)


def parse_status(status: Dict):
    vals = status.values()
    res = []
    for val in vals:
        res1 = []
        for item in val:
            for key, value in item.items():
                if key == "changed deps":
                    res1.append(value)
                elif key == "changed outs":
                    pass
                else:
                    raise AssertionError()
            res.append(res1)
    return res


class ReproError(Exception):
    pass


def repro(
    stages: Union[List[Stage], MultiStage],
    force: bool = False,
    no_pull: bool = False,
    storage_engine: StorageEngine = None,
) -> None:
    if not storage_engine.fs.exists(".dvc"):
        init_dvc(storage_engine)
    dvc_repo = Repo()
    substages = (
        stages.get_stage_names()
        if isinstance(stages, MultiStage)
        else [stage.stage_name() for stage in stages]
    )
    if not force and not no_pull:
        try:
            dvc_repo.pull(substages)
        except CheckoutError:
            pass

    if not force:
        if len(substages) > 30:
            print("Checking if any stages need to be recomputed ...")
        dvc_status = dvc_repo.status(targets=substages)
        if len(dvc_status) > 0:
            print(f"Reproducing {len(dvc_status)} out of {len(substages)} sub-stages.")
            if len(dvc_status) < 5:
                reason = parse_status(dvc_status)
                pprint(f"Reason:\n {reason}")
        else:
            print("All outputs are cached.")
        substages_to_rerun = dvc_status.keys()
    else:
        substages_to_rerun = substages
    failed_stages = []
    for substage in tqdm(substages_to_rerun):
        try:
            dvc_repo.reproduce(substage, force=True, single_item=True)
        except ReproductionError:
            failed_stages.append(substage)
    if len(failed_stages) > 0:
        raise ReproError(f"Substages failed to be reproduced:\n {failed_stages}")


def save_stages_to_pipeline_config(stages: List[Union[MultiStage, List[Stage]]], storage_engine: StorageEngine):
    dvc_config = dvc_config_from_tasks(stages)
    with storage_engine.fs.open("dvc.yaml", "w") as f:
        f.write(yaml.dump(dvc_config))


def dvc_config_from_tasks(stages: List[Union[MultiStage, List[Stage]]]) -> Dict:
    """
    >>> from bohrlabels.core import Label, LabelSet
    >>> from bohrlabels.labels import MatchLabel
    >>> from enum import auto
    >>> from bohrruntime.tasktypes.labeling.core import LabelingTask
    >>> from bohrruntime.testtools import get_stub_storage_engine
    >>> class TestLabel(Label): Yes = auto(); No = auto()
    >>> train = Dataset("id.train", Commit)
    >>> test = Dataset("id.test", Commit)
    >>> labels = (LabelSet.of(MatchLabel.NoMatch), LabelSet.of(MatchLabel.Match))
    >>> task = LabelingTask("name", "author", "desc", Commit, {test: lambda x:x}, labels)
    >>> from bohrruntime import bohr_framework_root
    >>> storage_engine = get_stub_storage_engine()
    >>> workspace = BohrConfig('0.x.x', [Experiment('exp', task, train, 'bugginess/conventional_commit_regex')])
    >>> stages = [LoadDatasetsStage(storage_engine, workspace), ApplyHeuristicsCommand(storage_engine, workspace)]
    >>> dvc_config_from_tasks(stages)
    {'stages': {'LoadDatasets': {'foreach': ['id.test', 'id.train'], 'do': {'cmd': 'bohr porcelain load-dataset "${item}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': [], 'outs': ['cached-datasets/${item}.jsonl', {'cached-datasets/${item}.jsonl.metadata.json': {'cache': False}}], 'metrics': [], 'always_changed': False}}, 'ApplyHeuristics': {'foreach': {'id.test__/heuristic1': {'dataset': 'id.test', 'heuristic_group': '/heuristic1'}, 'id.test__/heuristic2': {'dataset': 'id.test', 'heuristic_group': '/heuristic2'}, 'id.train__/heuristic1': {'dataset': 'id.train', 'heuristic_group': '/heuristic1'}, 'id.train__/heuristic2': {'dataset': 'id.train', 'heuristic_group': '/heuristic2'}}, 'do': {'cmd': 'bohr porcelain apply-heuristics --heuristic-group "${item.heuristic_group}" --dataset "${item.dataset}"', 'params': [{'bohr.lock': ['bohr_runtime_version']}], 'deps': ['cloned-bohr/heuristics/${item.heuristic_group}', 'cached-datasets/${item.dataset}.jsonl'], 'outs': ['runs/__heuristics/${item.dataset}/${item.heuristic_group}/heuristic_matrix.pkl'], 'metrics': [], 'always_changed': False}}}}
    """
    final_dict = {"stages": {}}
    for multi_stage in stages:
        for stage in multi_stage if isinstance(multi_stage, list) else [multi_stage]:
            name, dvc_dct = next(iter(stage.to_dvc_config_dict().items()))
            final_dict["stages"][name] = dvc_dct
    return final_dict
