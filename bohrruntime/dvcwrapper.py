import logging
import subprocess
from pprint import pprint
from typing import Dict, List, Optional, Union

from dvc.exceptions import ReproductionError
from dvc.repo import Repo
from tqdm import tqdm

from bohrruntime.pipeline import MultiStage, Stage
from bohrruntime.storageengine import StorageEngine

logger = logging.getLogger(__name__)


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
    storage_engine: StorageEngine = None,
) -> None:
    if not storage_engine.fs.exists(".dvc"):
        init_dvc(storage_engine)
    dvc_repo = Repo()
    if not force:
        dvc_repo.pull()

    substages = (
        stages.get_stage_names()
        if isinstance(stages, MultiStage)
        else [stage.stage_name() for stage in stages]
    )
    if len(substages) > 30:
        print("Checking if any stages need to be recomputed ...")
    dvc_status = dvc_repo.status(targets=substages)
    if len(dvc_status) > 0:
        print(f"Reproducing {len(dvc_status)} out of {len(substages)} stages.")
        if len(dvc_status) < 5:
            reason = parse_status(dvc_status)
            pprint(f"Reason:\n {reason}")
    else:
        print("All outputs are cached.")
    substages = dvc_status.keys()
    failed_stages = []
    for substage in tqdm(substages):
        try:
            dvc_repo.reproduce(substage, force=True, single_item=True)
        except ReproductionError:
            failed_stages.append(substage)
    if len(failed_stages) > 0:
        raise ReproError(f"Substages failed to be reproduced:\n {failed_stages}")
