import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import jsons
from deepdiff import DeepDiff

from bohr.config import Config

logger = logging.getLogger(__name__)


def md5_folder(
    path: Path, project_root: Path, python_files: bool = True
) -> Dict[str, str]:
    res: Dict[str, str] = {}
    for root, dir, files in os.walk(path):
        for file in files:
            if python_files and (
                file.startswith("_")
                or root.endswith("__pycache__")
                or not file.endswith(".py")
            ):
                continue
            full_path = Path(root) / file
            with full_path.open("rb") as f:
                res[str(full_path.relative_to(project_root))] = hashlib.md5(
                    f.read()
                ).hexdigest()
    return res


def calculate_lock(config: Config) -> Dict[str, Any]:
    with (config.project_root / "bohr.json").open() as f:
        config_json = jsons.loads(f.read())
    heursistics_json = md5_folder(config.paths.heuristics, config.project_root)
    manual_stages_json = md5_folder(
        config.paths.manual_stages, config.project_root, python_files=False
    )

    return {
        "config": config_json,
        "heuristics": heursistics_json,
        "manual_stages": manual_stages_json,
    }


def bohr_up_to_date(config: Config) -> bool:
    bohr_lock_path = config.project_root / "bohr.lock"
    try:
        with bohr_lock_path.open() as f:
            current_lock: Dict[str, Any] = jsons.loads(f.read())
    except (jsons.exceptions.DecodeError, FileNotFoundError):
        current_lock = {}
    new_lock: Dict[str, Any] = calculate_lock(config)
    diff = DeepDiff(current_lock, new_lock, ignore_order=True)
    up_to_date = str(diff) == "{}"
    if not up_to_date:
        logger.debug(f"Changes:\n\n{diff}")
    return up_to_date


def update_lock(config: Config) -> None:
    new_lock: Dict[str, Any] = calculate_lock(config)
    with (config.project_root / "bohr.lock").open("w") as f:
        json.dump(new_lock, f, indent=4)
        logger.debug("Bohr lock is updated")
