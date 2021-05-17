import logging
import os
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

from bohr.config.appconfig import AppConfig

FORMAT = "%(message)s"
logging.basicConfig(
    level="WARN", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


logger = logging.getLogger("bohr")
logger.setLevel("INFO")


bohr_framework_root = Path(__file__).parent


def version() -> str:
    with open(os.path.join(bohr_framework_root, "VERSION")) as version_file:
        return version_file.read().strip()


def appauthor() -> str:
    return "giganticode"


def appname() -> str:
    return "bohr"


__version__ = version()


def setup_loggers(verbose: Optional[bool] = None):
    if verbose is None:
        verbose = AppConfig.load().verbose
    logging.captureWarnings(True)
    root = logging.root
    for (logger_name, logger) in root.manager.loggerDict.items():
        if logger_name != "bohr" and not logger_name.startswith("bohr."):
            logger.disabled = True
        else:
            if verbose:
                logging.getLogger("bohr").setLevel(logging.DEBUG)
