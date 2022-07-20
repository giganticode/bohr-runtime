import logging
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Optional

import pkg_resources
from rich.logging import RichHandler

# from bohrruntime.config.appconfig import AppConfig
from bohrruntime.storageengine import StorageEngine

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


bohr_framework_root = Path(__file__).parent


def version() -> str:
    return pkg_resources.get_distribution("bohr-runtime").version


def appauthor() -> str:
    return "giganticode"


def appname() -> str:
    return "bohr-runtime"


__version__ = version()


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    print("============================")
    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    print("============================")


def setup_loggers(verbose: Optional[bool] = None):
    if verbose is None:
        # verbose = AppConfig.load().verbose
        verbose = False
    if not verbose:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)
    else:
        import warnings

        warnings.showwarning = warn_with_traceback
    root = logging.root
    for (logger_name, logger) in root.manager.loggerDict.items():
        if logger_name != "bohrruntime" and not logger_name.startswith("bohrruntime."):
            # logger.disabled = True
            pass
        else:
            if verbose:
                logging.getLogger("bohrruntime").setLevel(logging.DEBUG)


BOHR_ORGANIZATION = "giganticode"
BOHR_REPO_NAME = "bohr"
BOHR_REMOTE_URL = f"git@github.com:{BOHR_ORGANIZATION}/{BOHR_REPO_NAME}"
