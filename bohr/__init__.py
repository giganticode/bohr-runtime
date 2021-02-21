import logging
import os
from pathlib import Path

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("bohr")


bohr_framework_root = Path(__file__).parent


def version():
    with open(os.path.join(bohr_framework_root, "VERSION")) as version_file:
        return version_file.read().strip()
