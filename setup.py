import os
from pathlib import Path
from typing import List

from setuptools import find_packages, setup

bohr_framework_root = Path(__file__).parent


def version() -> str:
    with open(os.path.join(bohr_framework_root / "bohr", "VERSION")) as version_file:
        return version_file.read().strip()


def requirements() -> List[str]:
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="bohr",
    version=version(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements(),
    entry_points="""
        [console_scripts]
        bohr=bohr.cli:bohr
    """,
)
