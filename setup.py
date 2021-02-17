import os

from setuptools import find_packages, setup

root_package_name = "bohr"


def version():
    with open(os.path.join(root_package_name, "VERSION")) as version_file:
        return version_file.read().strip()


setup(
    name="bohr",
    version=version(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "cachetools>=4.2.0, <5",
        "click>=7.1.2, <8",
        "Cython>=0.29.21, <0.30",
        "dask==2020.12.0",
        "distributed==2020.12.0",
        "fsspec>=0.8.5, <0.9",
        "jinja2>=2.11.2, <3",
        "jsons>=1.4.0, <2",
        "nltk>=3.5, <4",
        "numpy>=1.19.4, <2",
        "numpyencoder>=0.3.0, <0.4",
        "pandas>=1.2.0, <2",
        "requests>=2.25.1, <3",
        "rich>=9.6.1, <10",
        "snorkel>=0.9.6, <0.10",
        "toolz>=0.11.1, <0.12",
    ],
    entry_points="""
        [console_scripts]
        bohr=bohr.cli:bohr
    """,
)
