import logging
from functools import cached_property

import pandas as pd

from bohr.datamodel import DatasetLinker
from bohr.pathconfig import load_path_config

logger = logging.getLogger(__name__)


class CommitIssueLinker(DatasetLinker):
    @cached_property
    def get_resources(self) -> pd.DataFrame:
        logger.debug("Reading bug reports...")
        link_file_absolute_path = load_path_config().project_root / self.link_file
        link_df = pd.read_csv(
            link_file_absolute_path, index_col=["owner", "repository", "sha"]
        )
        issue_df = pd.read_csv(
            self.to.path_preprocessed,
            keep_default_na=False,
            dtype={"labels": "str"},
        )
        return pd.merge(link_df, issue_df, on="issue_id")


class CommitFileLinker(DatasetLinker):
    @cached_property
    def get_resources(self) -> pd.DataFrame:
        logger.debug("Reading commit changes...")
        return pd.read_csv(
            self.to.path_preprocessed, index_col=["owner", "repository", "sha"]
        )
