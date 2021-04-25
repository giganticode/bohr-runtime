import functools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Artifact:
    proxies: Dict[str, "ArtifactProxy"] = field(init=False)
    keys: Union[str, Tuple[str, ...]] = field(init=False)

    def __getattr__(self, item):
        if item in self.__slots__:
            return (
                self.proxies[item].load_artifact(self.keys)
                if item in self.proxies
                else []
            )


ArtifactSubclass = TypeVar("ArtifactSubclass", bound="Artifact")


class ArtifactProxy(Artifact):
    """
    #TODO doctests!!!
    """

    def __init__(
        self,
        artifact_dataset: "Dataset",
        index: List[str],
        link: Optional["Dataset"],
    ):
        self.artifact_dataset = artifact_dataset
        self.index = index
        self.link = link

    @functools.cached_property
    def loaded_dataframe(self) -> pd.DataFrame:
        artifact_type_name = self.artifact_dataset.artifact_type.__name__
        logger.debug(f"Reading {artifact_type_name}s ... ")
        artifact_df = self.artifact_dataset.load()
        logger.debug(
            f"Index: {list(artifact_df.index.names)}, "
            f"columns: {list(artifact_df.columns)}, "
            f"n_rows: {len(artifact_df.index)}"
        )
        if self.link is None and self.artifact_dataset.foreign_key is None:
            raise ValueError(
                f"Linker: {self}.\n"
                f"Either linking dataset has to be defined "
                f"or destination dataset has to have foreign key defined, "
                f"however its foreign key is {self.artifact_dataset.foreign_key}"
            )
        if self.link is not None:
            logger.debug(f"Reading linker dataset ... ")
            link_df = self.link.load()
            logger.debug(
                f"Index: {list(link_df.index.names)}, "
                f"columns: {list(link_df.columns)}, "
                f"n_rows: {len(link_df.index)}"
            )
            logger.debug(f"Merging on {self.artifact_dataset.primary_key}")
            link_df = link_df.reset_index()
            combined_df = pd.merge(
                link_df, artifact_df, on=self.artifact_dataset.primary_key
            )
            combined_df.set_index(self.index, inplace=True)
            logger.debug(
                f"Merged dataset -> Index: {list(combined_df.index.names)}, "
                f"columns: {list(combined_df.columns)}, "
                f"n_rows: {len(combined_df.index)}"
            )
            return combined_df
        else:
            return artifact_df

    def load_artifact(
        self, keys: Union[str, Tuple[str, ...]]
    ) -> List[ArtifactSubclass]:
        try:
            df = self.loaded_dataframe.loc[[keys]]
        except KeyError:
            return []
        lst = [
            self.artifact_dataset.dataloader.mapper.cached_map(dependency[1])
            for dependency in df.iterrows()
        ]
        return lst
