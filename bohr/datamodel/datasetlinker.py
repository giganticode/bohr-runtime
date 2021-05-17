from abc import ABC
from typing import Any, Dict, Optional

import jsons

from bohr.datamodel.dataset import Dataset
from bohr.util.paths import RelativePath


class DatasetLinker(ABC):
    def __init__(self, from_: Dataset, to: Dataset, link: Optional[Dataset] = None):
        self.from_ = from_
        self.link = link
        self.to = to

    def __str__(self):
        return f"{self.from_} -> {self.to}, linker: {self.link}"

    def serealize(self, **kwargs) -> Dict[str, Any]:
        dct = {"from": self.from_.name, "to": self.to.name}
        if self.link:
            dct["link"] = self.link.name
        return dct


def desearialize_linker(
    dct: Dict[str, Any],
    cls,
    datasets: Dict[str, Dataset],
    data_dir: RelativePath,
    **kwargs,
) -> "DatasetLinker":
    extras = {}
    if "link" in dct:
        extras["link"] = datasets[dct["link"]]
    return DatasetLinker(from_=datasets[dct["from"]], to=datasets[dct["to"]], **extras)


jsons.set_deserializer(desearialize_linker, DatasetLinker)
jsons.set_serializer(DatasetLinker.serealize, DatasetLinker)
