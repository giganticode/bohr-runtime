import json
from typing import Dict, List, Optional, Tuple, Union

from bohrapi.core import HeuristicObj
from fs.base import FS
from jsonlines import jsonlines
from pymongo import MongoClient
from tqdm import tqdm

from bohrruntime.datamodel.dataset import Dataset


class DataSource:
    def get_connection(self):
        return MongoClient("mongodb://read-only-user:123@10.10.20.160:27017")[
            "commit_explorer"
        ]


def save_dataset(artifacts, metadata, fs: FS) -> None:
    """
    >>> from fs.memoryfs import MemoryFS
    >>> save_dataset([], {'name': 'stub-dataset'}, MemoryFS())
    """
    path = f"{metadata['name']}.jsonl"
    with fs.open(path, "w") as f:
        with jsonlines.open(f.name, "w") as writer:
            for artifact in tqdm(artifacts):
                writer.write(artifact)
    del metadata["name"]
    with fs.open(f"{path}.metadata.json", "w") as f:
        json.dump(metadata, f)


def query_dataset(
    name: str,
    match: Dict,
    projection: Optional[Dict] = None,
    n_datapoints: Optional[int] = None,
    lookup: Optional[Dict] = None,
    db: DataSource = None,
) -> Tuple[List, Dict]:
    # TODO test with some in-memory mongo
    query = [{"$match": match}]
    if lookup is not None:
        query.append({"$lookup": lookup})
    if projection is not None:
        query.append({"$project": projection})
    query.append({"$sort": {"_id": 1}})
    if n_datapoints is not None:
        query.append({"$limit": n_datapoints})

    db = db or DataSource()
    mongo_client = db.get_connection()
    artifacts = mongo_client.commits.aggregate(query, allowDiskUse=True)
    return artifacts, {
        "name": name,
        "match": match,
        "projection": projection,
        "lookup": lookup,
        "limit": n_datapoints,
    }


def get_projection(heuristics: List[HeuristicObj]) -> Dict:
    mock = ArtifactMock()
    for heuristic in heuristics:
        heuristic.non_safe_func(mock, **heuristic.resources)
    return mock.projection


class ArtifactMock:
    """
    >>> mock = ArtifactMock()
    >>> _ = mock['a']['b']
    >>> _ = mock['c']
    >>> mock
    {'a': {'b': 1}, 'c': 1}

    """

    def __init__(self):
        self.projection: Union[Dict, int] = 1

    def __getattr__(self, attr):
        # only called when self.attr doesn't exist
        raise KeyError(
            f"Tried to access property {attr} of artifact. Artifacts should be accessed as dictionaries."
        )

    def __getitem__(self, item):
        if self.projection == 1:
            self.projection = {}
        if item not in self.projection:
            self.projection[item] = ArtifactMock()
        return self.projection[item]

    def __repr__(self):
        return f"{self.projection}"


def load_local_dataset(
    dataset: Dataset,
) -> Tuple[List[Dict], Dict]:  # TODO more precise types?
    if dataset.path is None:
        raise ValueError(f'The dataset should contain "path" attribute')

    artifacts = []
    with jsonlines.open(dataset.path, "r") as reader:
        for artifact in reader:
            artifacts.append(artifact)

    return artifacts, {
        "name": dataset.id,
        "match": None,
        "projection": None,
        "lookup": None,
        "limit": None,
    }


def query_dataset_from_explorer(dataset: Dataset) -> Tuple[List[Dict], Dict]:
    if dataset.query is None:
        artifacts, metadata = query_dataset(
            dataset.id,
            {dataset.id: {"$exists": True}},
            dataset.projection,
            dataset.n_datapoints,
            lookup={
                "from": "issues",
                "localField": "links.bohr.issues",
                "foreignField": "_id",
                "as": "issues",
            },
        )
    else:
        artifacts, metadata = query_dataset(
            dataset.id,
            dataset.query,
            dataset.projection,
            dataset.n_datapoints,
            lookup={
                "from": "issues",
                "localField": "links.bohr.issues",
                "foreignField": "_id",
                "as": "issues",
            },
        )
    return artifacts, metadata
