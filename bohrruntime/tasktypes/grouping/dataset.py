from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from bohrapi.core import Artifact, MergeableArtifact
from bohrlabels.labels import MatchLabel
from fs.base import FS
from sklearn.cluster import AgglomerativeClustering

from bohrruntime.datamodel.dataset import DatapointList, Dataset
from bohrruntime.datamodel.model import GroundTruthLabels


# Dataset where a datapoint is a pair of MergeableArtifacts
@dataclass(frozen=True)
class GroupingDataset(Dataset):
    @staticmethod
    def _form_pairs(lst: List[Any]) -> Generator[Tuple[Any, Any], None, None]:
        """
        >>> [a for a in GroupingDataset._form_pairs([1, 2, 3])]
        [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
        """
        for a in lst:
            for b in lst:
                if a is not b:
                    yield a, b

    def load_expanded_view(
        self, fs: FS, n_datapoints: Optional[int] = None
    ) -> List[MergeableArtifact]:
        return super(GroupingDataset, self).load_artifacts(fs, n_datapoints)

    def load_artifacts(
        self, cached_datasets_fs: FS, n_datapoints: Optional[int] = None
    ) -> DatapointList:
        artifacts = self.load_expanded_view(cached_datasets_fs, n_datapoints)
        return [pair for pair in GroupingDataset._form_pairs(artifacts)]

    def get_n_datapoints(self, fs: FS) -> int:
        artifacts = super(GroupingDataset, self).load_artifacts(fs, self.n_datapoints)
        return len(artifacts) * (len(artifacts) - 1)

    def load_ground_truth_labels(
        self, label_from_datapoint_function: Callable, fs: FS
    ) -> Optional[GroundTruthLabels]:
        artifacts = super(GroupingDataset, self).load_artifacts(fs)
        label_series = []
        for a, b in GroupingDataset._form_pairs(artifacts):
            if label_from_datapoint_function(a) == label_from_datapoint_function(b):
                label_series.append(MatchLabel.Match)
            else:
                label_series.append(MatchLabel.NoMatch)
        return GroundTruthLabels(label_series)


# TODO merge to functions below, also make it more general, not specific to name and email
def into_clusters(
    artifacts: List[Artifact],
    predicted_cluster_numbers: List[int],
    true_clusters_numbers: List[int],
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    >>> from bohrapi.artifacts import Identity
    >>> into_clusters([Identity({"emails": ["a@gmail.com"]}), Identity({"names": ["A"]}), Identity({"emails": ["b@gmail.com"]}), Identity({"names": ["B"]})],[2, 1, 2, 2],[1, 1, 2, 2])
    {'A': (['a@gmail.com'], []), 'B': (['b@gmail.com'], ['a@gmail.com', 'b@gmail.com'])}
    """
    if not (
        len(artifacts) == len(true_clusters_numbers) == len(predicted_cluster_numbers)
    ):
        raise ValueError("")
    true_emails: Dict[int, List[str]] = {}
    predicted_emails: Dict[int, List[str]] = {}
    name_to_cluster_numbers = {}
    for artifact, true_cluster_number, predicted_cluster_number in zip(
        artifacts, true_clusters_numbers, predicted_cluster_numbers
    ):
        if artifact.name is not None:
            name_to_cluster_numbers[artifact.name] = (
                true_cluster_number,
                predicted_cluster_number,
            )
        elif artifact.email is not None:
            email = artifact.email
            if true_cluster_number not in true_emails:
                true_emails[true_cluster_number] = []
            true_emails[true_cluster_number].append(email)
            if predicted_cluster_number not in predicted_emails:
                predicted_emails[predicted_cluster_number] = []
            predicted_emails[predicted_cluster_number].append(email)
        else:
            raise AssertionError()
    name_to_cluster_numbers = {
        name: (
            true_emails[tn] if tn in true_emails else [],
            predicted_emails[pn] if pn in predicted_emails else [],
        )
        for name, (tn, pn) in name_to_cluster_numbers.items()
    }
    return name_to_cluster_numbers


def into_clusters2(
    artifacts: List[MergeableArtifact], predicted_cluster_numbers: List[int]
) -> Dict[str, List[str]]:
    """
    >>> from bohrapi.artifacts import Identity
    >>> into_clusters2([Identity({"emails": ["a@gmail.com"]}), Identity({"names": ["A"]}), Identity({"emails": ["b@gmail.com"]}), Identity({"names": ["B"]})],[2, 1, 2, 2])
    {'A': [], 'B': ['a@gmail.com', 'b@gmail.com']}
    """
    if not (len(artifacts) == len(predicted_cluster_numbers)):
        raise ValueError("")
    predicted_emails: Dict[int, List[str]] = {}
    name_to_cluster_numbers = {}
    for artifact, predicted_cluster_number in zip(artifacts, predicted_cluster_numbers):
        if artifact.name is not None:
            name_to_cluster_numbers[artifact.name] = predicted_cluster_number
        elif artifact.email is not None:
            email = artifact.email
            if predicted_cluster_number not in predicted_emails:
                predicted_emails[predicted_cluster_number] = []
            predicted_emails[predicted_cluster_number].append(email)
        else:
            raise AssertionError()
    name_to_cluster_numbers = {
        name: (predicted_emails[pn] if pn in predicted_emails else [])
        for name, pn in name_to_cluster_numbers.items()
    }
    return name_to_cluster_numbers


def accuracy(clusters: Dict[str, Tuple[List[str], List[str]]]) -> float:
    """
    >>> accuracy({'A': (['a@gmail.com'], []), 'B': (['b@gmail.com'], ['a@gmail.com', 'b@gmail.com'])})
    0.25
    """
    ln = len(clusters)
    acc = (
        sum(
            [
                overlapping(c[0], c[1]) / float(len(c[1])) if len(c[1]) != 0 else 0
                for c in clusters.values()
            ]
        )
        / ln
    )
    return acc


def recall(clusters: Dict[str, Tuple[List[str], List[str]]]) -> float:
    """
    >>> recall({'A': (['a@gmail.com'], []), 'B': (['b@gmail.com'], ['a@gmail.com', 'b@gmail.com'])})
    0.5
    """
    ln = len(clusters)
    acc = (
        sum(
            [
                overlapping(c[0], c[1]) / float(len(c[0])) if len(c[0]) != 0 else 0
                for c in clusters.values()
            ]
        )
        / ln
    )
    return acc


def overlapping(a: List, b: List) -> int:
    """
    >>> overlapping([], [])
    0
    >>> overlapping([1, 2, 3], [3])
    1
    >>> overlapping([1], [2])
    0
    >>> overlapping([1, 2, 4], [1, 2, 3])
    2
    """
    set_a = set(a)
    return sum(1 for bb in b if bb in set_a)


def distances_into_clusters(matrix):
    """
    >>> distances_into_clusters([[0.0, 0.8, 0.7],[0.0, 0.0, 0.6],[0.0, 0.6, 0.0]])
    array([1, 0, 0])
    """
    ac = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage="complete",
        distance_threshold=0.8,
    )
    res = ac.fit_predict(matrix)
    return res
