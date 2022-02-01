import logging
from typing import List

from bohrapi.core import Task
from bohrlabels.core import NumericLabel, belongs_to
from cachetools import LRUCache

logger = logging.getLogger(__name__)


class CategoryMappingCache(LRUCache):
    """
    >>> from bohrlabels.labels import CommitLabel
    >>> logger.setLevel("CRITICAL")

    >>> cache = CategoryMappingCache([CommitLabel.NonBugFix, CommitLabel.BugFix], 10)
    >>> cache[CommitLabel.NonBugFix]
    0
    >>> cache[CommitLabel.BugFix]
    1
    >>> cache[CommitLabel.MinorBugFix]
    1
    """

    def __init__(self, label_categories: List[NumericLabel], maxsize: int):
        super().__init__(maxsize)
        self.label_categories = label_categories
        self.category_hierarchy = type(label_categories[0])
        self.map = {cat.label: i for i, cat in enumerate(self.label_categories)}

    def __missing__(self, label: NumericLabel) -> int:
        selected_label = belongs_to(label, self.label_categories)
        if selected_label.label in self.map:
            snorkel_label = self.map[selected_label.label]
            self[label] = snorkel_label
            logger.info(
                f"Converted {'|'.join(label.to_commit_labels_set())} label into {snorkel_label}"
            )
            return snorkel_label
        elif selected_label.label > 0:
            logger.info(
                f"Label {'|'.join(label.to_commit_labels_set())} cannot be unambiguously converted to any label, abstaining.."
            )
            self[label] = -1
            return -1
        else:
            raise AssertionError(
                f"Something went wrong. Value has to be > 0 but is: {selected_label.label}"
            )


def map_numeric_label_value(value: int, cache: CategoryMappingCache, task: Task) -> int:
    if value == -1:
        return value

    return cache[NumericLabel(value, task.hierarchy)]
