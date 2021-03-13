import logging
from typing import List

from bohr.datamodel import ArtifactMapper, Heuristic
from bohr.labels.cache import CategoryMappingCache
from bohr.snorkel_util import SnorkelLabelingFunction, to_snorkel_label

KEYWORD_GROUP_SEPARATOR = "|"


logger = logging.getLogger(__name__)


def apply_heuristic_and_convert_to_snorkel_label(
    heuristic: Heuristic, cache: CategoryMappingCache, *args, **kwargs
) -> int:
    return to_snorkel_label(heuristic(*args, **kwargs), cache)


def to_labeling_functions(
    heuristics: List[Heuristic], mapper: ArtifactMapper, labels: List[str]
) -> List[SnorkelLabelingFunction]:
    category_mapping_cache = CategoryMappingCache(labels, maxsize=10000)
    labeling_functions = list(
        map(
            lambda h: SnorkelLabelingFunction(
                name=h.__name__,
                f=lambda *args, **kwargs: apply_heuristic_and_convert_to_snorkel_label(
                    h, category_mapping_cache, *args, **kwargs
                ),
                mapper=mapper,
                resources=h.resources,
            ),
            heuristics,
        )
    )
    return labeling_functions
