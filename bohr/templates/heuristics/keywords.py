from typing import Callable, Iterable, List, Set, Tuple, Type, Union

from bohr import datamodel
from bohr.artifacts.core import Artifact
from bohr.decorators import Heuristic
from bohr.labels.labelset import Label
from bohr.nlp_utils import NgramSet


class KeywordHeuristics(Heuristic):
    def __init__(
        self,
        artifact_type_applied_to: Type[Artifact],
        keywords: List[Union[str, List]],
        name_pattern: str,
        lf_per_key_word: bool = True,
        resources=None,
    ):
        super().__init__(artifact_type_applied_to)
        self.keywords = self._parse_keywords(keywords)
        self.name_pattern = name_pattern
        self.lf_per_key_word = lf_per_key_word
        self.resources = resources

        if not self.lf_per_key_word and "%1" in self.name_pattern:
            raise ValueError(f"Wrong name pattern: {self.name_pattern}")

    @staticmethod
    def _parse_keywords(
        keywords: Iterable[Union[str, Iterable[str]]]
    ) -> List[NgramSet]:
        """
        >>> keywords = KeywordHeuristics._parse_keywords([["b c", "d"], "a", "e f"])
        >>> keywords[1:]
        [{'a'}, {('e', 'f')}]
        >>> sorted(keywords[0], key=lambda k: k if isinstance(k, str) else "".join(k))
        [('b', 'c'), 'd']
        """

        def split_words(s: str) -> Union[str, Tuple[str, ...]]:
            spl = s.split(" ")
            return spl[0] if len(spl) == 1 else tuple(spl)

        return [
            {split_words(group)}
            if isinstance(group, str)
            else {split_words(k) for k in group}
            for group in keywords
        ]

    def _create_heuristic(
        self,
        f: Callable[..., Label],
        keywords: Union[NgramSet, List[NgramSet]],
        name: str,
    ) -> datamodel.Heuristic:
        resources = dict(keywords=keywords)

        safe_func = self.get_artifact_safe_func(f)
        safe_func.__name__ = name
        return datamodel.Heuristic(
            safe_func,
            artifact_type_applied_to=self.artifact_type_applied_to,
            resources=resources,
        )

    @staticmethod
    def _create_name_from_ngram_set(ngram_set: NgramSet, pattern: str) -> str:
        """
        >>> KeywordHeuristics._create_name_from_ngram_set({'finish', ('add', 'smth')}, 'heuristic_%1')
        'heuristic_add|smth'
        >>> KeywordHeuristics._create_name_from_ngram_set({'finish', ('add', 'smth')}, 'heuristic')
        Traceback (most recent call last):
        ...
        ValueError: Pattern should contain the placeholder %1, but is heuristic
        """
        if "%1" not in pattern:
            raise ValueError(
                f"Pattern should contain the placeholder %1, but is {pattern}"
            )
        sorted_n_gram_set = sorted(
            [
                ngram if isinstance(ngram, str) else "|".join(ngram)
                for ngram in ngram_set
            ]
        )
        name = pattern.replace("%1", sorted_n_gram_set[0])
        return name

    def __call__(
        self, f: Callable[..., Label]
    ) -> Union[datamodel.Heuristic, List[datamodel.Heuristic]]:
        heuristic_list = []
        if self.lf_per_key_word:
            for keyword_group in self.keywords:
                name = self._create_name_from_ngram_set(
                    keyword_group, self.name_pattern
                )
                heuristic = self._create_heuristic(f, keyword_group, name)
                heuristic_list.append(heuristic)
            return heuristic_list
        else:
            return self._create_heuristic(f, self.keywords, self.name_pattern)
