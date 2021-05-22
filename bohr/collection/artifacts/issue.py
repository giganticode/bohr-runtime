from dataclasses import dataclass
from functools import cached_property
from typing import List, Set

from bohr.datamodel.artifact import Artifact
from bohr.util.misc import NgramSet


@dataclass
class Issue(Artifact):
    title: str
    body: str
    labels: List[str]

    @cached_property
    def stemmed_labels(self) -> Set[str]:
        from nltk import PorterStemmer

        stemmer = PorterStemmer()
        return {stemmer.stem(label) for label in self.labels}

    @cached_property
    def tokens(self) -> Set[str]:
        from bohr.util.nlp import safe_tokenize

        if self.body is None:
            return set()
        return safe_tokenize(self.body)

    @cached_property
    def ordered_stems(self) -> List[str]:
        from nltk import PorterStemmer

        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in self.tokens]

    @cached_property
    def stemmed_ngrams(self) -> NgramSet:
        from nltk import bigrams

        return set(self.ordered_stems).union(set(bigrams(self.ordered_stems)))
