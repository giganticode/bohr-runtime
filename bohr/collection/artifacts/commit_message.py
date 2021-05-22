from dataclasses import dataclass
from functools import cached_property
from typing import List, Set

from bohr.datamodel.artifact import Artifact
from bohr.util.misc import NgramSet
from bohr.util.nlp import safe_tokenize


@dataclass
class CommitMessage(Artifact):
    raw: str

    @cached_property
    def tokens(self) -> Set[str]:

        if self.raw is None:
            return set()
        return safe_tokenize(self.raw)

    @cached_property
    def ordered_stems(self) -> List[str]:
        from nltk import PorterStemmer

        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in self.tokens]

    @cached_property
    def stemmed_ngrams(self) -> NgramSet:
        from nltk import bigrams

        return set(self.ordered_stems).union(set(bigrams(self.ordered_stems)))

    def match_ngrams(self, stemmed_keywords: NgramSet) -> bool:
        return not self.stemmed_ngrams.isdisjoint(stemmed_keywords)
