from typing import Any, Set, Tuple, Union

import pandas as pd
import regex
from nltk.tokenize import RegexpTokenizer

NgramSet = Set[Union[Tuple[str], str]]

_tokenizer = RegexpTokenizer(r"[\s_\.,%#/\?!\-\'\"\)\(\]\[\:;]", gaps=True)


def safe_tokenize(text: Any) -> Set[str]:
    if text is None:
        return set()
    if pd.isna(text):
        return set()

    tokens = _tokenizer.tokenize(str(text).lower())
    return tokens


def camel_case_to_snake_case(identifier: str) -> str:
    parts = [
        m[0]
        for m in regex.finditer(
            "(_|[0-9]+|[[:upper:]]?[[:lower:]]+|[[:upper:]]+(?![[:lower:]])|[^ ])",
            identifier,
        )
    ]
    return "_".join(parts).lower()
