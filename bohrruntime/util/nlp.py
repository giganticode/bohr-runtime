from typing import Any, Set

import pandas as pd
from nltk import RegexpTokenizer


def safe_tokenize(text: Any) -> Set[str]:
    if text is None:
        return set()
    if pd.isna(text):
        return set()

    tokens = _tokenizer.tokenize(str(text).lower())
    return tokens


_tokenizer = RegexpTokenizer(r"[\s_\.,%#/\?!\-\'\"\)\(\]\[\:;]", gaps=True)
