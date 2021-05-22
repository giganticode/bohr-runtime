from typing import Set, Tuple, Union

import regex

NgramSet = Set[Union[Tuple[str], str]]


def camel_case_to_snake_case(identifier: str) -> str:
    parts = [
        m[0]
        for m in regex.finditer(
            "(_|[0-9]+|[[:upper:]]?[[:lower:]]+|[[:upper:]]+(?![[:lower:]])|[^ ])",
            identifier,
        )
    ]
    return "_".join(parts).lower()
