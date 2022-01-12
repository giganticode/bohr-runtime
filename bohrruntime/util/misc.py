from typing import Any, Dict, List, Set, Tuple, Union

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


def merge_dicts_(
    a: Dict[str, List[Any]], b: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """
    >>> a = {}
    >>> merge_dicts_(a, {})
    {}
    >>> merge_dicts_(a, {'x': ['x1']})
    {'x': ['x1']}
    >>> merge_dicts_(a, {'x': ['x2']})
    {'x': ['x1', 'x2']}
    >>> merge_dicts_(a, {'x': ['x3'], 'y': ['y1']})
    {'x': ['x1', 'x2', 'x3'], 'y': ['y1']}
    """
    for k, v in b.items():
        if k not in a:
            a[k] = []
        a[k].extend(v)
    return a
