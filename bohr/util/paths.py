from pathlib import Path
from typing import NewType, TypeVar

RelativePath = NewType("RelativePath", Path)
AbsolutePath = NewType("AbsolutePath", Path)
RelativeOrAbsolute = TypeVar("RelativeOrAbsolute", RelativePath, AbsolutePath)


def relative_to_safe(
    path: RelativeOrAbsolute, base_path: RelativeOrAbsolute
) -> RelativePath:
    return path.relative_to(base_path)


def concat_paths_safe(p1: RelativeOrAbsolute, p2: RelativePath) -> RelativeOrAbsolute:
    return p1 / p2
