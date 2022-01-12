import importlib
from pathlib import Path
from typing import NewType, Type, TypeVar

RelativePath = NewType("RelativePath", Path)
AbsolutePath = NewType("AbsolutePath", Path)
RelativeOrAbsolute = TypeVar("RelativeOrAbsolute", RelativePath, AbsolutePath)


def relative_to_safe(
    path: RelativeOrAbsolute, base_path: RelativeOrAbsolute
) -> RelativePath:
    return path.relative_to(base_path)


def concat_paths_safe(p1: RelativeOrAbsolute, p2: RelativePath) -> RelativeOrAbsolute:
    return p1 / p2


def load_class_by_full_path(path_to_mapper_obj: str) -> Type:
    *path, name = path_to_mapper_obj.split(".")
    try:
        module = importlib.import_module(".".join(path))
    except ModuleNotFoundError as e:
        raise ValueError(f'Module {".".join(path)} not defined.') from e
    except ValueError as e:
        raise ValueError(f"Invalid full path: {path_to_mapper_obj}") from e

    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ValueError(f"Class {name} not found in module {module}") from e
