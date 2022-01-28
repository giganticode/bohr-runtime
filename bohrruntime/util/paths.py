import importlib
import os
from pathlib import Path
from typing import Callable, List, NewType, Type, TypeVar

AbsolutePath = NewType("AbsolutePath", Path)


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


def normalize_paths(
    paths: List[str], base_dir: AbsolutePath, predicate: Callable
) -> List[str]:
    """
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     os.makedirs(tmpdirname / Path('root'))
    ...     os.makedirs(tmpdirname / Path('root/dir1'))
    ...     os.makedirs(tmpdirname / Path('root/dir2'))
    ...     open(tmpdirname / Path('root/file0.txt'), 'a').close()
    ...     open(tmpdirname / Path('root/dir1/file11.txt'), 'a').close()
    ...     open(tmpdirname / Path('root/dir1/file12.txt'), 'a').close()
    ...     open(tmpdirname / Path('root/dir2/file21.txt'), 'a').close()
    ...     open(tmpdirname / Path('root/dir2/file22.txt'), 'a').close()
    ...
    ...     absolute_paths = normalize_paths(['root/file0.txt'], Path(tmpdirname), lambda x: True)
    ...     res1 = [str(Path(path)) for path in absolute_paths]
    ...
    ...     absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1/file11.txt'], Path(tmpdirname), lambda x: True)
    ...     res2 = [str(Path(path)) for path in absolute_paths]
    ...     absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1/file11.txt', 'root/dir1/file12.txt'], Path(tmpdirname), lambda x: True)
    ...     res3 = [str(Path(path)) for path in absolute_paths]
    ...     absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1', 'root/dir1/file12.txt'], Path(tmpdirname), lambda x: True)
    ...     res4 = [str(Path(path)) for path in absolute_paths]
    ...     absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1', 'root/dir1/file11.txt', 'root/dir1/file12.txt'], Path(tmpdirname), lambda x: True)
    ...     res5 = [str(Path(path)) for path in absolute_paths]
    ...     res1, res2, res3, res4, res5
    (['root/file0.txt'], ['root/dir1/file11.txt', 'root/file0.txt'], ['root/dir1', 'root/file0.txt'], ['root/dir1', 'root/file0.txt'], ['root/dir1', 'root/file0.txt'])
    """
    non_collapsable = set()

    absolute_paths = [base_dir / path for path in paths]
    grouped = {}
    for path in absolute_paths:
        if path.parent not in grouped:
            grouped[path.parent] = set()
        grouped[path.parent].add(path.name)
    while len(grouped) > 0:
        group, children = next(iter(grouped.items()))
        if not group.exists():
            raise ValueError(f"Path {group} does not exist")
        if (
            not group.parent in grouped or not group.name in grouped[group.parent]
        ) and not str(group.relative_to(base_dir)) in non_collapsable:
            all_children_included = True
            _, dirs, files = next(os.walk(str(group)))
            for file in files + dirs:
                path = Path(file)
                if path.parts[0] not in children and predicate(path):
                    all_children_included = False
                    break
            if all_children_included:
                if group.parent not in grouped:
                    grouped[group.parent] = set()
                grouped[group.parent].add(group.name)
            else:
                non_collapsable = non_collapsable.union(
                    [str((group / child).relative_to(base_dir)) for child in children]
                )
        del grouped[group]

    return sorted(non_collapsable)
