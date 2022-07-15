import importlib
from pathlib import Path
from typing import Callable, List, NewType, Type

from fs.base import FS
from fs.osfs import OSFS

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
    paths: List[str], fs: FS, predicate: Callable[[Path], bool]
) -> List[str]:
    """
    >>> from fs.memoryfs import MemoryFS
    >>> fs = MemoryFS()
    >>> _ = fs.makedirs('root')
    >>> _ = fs.makedirs('root/dir1')
    >>> _ = fs.makedirs('root/dir2')
    >>> _ = fs.touch('root/file0.txt')
    >>> fs.touch('root/dir1/file11.txt')
    >>> fs.touch('root/dir1/file12.txt')
    >>> fs.touch('root/dir2/file21.txt')
    >>> fs.touch('root/dir2/file22.txt')

    >>> absolute_paths = normalize_paths(['root/file0.txt'], fs, lambda x: True)
    >>> [str(Path(path)) for path in absolute_paths]
    ['root/file0.txt']

    >>> absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1/file11.txt'], fs, lambda x: True)
    >>> [str(Path(path)) for path in absolute_paths]
    ['root/dir1/file11.txt', 'root/file0.txt']

    >>> absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1/file11.txt', 'root/dir1/file12.txt'], fs, lambda x: True)
    >>> [str(Path(path)) for path in absolute_paths]
    ['root/dir1', 'root/file0.txt']

    >>> absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1', 'root/dir1/file12.txt'], fs, lambda x: True)
    >>> [str(Path(path)) for path in absolute_paths]
    ['root/dir1', 'root/file0.txt']

    >>> absolute_paths = normalize_paths(['root/file0.txt', 'root/dir1', 'root/dir1/file11.txt', 'root/dir1/file12.txt'], fs, lambda x: True)
    >>> [str(Path(path)) for path in absolute_paths]
    ['root/dir1', 'root/file0.txt']
    """
    non_collapsable = set()

    absolute_paths = [Path(path) for path in paths]
    grouped = {}
    for path in absolute_paths:
        if path.parent not in grouped:
            grouped[path.parent] = set()
        grouped[path.parent].add(path.name)
    while len(grouped) > 0:
        group, children = next(iter(grouped.items()))
        if not fs.exists(str(group)):
            raise ValueError(f"Path {group} does not exist")
        if (
            not group.parent in grouped or not group.name in grouped[group.parent]
        ) and not str(group) in non_collapsable:
            all_children_included = True
            _, dirs, files = next(fs.walk(str(group)))
            for path in dirs + files:
                path = Path(path.name)
                if path.parts[0] not in children and (
                    predicate is None or predicate(path)
                ):
                    all_children_included = False
                    break
            if all_children_included:
                if group.parent not in grouped:
                    grouped[group.parent] = set()
                grouped[group.parent].add(group.name)
            else:
                non_collapsable = non_collapsable.union(
                    [str((group / child)) for child in children]
                )
        del grouped[group]

    return sorted(non_collapsable)


def create_fs() -> FS:
    path = Path(".").resolve()
    current_path = path
    while True:
        lst = list(current_path.glob("bohr.py"))
        if len(lst) > 0 and lst[0].is_file():
            return OSFS(str(current_path))
        elif current_path == Path("/"):
            raise ValueError(
                f"Not a bohr directory: {path}. "
                f"Bohr config dir is not found in this or any parent directory"
            )
        else:
            current_path = current_path.parent


def gitignore_file(fs: FS, filename: str):
    """
    >>> from fs.memoryfs import MemoryFS
    >>> fs = MemoryFS()
    >>> gitignore_file(fs, 'file')
    >>> fs.readtext(".gitignore")
    'file\\n'
    >>> fs = MemoryFS()
    >>> fs.touch('file')
    >>> gitignore_file(fs, 'file')
    >>> fs.readtext(".gitignore")
    'file\\n'
    >>> gitignore_file(fs, 'file')
    >>> fs.readtext(".gitignore")
    'file\\n'
    >>> gitignore_file(fs, 'file2')
    >>> fs.readtext(".gitignore")
    'file\\nfile2\\n'
    """
    fs.touch(".gitignore")
    with fs.open(".gitignore", "r") as f:
        lines = list(map(lambda l: l.rstrip("\n"), f.readlines()))
        if filename not in lines:
            with fs.open(".gitignore", "a") as a:
                a.write(f"{filename}\n")
