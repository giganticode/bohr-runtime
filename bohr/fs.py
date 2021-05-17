from pathlib import Path
from typing import Optional

from bohr.util.paths import AbsolutePath, RelativePath


def get_preprocessed_path(
    path_preprocessed: Optional[RelativePath],
    path: RelativePath,
    data_dir: RelativePath,
    preprocessor: str,
) -> RelativePath:
    """
    >>> get_preprocessed_path(Path('prep/path.csv'), Path('prep/path.csv.zip'), Path('data'), 'zip').as_posix()
    'data/prep/path.csv'
    >>> get_preprocessed_path(None, Path('prep/path.csv.zip'), Path('data'), 'zip').as_posix()
    'data/prep/path.csv'
    >>> get_preprocessed_path(None, Path('prep/path.csv.foo'), Path('data'), 'foobar').as_posix()
    'data/prep/path.csv.foo'
    """
    if path_preprocessed is not None:
        path_preprocessed = data_dir / path_preprocessed
    elif preprocessor in ["zip", "7z"]:
        *name, ext = str(path).split(".")
        path_preprocessed = data_dir / ".".join(name)
    else:
        path_preprocessed = data_dir / path
    return path_preprocessed


def find_project_root() -> AbsolutePath:
    path = Path(".").resolve()
    current_path = path
    while True:
        lst = list(current_path.glob("bohr.json"))
        if len(lst) > 0 and lst[0].is_file():
            return current_path
        elif current_path == Path("/"):
            raise ValueError(
                f"Not a bohr directory: {path}. "
                f"Bohr config dir is not found in this or any parent directory"
            )
        else:
            current_path = current_path.parent


def gitignore_file(dir: AbsolutePath, filename: str):
    """
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     gitignore_file(Path(tmpdirname), 'file')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n']
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     (Path(tmpdirname) / '.gitignore').touch()
    ...     gitignore_file(Path(tmpdirname), 'file')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n']
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     with open(Path(tmpdirname) / '.gitignore', 'w') as f:
    ...         n = f.write("file\\n")
    ...     gitignore_file(Path(tmpdirname), 'file')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n']
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     with open(Path(tmpdirname) / '.gitignore', 'w') as f:
    ...         n = f.write('file\\n')
    ...     gitignore_file(Path(tmpdirname), 'file2')
    ...     with open(Path(tmpdirname) / '.gitignore') as f:
    ...         print(f.readlines())
    ['file\\n', 'file2\\n']
    """
    path_to_gitignore = dir / ".gitignore"
    path_to_gitignore.touch(exist_ok=True)
    with open(path_to_gitignore, "r") as f:
        lines = list(map(lambda l: l.rstrip("\n"), f.readlines()))
        if filename not in lines:
            with open(path_to_gitignore, "a") as a:
                a.write(f"{filename}\n")
