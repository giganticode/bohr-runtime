from dataclasses import dataclass
from pathlib import Path
from typing import NewType, Union

from fs.base import FS
from fs.errors import NoSysPath
from fs.memoryfs import MemoryFS

from bohrruntime.util.paths import AbsolutePath

PathTemplate = NewType("PathTemplate", str)


@dataclass(frozen=True)
class HeuristicURI:
    path: Path
    fs: FS

    def __str__(self):
        raise ValueError()

    def to_module(self) -> str:
        return ".".join(str(self.path).replace("/", ".").split(".")[:-1])

    def to_filesystem_path(self) -> AbsolutePath:
        return AbsolutePath(Path(self.fs.getsyspath(str(self.path))))

    # def __truediv__(self, other) -> "HeuristicURI":
    #     if isinstance(other, str) or isinstance(other, Path):
    #         return HeuristicURI(self.path / other)
    #     else:
    #         raise ValueError(
    #             f"Cannot concatenate {HeuristicURI.__name__} and {type(other).__name__}"
    #         )

    @classmethod
    def from_path_and_fs(cls, path: Union[Path, str], fs: FS):
        return HeuristicURI(Path(path), fs)

    def __lt__(self, other) -> bool:
        if isinstance(other, HeuristicURI):
            return str(self.path) < str(other.path)
        else:
            raise ValueError(
                f"Cannot compare {type(self).__name__} and {type(other).__name__}"
            )

    def is_heuristic_file(self) -> bool:
        """
        >>> from pathlib import Path
        >>> fs = MemoryFS()
        >>> heuristics = fs.makedir('heuristics')
        >>> _ = heuristics.makedir('__pycache__')
        >>> heuristics.touch('mult.py')
        >>> heuristics.touch('_mult.py')
        >>> heuristics.touch('mult')
        >>> heuristics.touch('__pycache__/mult.py')
        >>> HeuristicURI.from_path_and_fs('notexist.py', fs).is_heuristic_file()
        False
        >>> HeuristicURI.from_path_and_fs('heuristics/mult.py', fs).is_heuristic_file()
        True
        >>> HeuristicURI.from_path_and_fs('heuristics/_mult.py', fs).is_heuristic_file()
        False
        >>> HeuristicURI.from_path_and_fs('heuristics/mult', fs).is_heuristic_file()
        False
        >>> HeuristicURI.from_path_and_fs('heuristics/__pycache__/mult.py', fs).is_heuristic_file()
        False
        >>> HeuristicURI.from_path_and_fs('heuristics', fs).is_heuristic_file()
        False
        """
        return (
            self.fs.isfile(str(self.path))
            and not str(self.path.name).startswith("_")
            and not str(self.path.parent).endswith("__pycache__")
            and str(self.path.name).endswith(".py")
        )

    def absolute_path(self):
        """
        >>> from fs.memoryfs import MemoryFS
        >>> HeuristicURI.from_path_and_fs('path/file', MemoryFS()).absolute_path()
        'path/file'
        """
        try:
            return self.fs.getsyspath(str(self.path))
        except NoSysPath:
            return str(self.path)
