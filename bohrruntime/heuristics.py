import importlib.util
import inspect
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NewType, Optional, Tuple, Type, Union

import fs
from bohrapi.core import ArtifactType, HeuristicObj
from fs.base import FS
from fs.errors import NoSysPath
from fs.memoryfs import MemoryFS

from bohrruntime.util.paths import AbsolutePath

PathTemplate = NewType("PathTemplate", str)

TEMPLATE_HEURISTIC = """# This is automatically generated code. Do not edit manually.

from bohrapi.core import Heuristic
from bohrlabels.core import OneOrManyLabels

from bohrapi.artifacts import Commit
from bohrlabels.labels import CommitLabel


@Heuristic(Commit)
def always_non_bubfix(commit: Commit) -> OneOrManyLabels:
    return CommitLabel.NonBugFix
"""


def get_template_heuristic() -> str:
    return TEMPLATE_HEURISTIC


def add_template_heuristic(fs: FS, path: str) -> None:
    dir = str(Path(path).parent)
    fs.makedirs(dir)
    with fs.open(path, "w") as f:
        f.write(get_template_heuristic())


@dataclass(frozen=True)
class HeuristicURI:
    path: Path
    fs: FS

    def __str__(self):
        return f"{self.fs}/{self.path}"

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


@dataclass
class HeuristicLoader:
    heuristic_fs: FS

    def get_heuristic_uris(
        self,
        heuristic_uri: HeuristicURI = None,
        input_artifact_type: Optional[ArtifactType] = None,
        error_if_none_found: bool = True,
    ) -> List[HeuristicURI]:
        """
        >>> from fs.memoryfs import MemoryFS
        >>> hl = HeuristicLoader(MemoryFS())
        >>> hl.get_heuristic_uris(error_if_none_found=False)
        []
        >>>
        """
        heuristic_uri = heuristic_uri or HeuristicURI.from_path_and_fs(
            "/", self.heuristic_fs
        )
        heuristic_files: List[HeuristicURI] = []
        if heuristic_uri.is_heuristic_file():
            heuristic_files.append(heuristic_uri)
        else:
            for root, dirs, files in self.heuristic_fs.walk(str(heuristic_uri.path)):
                for file in files:
                    path_to_heuristic: HeuristicURI = HeuristicURI(
                        Path(f"{root}/{file.name}"), self.heuristic_fs
                    )
                    if path_to_heuristic.is_heuristic_file():
                        heuristic_files.append(path_to_heuristic)
        if input_artifact_type is not None:
            res: List[HeuristicURI] = []
            for h in heuristic_files:
                if self.load_heuristics(h, input_artifact_type):
                    res.append(h)
            heuristic_files = res
        if error_if_none_found and len(heuristic_files) == 0:
            raise RuntimeError(
                f"No heuristic groups are found at path: {heuristic_uri}."
            )
        return sorted(heuristic_files)

    def load_all_heuristics(
        self, artifact_type: Type = None
    ) -> Dict[HeuristicURI, List[HeuristicObj]]:
        map: Dict[HeuristicURI, List[HeuristicObj]] = {}

        for heuristic_uri in self.get_heuristic_uris(artifact_type):
            hs = self.load_heuristics(heuristic_uri, artifact_type)
            if len(hs) > 0:
                map[heuristic_uri] = hs
        return map

    @abstractmethod
    def load_heuristics_by_uri(
        self, heuristic_uri: HeuristicURI
    ) -> List[Tuple[str, Union[HeuristicObj, List[HeuristicObj]]]]:
        pass

    def load_heuristics(
        self, heuristic_uri: HeuristicURI, artifact_type: Optional[Type] = None
    ) -> List[HeuristicObj]:

        loaded_heuristics = self.load_heuristics_by_uri(heuristic_uri)

        def is_heuristic_of_needed_type(obj, artifact_type):
            return isinstance(obj, HeuristicObj) and (
                obj.artifact_type_applied_to == artifact_type or artifact_type is None
            )

        heuristics: List[HeuristicObj] = []
        for name, obj in loaded_heuristics:
            if is_heuristic_of_needed_type(obj, artifact_type):
                ext_len = len(".py")
                if name != (filename := heuristic_uri.path.name[:-ext_len]):
                    raise ValueError(
                        f"For consistency, file and heuristic name must be the same.\n"
                        f"Hovewer, filename is {filename}, heuristic name is {name}."
                    )
                heuristics.append(obj)
        for name, obj in loaded_heuristics:
            if (
                isinstance(obj, list)
                and len(obj) > 0
                and (
                    is_heuristic_of_needed_type(obj[0], artifact_type)
                    or artifact_type is None
                )
            ):
                heuristics.extend(obj)
        check_names_unique(heuristics)
        return heuristics


@dataclass
class FileSystemHeuristicLoader(HeuristicLoader):
    def load_heuristics_by_uri(
        self, heuristic_uri: HeuristicURI
    ) -> List[Tuple[str, Union[HeuristicObj, List[HeuristicObj]]]]:
        import importlib.util

        heuristic_file_abs_path = heuristic_uri.to_filesystem_path()
        spec = importlib.util.spec_from_file_location(
            "heuristic.module", heuristic_file_abs_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return [(name, obj) for name, obj in inspect.getmembers(module)]


def check_names_unique(heuristics: List[HeuristicObj]) -> None:
    name_set = set()
    for heuristic in heuristics:
        name = heuristic.func.__name__
        if name in name_set:
            raise ValueError(f"Heuristic with name {name} already exists.")
        name_set.add(name)
