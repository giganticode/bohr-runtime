import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import FileSystemLoader

from bohr.config.pathconfig import PathConfig
from bohr.labeling.hierarchies import LabelHierarchy
from bohr.util.misc import merge_dicts_
from bohr.util.paths import AbsolutePath

logger = logging.getLogger(__name__)

FlattenedMultiHierarchy = Dict[str, List[List[str]]]


def load(f: List[str]) -> FlattenedMultiHierarchy:
    """
    >>> load([])
    {}
    >>> load(["Commit: BugFix, NonBugFix"])
    {'Commit': [['BugFix', 'NonBugFix']]}
    >>> load(["Commit: BugFix, NonBugFix", "Commit: Tangled, NonTangled"])
    {'Commit': [['BugFix', 'NonBugFix'], ['Tangled', 'NonTangled']]}
    >>> load(["Commit: BugFix, NonBugFix", "Commit: Tangled, NonTangled", "BugFix:Minor,Major"])
    {'Commit': [['BugFix', 'NonBugFix'], ['Tangled', 'NonTangled']], 'BugFix': [['Minor', 'Major']]}
    """
    res: FlattenedMultiHierarchy = {}
    for line in f:
        spl_line: List[str] = line.strip("\n").split(":")
        if len(spl_line) != 2:
            raise ValueError(
                f"Invalid line: {line}\n The format must be: Parent: child1, child2, ..., childN"
            )
        parent, children = spl_line
        parent = parent.strip()
        if parent not in res:
            res[parent] = []
        res[parent].append(list(map(lambda x: x.strip(), children.split(","))))
    return res


def build_label_tree(flattened_multi_hierarchy: FlattenedMultiHierarchy) -> LabelHierarchy:
    """
    >>> build_label_tree({})
    Label
    >>> build_label_tree({"Label": [["BugFix", "NonBugFix"]]})
    Label{BugFix|NonBugFix}-> None
    >>> build_label_tree({"Label": [["BugFix", "NonBugFix"], ["Tangled", "NonTangled"]]})
    Label{BugFix|NonBugFix}-> Label{Tangled|NonTangled}-> None
    >>> build_label_tree({"Label": [["BugFix", "NonBugFix"], ["Tangled", "NonTangled"]], "BugFix": [["MinorBugFix"]]})
    Label{BugFix{MinorBugFix}-> None|NonBugFix}-> Label{Tangled|NonTangled}-> None
    """
    tree = LabelHierarchy.create_root("Label")
    pool = [tree]
    while len(pool) > 0:
        node = pool.pop()
        if node.label in flattened_multi_hierarchy:
            children = flattened_multi_hierarchy[node.label]
            children_nodes = node.add_children(children[0])
            pool.extend(children_nodes)
            for other_children in children[1:]:
                node.mounted_hierarchy = LabelHierarchy(node.label, None, [])
                node = node.mounted_hierarchy
                children_nodes = node.add_children(other_children)
                pool.extend(children_nodes)
    return tree


def load_label_tree(path_to_labels: AbsolutePath) -> LabelHierarchy:
    flattened_multi_hierarchy: FlattenedMultiHierarchy = {}
    for label_file in sorted(glob(f"{path_to_labels}/*.txt")):
        with open(label_file, "r") as f:
            merge_dicts_(flattened_multi_hierarchy, load(f.readlines()))
    return build_label_tree(flattened_multi_hierarchy)


def parse_labels(path_config: Optional[PathConfig] = None) -> None:
    path_config = path_config or PathConfig.load()
    label_tree = load_label_tree(path_config.labels)
    from jinja2 import Environment

    env = Environment(loader=FileSystemLoader(Path(__file__).parent.parent))
    template = env.get_template("resources/labels.template")
    s = template.render(hierarchies=label_tree.flatten())
    with open("labels.py", "w") as f:
        f.write(s)
