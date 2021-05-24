import logging
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import FileSystemLoader

from bohr.config.pathconfig import PathConfig
from bohr.labeling.hierarchies import LabelHierarchy
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
    Traceback (most recent call last):
    ...
    ValueError: Parent Commit has at least two hierarchies without having classification type specified
    >>> load(["Commit: BugFix, NonBugFix", "Commit: Tangled"])
    Traceback (most recent call last):
    ...
    ValueError: Commit has to have more than one child.
    >>> load(["Commit: BugFix, NonBugFix", "Commit(CommitTangling): Tangled, NonTangled"])
    {'Commit': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']]}
    >>> load(["Commit: BugFix, NonBugFix", "Commit(CommitTangling): Tangled, NonTangled", "BugFix:Minor,Major"])
    {'Commit': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']], 'BugFix': [['Minor', 'Major']]}
    >>> load(["BugFix, NonBugFix"])
    Traceback (most recent call last):
    ...
    ValueError: Invalid line: BugFix, NonBugFix
     The format must be: Parent: child1, child2, ..., childN
    >>> load(["Commit() : BugFix, NonBugFix"])
    Traceback (most recent call last):
    ...
    ValueError: Invalid parent format: Commit() .
    """
    res: FlattenedMultiHierarchy = {}

    def add_parent_and_children(parent, children, res):
        parent = parent.strip()
        if parent not in res:
            res[parent] = []
        elif isinstance(children, list):
            for elm in res[parent]:
                if isinstance(elm, list):
                    raise ValueError(f'Parent {parent} has at least two hierarchies without having classification type specified')
        res[parent].append(children)

    for line in f:
        spl_line: List[str] = line.strip("\n").split(":")
        if len(spl_line) != 2:
            raise ValueError(
                f"Invalid line: {line}\n The format must be: Parent: child1, child2, ..., childN"
            )
        left, right = spl_line
        split_list = list(map(lambda x: x.strip(), right.split(",")))
        if len(split_list) < 2:
            raise ValueError(f"{left} has to have more than one child.")
        if re.match('^\\w+$', left):
            add_parent_and_children(left, split_list, res)
        else:
            m = re.match('^(\\w+)\((\\w+)\)$', left)
            if m is None:
                raise ValueError(f"Invalid parent format: {left}.")
            add_parent_and_children(m.group(1), m.group(2), res)
            add_parent_and_children(m.group(2), split_list, res)
    return res


def build_label_tree(flattened_multi_hierarchy: FlattenedMultiHierarchy, top_label: str = 'Label') -> LabelHierarchy:
    """
    >>> build_label_tree({})
    Label
    >>> build_label_tree({"Label": [["BugFix", "NonBugFix"]]})
    Label{BugFix|NonBugFix}-> None
    >>> build_label_tree({'Label': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']]})
    Label{BugFix|NonBugFix}-> CommitTangling{Tangled|NonTangled}-> None
    >>> build_label_tree({'Label': [['BugFix', 'NonBugFix'], 'CommitTangling'], 'CommitTangling': [['Tangled', 'NonTangled']], 'BugFix': [['Minor']]})
    Label{BugFix{Minor}-> None|NonBugFix}-> CommitTangling{Tangled|NonTangled}-> None
    """
    tree = LabelHierarchy.create_root(top_label)
    pool = [tree]
    while len(pool) > 0:
        node = pool.pop()
        if node.label in flattened_multi_hierarchy:
            children = flattened_multi_hierarchy[node.label]
            children_nodes = node.add_children(children[0])
            pool.extend(children_nodes)
            for other_children in children[1:]:
                node.mounted_hierarchy = LabelHierarchy(other_children, None, [])
                node = node.mounted_hierarchy
                pool.append(node)
    return tree


def load_label_tree(path_to_labels: AbsolutePath) -> LabelHierarchy:
    top = None
    for label_file in sorted(glob(f"{path_to_labels}/*.txt")):
        with open(label_file, "r") as f:
            if top is None:
                top = build_label_tree(load(f.readlines()), Path(label_file).stem)
                tree = top
            else:
                while tree.mounted_hierarchy is not None:
                    tree = tree.mounted_hierarchy
                tree.mounted_hierarchy = build_label_tree(load(f.readlines()), Path(label_file).stem)
    return top


def parse_labels(path_config: Optional[PathConfig] = None) -> None:
    path_config = path_config or PathConfig.load()
    label_tree = load_label_tree(path_config.labels)
    from jinja2 import Environment

    env = Environment(loader=FileSystemLoader(Path(__file__).parent.parent))
    template = env.get_template("resources/labels.template")
    s = template.render(hierarchies=label_tree.flatten())
    with open("labels.py", "w") as f:
        f.write(s)
