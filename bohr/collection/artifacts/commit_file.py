from dataclasses import dataclass
from typing import Optional

from bohr.datamodel.artifact import Artifact

patch = """
@@ -86,7 +86,7 @@
     "html-react-parser": "^0.4.0",
     "lodash": "^4.17.4",
     "mobile-detect": "^1.3.6",
-    "radium": "^0.19.0",
+    "radium": "^0.21.1",
     "react-dom": "^16.0.0",
     "react-event-listener": "^0.4.5",
     "react-onclickoutside": "^6.6.3",
"""


@dataclass
class CommitFile(Artifact):
    """
    >>> commit_file = CommitFile("package.json", "modified", patch, '<eq>"radium": "^0.</eq><ins>2</ins><eq>1</eq><del>9</del><eq>.</eq><re>0<to>1</re><eq>",</eq>')
    >>> commit_file.no_added_lines()
    False
    >>> commit_file.no_removed_lines()
    False
    """

    filename: str
    status: str
    patch: Optional[str]
    changes: Optional[str]

    def no_added_lines(self):  # TODO implementations are not correct
        return "<ins>" not in self.changes

    def no_removed_lines(self):  # TODO implementations are not correct
        return "<del>" not in self.changes
