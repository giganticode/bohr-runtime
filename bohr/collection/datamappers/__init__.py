from bohr.collection.artifacts import Commit, CommitFile, Issue, Method
from bohr.collection.datamappers.commit import CommitMapper
from bohr.collection.datamappers.commitfile import CommitFileMapper
from bohr.collection.datamappers.issue import IssueMapper
from bohr.collection.datamappers.manuallabels import ManualLabelMapper
from bohr.collection.datamappers.method import MethodMapper
from bohr.labeling.labelset import Label

default_mappers = {
    Commit: CommitMapper,
    CommitFile: CommitFileMapper,
    Issue: IssueMapper,
    Label: ManualLabelMapper,
    Method: MethodMapper,
}
