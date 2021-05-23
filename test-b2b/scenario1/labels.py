# This is automatically generated code. Do not edit manually.

from enum import auto

from bohr.labeling.labelset import Label


class CommitLabel(Label):
    MinorBugFix = auto()
    MajorBugFix = auto()
    CriticalBugFix = auto()
    OtherSeverityLevelBugFix = auto()
    BugFix = MinorBugFix | MajorBugFix | CriticalBugFix | OtherSeverityLevelBugFix
    DocFix = auto()
    TestFix = auto()
    BogusFix = DocFix | TestFix
    Refactoring = auto()
    NonBugFix = BogusFix | Refactoring
    CommitLabel = BugFix | NonBugFix

    def parent(self):
        return None


class CommitLabelTangling(Label):
    Tangled = auto()
    NonTangled = auto()
    CommitLabelTangling = Tangled | NonTangled

    def parent(self):
        return CommitLabel.CommitLabel


class SStuB(Label):
    WrongIdentifier = auto()
    WrongNumericLiteral = auto()
    WrongModifier = auto()
    WrongBooleanLiteral = auto()
    WrongFunctionName = auto()
    TooFewArguments = auto()
    TooManyArguments = auto()
    WrongFunction = WrongFunctionName | TooFewArguments | TooManyArguments
    WrongBinaryOperator = auto()
    WrongUnaryOperator = auto()
    WrongOperator = WrongBinaryOperator | WrongUnaryOperator
    MissingThrowsException = auto()
    SStuB = WrongIdentifier | WrongNumericLiteral | WrongModifier | WrongBooleanLiteral | WrongFunction | WrongOperator | MissingThrowsException

    def parent(self):
        return CommitLabelTangling.CommitLabelTangling


class SnippetLabel(Label):
    LongMethod = auto()
    LongParameterList = auto()
    Smelly = LongMethod | LongParameterList
    NonSmelly = auto()
    SnippetLabel = Smelly | NonSmelly

    def parent(self):
        return SStuB.SStuB
