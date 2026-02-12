from enum import StrEnum, auto

import pyochain as pc


class PyLit(StrEnum):
    INTO_DUCKDB = auto()
    SQLEXPR = "SqlExpr"
    DUCK_REL = "DuckDBPyRelation"
    DUCK_EXPR = "Expression"
    NONE = "None"
    ANY = "Any"
    SELF_RET = "Self"
    SELF = auto()
    STR = auto()


PYTYPING_REWRITES: pc.Dict[str, str] = pc.Dict.from_ref(
    {
        "pytyping.Any": PyLit.ANY,
        "pytyping.SupportsInt": "SupportsInt",
        "pytyping.List": "list",
        "pytyping.Literal": "Literal",
        "pytyping.Union": "Union",
        "pytorch": "torch",
        "pyarrow.lib.": "pa.",
        "pyarrow.": "pa.",
    }
)

TYPE_SUBS: pc.Dict[PyLit, PyLit] = pc.Dict.from_ref(
    {PyLit.DUCK_EXPR: PyLit.SQLEXPR, PyLit.DUCK_REL: PyLit.SELF_RET}
)
