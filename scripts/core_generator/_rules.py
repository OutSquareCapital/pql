import pyochain as pc

from .._utils import Builtins, DuckDB, Pql, Typing

_PA = "pa."
PYTYPING_REWRITES = pc.Dict.from_ref(
    {
        "typing.Any": Typing.ANY,
        "typing.SupportsInt": Typing.SUPPORTS_INT,
        "typing.List": Builtins.LIST,
        "typing.Literal": Typing.LITERAL,
        "typing.Union": Typing.UNION,
        "pyarrow.lib.": _PA,
        "pyarrow.": _PA,
    }
)

TYPE_SUBS = pc.Dict({DuckDB.EXPRESSION: Pql.DUCK_HANDLER, DuckDB.RELATION: Typing.SELF})

EXPR_TYPE_SUBS = pc.Dict(
    {DuckDB.EXPRESSION: Typing.SELF, DuckDB.RELATION: Pql.RELATION}
)
