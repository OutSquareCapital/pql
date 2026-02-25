import pyochain as pc

from .._utils import Builtins, CollectionsABC, DuckDB, Pql, Typing

PYTYPING_REWRITES = pc.Dict.from_ref(
    {
        "pytyping.Any": Typing.ANY,
        "pytyping.SupportsInt": Typing.SUPPORTS_INT,
        "pytyping.List": Builtins.LIST,
        "pytyping.Literal": Typing.LITERAL,
        "pytyping.Union": Typing.UNION,
        "pytorch": "torch",
        "pyarrow.lib.": "pa.",
        "pyarrow.": "pa.",
    }
)

TYPE_SUBS = pc.Dict({DuckDB.EXPRESSION: Pql.DUCK_HANDLER, DuckDB.RELATION: Typing.SELF})

EXPR_TYPE_SUBS = pc.Dict(
    {DuckDB.EXPRESSION: Typing.SELF, DuckDB.RELATION: Pql.RELATION}
)


PARAM_TYPE_FIXES = pc.Dict.from_kwargs(
    aggregate=pc.Dict.from_kwargs(
        aggr_expr=DuckDB.EXPRESSION.into_union(
            Builtins.STR,
            CollectionsABC.ITERABLE.of_type(DuckDB.EXPRESSION, Builtins.STR),
        )
    )
)

RETURN_TYPE_FIXES = pc.Dict.from_kwargs(
    dtypes=Builtins.LIST.of_type("sqltypes.DuckDBPyType")
)

KW_ONLY_FIXES = pc.Set({"write_csv", "write_parquet"})
