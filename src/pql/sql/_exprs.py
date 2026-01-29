from duckdb import (
    CaseExpression as when,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    CoalesceOperator as coalesce,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    ColumnExpression as col,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    ConstantExpression as lit,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    DuckDBPyRelation as Relation,  # noqa: F401 # pyright: ignore[reportUnusedImport]
    Expression as SqlExpr,  # pyright: ignore[reportUnusedImport]
    FunctionExpression,
    LambdaExpression as fn_once,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    SQLExpression as raw,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    StarExpression as all,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    from_arrow,  # noqa: F401 # pyright: ignore[reportUnusedImport]
    from_query,  # noqa: F401 # pyright: ignore[reportUnusedImport]
)

from .._types import PyLiteral


def func(name: str, *args: SqlExpr | PyLiteral) -> SqlExpr:
    """Create a SQL function expression."""
    return FunctionExpression(name, *args)  # pyright: ignore[reportArgumentType]
