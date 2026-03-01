from collections.abc import Iterable
from functools import partial

import duckdb
import pyochain as pc

from .._args_iter import try_chain
from ._core import DuckHandler, func, into_duckdb
from ._expr import SqlExpr
from .typing import IntoExpr, IntoExprColumn, PythonLiteral


def row_number() -> SqlExpr:
    """Create a ROW_NUMBER() expression."""
    return SqlExpr(duckdb.FunctionExpression("row_number"))


def unnest(
    col: IntoExprColumn, max_depth: int | None = None, *, recursive: bool = False
) -> SqlExpr:
    """The unnest special function is used to unnest lists or structs by one level.

    The function can be used as a regular scalar function, but only in the SELECT clause.

    Invoking unnest with the recursive parameter will unnest lists and structs of multiple levels.

    The depth of unnesting can be limited using the max_depth parameter (which assumes recursive unnesting by default).

    Using `unnest` on a list emits one row per list entry.

    Regular scalar expressions in the same `SELECT` clause are repeated for every emitted row.

    When multiple lists are unnested in the same `SELECT` clause, the lists are unnested side-by-side.

    If one list is longer than the other, the shorter list is padded with `NULL` values.

    Empty and `NULL` lists both unnest to zero rows.

    Args:
        col (SqlExpr): The column to unnest.
        max_depth (int | None): Maximum depth of recursive unnesting.
        recursive (bool): Whether to recursively unnest lists and structs (default: `False`).  Note that lists *within* structs are not unnested.

    Returns:
        SqlExpr: An expression representing the unnesting operation.
    """
    rec = "recursive:=True" if recursive else None
    depth = f"max_depth:={max_depth}" if max_depth is not None else None
    return SqlExpr(func("unnest", col, depth, rec))


ELEM_NAME = "element"

ELEMENT = SqlExpr(duckdb.ColumnExpression(ELEM_NAME))
LAMBDA_EXPR = partial(duckdb.LambdaExpression, ELEM_NAME)


def element() -> SqlExpr:
    return ELEMENT


def fn_once(rhs: IntoExpr) -> SqlExpr:
    return SqlExpr(LAMBDA_EXPR(into_duckdb(rhs)))


def all(exclude: Iterable[IntoExprColumn] | None = None) -> SqlExpr:
    return (
        pc.Option(exclude)
        .map(lambda x: pc.Iter(x).map(into_duckdb))
        .map(lambda exc: SqlExpr(duckdb.StarExpression(exclude=exc)))
        .unwrap_or(SqlExpr(duckdb.StarExpression()))
    )


class Col:
    def __call__(self, name: str) -> SqlExpr:
        return SqlExpr(duckdb.ColumnExpression(name))

    def __getattr__(self, name: str) -> SqlExpr:
        return self.__call__(name)


col = Col()


def lit(value: PythonLiteral) -> SqlExpr:
    """Create a literal expression."""
    return SqlExpr(duckdb.ConstantExpression(value))


def coalesce(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    """Create a COALESCE expression."""
    return SqlExpr(
        duckdb.CoalesceOperator(*try_chain(exprs, more_exprs).map(into_duckdb))
    )


def into_expr(value: IntoExpr, *, as_col: bool = False) -> SqlExpr:
    """Convert a value to a DuckDB Expression.

    Args:
        value (IntoExpr): The value to convert.
        as_col (bool): Whether to treat `str` values as column names (default: `False`).

    Returns:
        SqlExpr: The resulting DuckDB wrapper Expression.
    """
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case DuckHandler():
            return SqlExpr(value.inner())
        case Expr():
            return value.inner()
        case duckdb.Expression():
            return SqlExpr(value)
        case str() if as_col:
            return col(value)
        case _:
            return lit(value)
