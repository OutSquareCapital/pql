from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from .._expr import Expr


import pyochain as pc
from duckdb import (
    CaseExpression as when,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    CoalesceOperator as coalesce,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    ColumnExpression as col,  # noqa: N813 # pyright: ignore[reportUnusedImport]
    ConstantExpression as lit,  # noqa: N813
    DuckDBPyRelation as Relation,
    Expression as SqlExpr,
    FunctionExpression,
    LambdaExpression as fn_once,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    SQLExpression as raw,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    StarExpression as all,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    from_arrow,
    from_query,
)

type FrameInit = Relation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes

type IntoExprColumn = Iterable[SqlExpr] | SqlExpr | str

type PyLiteral = (
    str
    | int
    | float
    | bool
    | date
    | datetime
    | time
    | timedelta
    | bytes
    | bytearray
    | memoryview
    | list[PyLiteral]
    | dict[Any, PyLiteral]
    | None
)
type IntoExpr = PyLiteral | Expr | SqlExpr


def from_expr(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case Expr():
            return value.expr
        case str():
            return col(value)
        case _:
            return lit(value)


def from_value(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case Expr():
            return value.expr
        case _:
            return lit(value)


def from_cols(exprs: IntoExprColumn) -> Iterable[SqlExpr | str]:
    """Convert one or more values or iterables of values to an iterable of DuckDB Expressions or strings."""
    match exprs:
        case str() | SqlExpr():
            return (exprs,)
        case Iterable():
            return exprs


def from_iter(*values: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _single_to_expr(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
        match value:
            case str() | bytes() | bytearray():
                return pc.Iter.once(from_expr(value))
            case Iterable():
                return pc.Iter(value).map(from_expr)
            case _:
                return pc.Iter.once(from_expr(value))

    match values:
        case (single,):
            return _single_to_expr(single)
        case _:
            return pc.Iter(values).map(_single_to_expr).flatten()


def rel_from_data(data: FrameInit) -> Relation:
    match data:
        case Relation():
            return data
        case pl.DataFrame():
            return from_arrow(data)
        case pl.LazyFrame():
            _ = data
            qry = """SELECT * FROM _"""
            return from_query(qry)

        case None:
            return from_arrow(pl.DataFrame({"_": ()}))
        case _:
            return from_arrow(pl.DataFrame(data))


def from_args_kwargs(
    *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
) -> pc.Iter[SqlExpr]:
    """Convert positional and keyword arguments to an iterator of DuckDB Expressions."""
    return from_iter(*exprs).chain(
        pc.Dict.from_ref(named_exprs)
        .items()
        .iter()
        .map_star(lambda name, expr: from_expr(expr).alias(name))
    )


def func(name: str, *args: Any) -> SqlExpr:  # noqa: ANN401
    """Create a SQL function expression."""
    return (
        pc.Iter(args)
        .filter(lambda a: a is not None)
        .map(from_expr)
        .into(lambda args: FunctionExpression(name, *args))
    )
