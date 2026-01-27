from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import duckdb
import polars as pl
import pyochain as pc
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from ._expr import Expr

type PyLiteral = str | int | float | bool | None
type FrameInit = (
    duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes
)
type IntoExpr = PyLiteral | Expr | duckdb.Expression


def data_to_rel(data: FrameInit) -> duckdb.DuckDBPyRelation:
    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case pl.DataFrame():
            return duckdb.from_arrow(data)
        case pl.LazyFrame():
            from sqlglot import exp

            _ = data
            return duckdb.sql(exp.select(exp.Star()).from_("_").sql(dialect="duckdb"))

        case None:
            return duckdb.from_arrow(pl.DataFrame({"_": []}))
        case _:
            return duckdb.from_arrow(pl.DataFrame(data))


def to_expr(value: IntoExpr) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case duckdb.Expression():
            return value
        case str():
            return duckdb.ColumnExpression(value)
        case _:
            return duckdb.ConstantExpression(value)


def to_value(value: IntoExpr) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case duckdb.Expression():
            return value
        case _:
            return duckdb.ConstantExpression(value)


def iter_to_exprs(
    *values: IntoExpr | Iterable[IntoExpr],
) -> pc.Iter[duckdb.Expression]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _to_exprs(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[duckdb.Expression]:
        match value:
            case str():
                return pc.Iter.once(to_expr(value))
            case Iterable():
                return pc.Iter(value).map(to_expr)
            case _:
                return pc.Iter.once(to_expr(value))

    match values:
        case (single,):
            return _to_exprs(single)
        case _:
            return pc.Iter(values).map(_to_exprs).flatten()
