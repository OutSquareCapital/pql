from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import duckdb
import polars as pl
import pyochain as pc
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from ._expr import Expr

type FrameInit = (
    duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes
)


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


def to_expr(value: object) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case str():
            return duckdb.ColumnExpression(value)
        case _:
            return duckdb.ConstantExpression(value)


def to_value(value: object) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case _:
            return duckdb.ConstantExpression(value)


def iter_to_exprs(
    values: str | Expr | Iterable[Expr] | Iterable[str],
) -> pc.Iter[str | Expr] | pc.Iter[Expr]:
    from ._expr import Expr

    match values:
        case str() | Expr():
            return pc.Iter.once(values)
        case Iterable():
            return pc.Iter(values)
