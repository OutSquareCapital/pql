from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import polars as pl
import pyochain as pc
from polars._typing import FrameInitTypes

type FrameInit = (
    duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes
)


@dataclass(slots=True)
class ExprHandler[T]:
    """A wrapper for expressions."""

    _expr: T

    def inner(self) -> T:
        """Unwrap the underlying expression."""
        return self._expr


def func(name: str, *args: Any) -> duckdb.Expression:  # noqa: ANN401
    """Create a SQL function expression."""

    def _to_expr(arg: Any) -> duckdb.Expression:  # noqa: ANN401
        from .._expr import Expr
        from ._expr import SqlExpr

        match arg:
            case duckdb.Expression():
                return arg
            case SqlExpr():
                return arg.inner()
            case Expr():
                return arg.inner().inner()
            case str():
                return duckdb.ColumnExpression(arg)
            case _:
                return duckdb.ConstantExpression(arg)

    return (
        pc.Iter(args)
        .filter(lambda a: a is not None)
        .map(_to_expr)
        .into(lambda args: duckdb.FunctionExpression(name, *args))
    )


def rel_from_data(data: FrameInit) -> duckdb.DuckDBPyRelation:
    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case pl.DataFrame():
            return duckdb.from_arrow(data)
        case pl.LazyFrame():
            _ = data
            qry = """SELECT * FROM _"""
            return duckdb.from_query(qry)

        case None:
            return duckdb.from_arrow(pl.DataFrame({"_": ()}))
        case _:
            return duckdb.from_arrow(pl.DataFrame(data))
