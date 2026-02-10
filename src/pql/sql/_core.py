from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, Self

import duckdb
import polars as pl
import pyochain as pc

if TYPE_CHECKING:
    from ._typing import FrameInit


@dataclass(slots=True)
class ExprHandler[T]:
    """A wrapper for expressions."""

    _expr: T

    def pipe[**P, R](
        self,
        function: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Apply a *function* to *Self* with *args* and *kwargs*.

        Allow to do `x.pipe(func, ...)` instead of `func(x, ...)`.

        This keep a fluent style for UDF, and is shared across `Expr` and `LazyFrame` objects.

        This is similar to **polars** `.pipe` method.

        Args:
            function (Callable[Concatenate[Self, P], R]): The *function* to apply.
            *args (P.args): Positional arguments to pass to *function*.
            **kwargs (P.kwargs): Keyword arguments to pass to *function*.

        Returns:
            R: The result of applying the *function*.
        """
        return function(self, *args, **kwargs)

    def _new(self, expr: T) -> Self:
        return self.__class__(expr)

    def inner(self) -> T:
        """Unwrap the underlying expression."""
        return self._expr


@dataclass(slots=True)
class NameSpaceHandler[T: ExprHandler[duckdb.Expression]]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _new(self, expr: duckdb.Expression) -> T:
        return self._parent.__class__(expr)

    def inner(self) -> duckdb.Expression:
        """Unwrap the underlying expression."""
        return self._parent.inner()


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
        case _:
            return duckdb.from_arrow(pl.DataFrame(data))
