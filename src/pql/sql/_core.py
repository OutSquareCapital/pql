from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, Self

import duckdb
import polars as pl
import pyochain as pc

if TYPE_CHECKING:
    from ._typing import FrameInit


def try_iter[T](val: Iterable[T] | T) -> pc.Iter[T]:
    match val:
        case str():
            return pc.Iter[T].once(val)
        case Iterable():
            return pc.Iter(val)  # pyright: ignore[reportUnknownArgumentType]
        case _:
            return pc.Iter[T].once(val)


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


class RelHandler(ExprHandler[duckdb.DuckDBPyRelation]):
    """A wrapper for DuckDB relations."""

    __slots__ = ()

    def __init__(self, data: FrameInit) -> None:
        match data:
            case duckdb.DuckDBPyRelation():
                self._expr = data
            case pl.DataFrame():
                self._expr = duckdb.from_arrow(data)
            case pl.LazyFrame():
                _ = data
                qry = """SELECT * FROM _"""
                self._expr = duckdb.from_query(qry)
            case str() as tbl:
                match tbl:
                    case fn if tbl.endswith("()"):
                        self._expr = duckdb.table_function(fn)
                    case _:
                        self._expr = duckdb.table(data)
            case _:
                self._expr = duckdb.from_arrow(pl.DataFrame(data))


class DuckHandler(ExprHandler[duckdb.Expression]):
    """A wrapper for DuckDB expressions."""

    __slots__ = ()


def iter_into_duckdb[T](
    value: Iterable[T | DuckHandler],
) -> pc.Iter[T | duckdb.Expression]:
    """Convert an iterable of values to an iterable of DuckDB Expressions, converting SqlExpr as needed."""
    return pc.Iter(value).map(into_duckdb)


def into_duckdb[T](value: T | DuckHandler) -> T | duckdb.Expression:
    """Convert a value to a DuckDB Expression if it's a SqlExpr, otherwise return it as is."""
    match value:
        case DuckHandler():
            return value.inner()
        case _:
            return value


def any_into_duckdb(arg: Any) -> duckdb.Expression:  # noqa: ANN401
    from .._expr import Expr

    match arg:
        case duckdb.Expression():
            return arg
        case DuckHandler():
            return arg.inner()
        case Expr():
            return arg.inner().inner()
        case str():
            return duckdb.ColumnExpression(arg)
        case _:
            return duckdb.ConstantExpression(arg)


@dataclass(slots=True)
class NameSpaceHandler[T: DuckHandler]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _new(self, expr: duckdb.Expression) -> T:
        return self._parent.__class__(expr)

    def inner(self) -> duckdb.Expression:
        """Unwrap the underlying expression."""
        return self._parent.inner()


def func(name: str, *args: Any) -> duckdb.Expression:  # noqa: ANN401
    """Create a SQL function expression."""
    return (
        pc.Iter(args)
        .filter(lambda a: a is not None)
        .map(any_into_duckdb)
        .into(lambda args: duckdb.FunctionExpression(name, *args))
    )
