from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Self, overload, override

import duckdb
import pyochain as pc
from sqlglot import exp

if TYPE_CHECKING:
    from .typing import (
        IntoDuckExpr,
        IntoDuckExprCol,
        IntoExpr,
        IntoExprColumn,
    )


@dataclass(slots=True, repr=False)
class CoreHandler[T]:
    """A wrapper for an inner value.

    Is used as a base class for Expressions, Relation, LazyFrame, and namespaces, since they all share the same pattern of wrapping an inner value and forwarding method calls to it.
    """

    _inner: T

    @override
    def __repr__(self) -> str:
        return self.inner().__repr__()

    @override
    def __str__(self) -> str:
        return self.inner().__str__()

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

    def _new(self, value: T) -> Self:
        """Create a new instance of *Self* with the given value."""
        return self.__class__(value)

    def inner(self) -> T:
        """Unwrap the underlying value."""
        return self._inner


@dataclass(slots=True, repr=False)
class DuckHandler(CoreHandler[exp.Expr]):
    """A wrapper for DuckDB expressions."""

    def into_duckdb(self) -> duckdb.Expression:
        """Convert the inner expression to a DuckDB expression."""
        return duckdb.SQLExpression(self.inner().sql(dialect="duckdb"))


def into_duckdb_mapping(value: Mapping[str, IntoExpr]) -> pc.Dict[str, IntoDuckExpr]:
    return (
        pc.Iter(value.items())
        .iter()
        .map_star(lambda k, v: (k, into_duckdb(v)))
        .collect(pc.Dict)
    )


@overload
def into_duckdb(value: IntoExprColumn) -> IntoDuckExprCol: ...
@overload
def into_duckdb(value: IntoExpr) -> IntoDuckExpr: ...
def into_duckdb(value: IntoExpr | IntoExprColumn) -> IntoDuckExpr | IntoDuckExprCol:
    from .._expr import Expr

    match value:
        case DuckHandler():
            return value.into_duckdb()
        case Expr():
            return value.inner().into_duckdb()
        case exp.Expr():
            return DuckHandler(value).into_duckdb()
        case _:
            return value


@dataclass(slots=True)
class NameSpaceHandler[T: DuckHandler]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _new(self, expr: exp.Expr) -> T:
        return self._parent.__class__(expr)

    def inner(self) -> T:
        """Unwrap the underlying expression."""
        return self._parent


def into_glot(value: IntoExpr) -> exp.Expr:
    """Convert an IntoExpr value into a sqlglot expression node."""
    from .._expr import Expr

    match value:
        case DuckHandler():
            return value.inner()
        case Expr():
            return value.inner().inner()
        case exp.Expr():
            return value
        case str():
            return exp.column(value)
        case _:
            return exp.convert(value)


def func(name: str, *args: IntoExpr) -> exp.Expr:
    """Create a SQL function expression."""
    arguments = pc.Iter(args).filter_map(pc.Option).map(into_glot).collect(list)
    return exp.Anonymous(this=name, expressions=arguments)
