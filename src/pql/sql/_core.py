from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Self, overload

import duckdb
import pyochain as pc

from ._rel_conversions import frame_init_into_duckdb

if TYPE_CHECKING:
    from .typing import (
        IntoDuckExpr,
        IntoDuckExprCol,
        IntoExpr,
        IntoExprColumn,
        IntoRel,
    )


@dataclass(slots=True)
class CoreHandler[T]:
    """A wrapper for an inner value.

    Is used as a base class for Expressions, Relation, LazyFrame, and namespaces, since they all share the same pattern of wrapping an inner value and forwarding method calls to it.
    """

    _inner: T

    def __repr__(self) -> str:
        return self.inner().__repr__()

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


class DuckHandler(CoreHandler[duckdb.Expression]):
    """A wrapper for DuckDB expressions."""

    __slots__ = ()


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
            return value.inner()
        case Expr():
            return value.inner().inner()
        case _:
            return value


class RelHandler(CoreHandler[duckdb.DuckDBPyRelation]):
    """A wrapper for DuckDB relations."""

    __slots__ = ()

    def __init__(self, data: IntoRel) -> None:
        self._inner = frame_init_into_duckdb(data)


@dataclass(slots=True)
class NameSpaceHandler[T: DuckHandler]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _new(self, expr: duckdb.Expression) -> T:
        return self._parent.__class__(expr)

    def inner(self) -> duckdb.Expression:
        """Unwrap the underlying expression."""
        return self._parent.inner()


def func(name: str, *args: IntoExpr) -> duckdb.Expression:
    """Create a SQL function expression."""
    return (
        pc.Iter(args)
        .filter_map(pc.Option)
        .map(into_duckdb)
        .into(lambda cleaned: duckdb.FunctionExpression(name, *cleaned))
    )
