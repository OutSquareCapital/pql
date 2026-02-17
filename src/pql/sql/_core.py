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
    """Try to iterate over a value that may or may not be iterable.

    Args:
        val (Iterable[T] | T): The value to try to iterate over.

    Returns:
        pc.Iter[T]: An iterator over the value if it is iterable, otherwise an iterator over a single element.
    """
    match val:
        case str() | bytes() | bytearray():
            return pc.Iter[T].once(val)
        case Iterable():
            return pc.Iter(val)  # pyright: ignore[reportUnknownArgumentType]
        case _:
            return pc.Iter[T].once(val)


def try_flatten[T](vals: T | Iterable[T]) -> pc.Iter[T]:
    """Try to flatten a value that may be nested iterables.

    A value that is not an iterable will be treated as a single element iterable.

    Args:
        vals (T | Iterable[T]): The value to try to flatten.

    Returns:
        pc.Iter[T]: An iterator over the flattened values.
    """
    return try_iter(vals).flat_map(try_iter)


@dataclass(slots=True)
class CoreHandler[T]:
    """A wrapper for an inner value.

    Is used as a base class for Expressions, Relation, LazyFrame, and namespaces, since they all share the same pattern of wrapping an inner value and forwarding method calls to it.
    """

    _inner: T

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


class RelHandler(CoreHandler[duckdb.DuckDBPyRelation]):
    """A wrapper for DuckDB relations."""

    __slots__ = ()

    def __init__(self, data: FrameInit) -> None:
        match data:
            case duckdb.DuckDBPyRelation():
                self._inner = data
            case pl.DataFrame():
                self._inner = duckdb.from_arrow(data)
            case pl.LazyFrame():
                _ = data
                qry = """SELECT * FROM _"""
                self._inner = duckdb.from_query(qry)
            case str() as tbl:
                match tbl:
                    case fn if tbl.endswith("()"):
                        self._inner = duckdb.table_function(fn)
                    case _:
                        self._inner = duckdb.table(data)
            case _:
                self._inner = duckdb.from_arrow(pl.DataFrame(data))


class DuckHandler(CoreHandler[duckdb.Expression]):
    """A wrapper for DuckDB expressions."""

    __slots__ = ()


@dataclass(slots=True)
class NameSpaceHandler[T: DuckHandler]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _new(self, expr: duckdb.Expression) -> T:
        return self._parent.__class__(expr)

    def inner(self) -> duckdb.Expression:
        """Unwrap the underlying expression."""
        return self._parent.inner()


def into_duckdb[T](value: T | DuckHandler) -> T | duckdb.Expression:
    """Convert a value to a DuckDB Expression if it's a SqlExpr, otherwise return it as is."""
    match value:
        case DuckHandler():
            return value.inner()
        case _:
            return value


def func(name: str, *args: Any) -> duckdb.Expression:  # noqa: ANN401
    """Create a SQL function expression."""
    return (
        pc.Iter(args)
        .filter(lambda a: a is not None)
        .map(into_duckdb)
        .into(lambda args: duckdb.FunctionExpression(name, *args))
    )
