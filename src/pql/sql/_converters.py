from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import pyochain as pc

from ._exprs import SqlExpr, col, lit

if TYPE_CHECKING:
    from .._types import IntoExpr


def from_expr(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from .._expr import Expr

    match value:
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
        case Expr():
            return value.expr
        case _:
            return lit(value)


def from_iter(
    *values: IntoExpr | Iterable[IntoExpr],
) -> pc.Iter[SqlExpr]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _to_exprs(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
        match value:
            case str():
                return pc.Iter.once(from_expr(value))
            case Iterable():
                return pc.Iter(value).map(from_expr)
            case _:
                return pc.Iter.once(from_expr(value))

    match values:
        case (single,):
            return _to_exprs(single)
        case _:
            return pc.Iter(values).map(_to_exprs).flatten()


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
