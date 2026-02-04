from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._types import IntoExpr


import pyochain as pc
from duckdb import (
    CaseExpression as when,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    CoalesceOperator as coalesce,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    ColumnExpression as col,  # noqa: N813 # pyright: ignore[reportUnusedImport]
    ConstantExpression as lit,  # noqa: N813 # pyright: ignore[reportUnusedImport]
    DuckDBPyRelation as Relation,  # noqa: F401 # pyright: ignore[reportUnusedImport]
    Expression as SqlExpr,  # pyright: ignore[reportUnusedImport]
    FunctionExpression,
    LambdaExpression as fn_once,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    SQLExpression as raw,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    StarExpression as all,  # noqa: F401, N813 # pyright: ignore[reportUnusedImport]
    from_arrow,  # noqa: F401 # pyright: ignore[reportUnusedImport]
    from_query,  # noqa: F401 # pyright: ignore[reportUnusedImport]
)

type IterExpr = IntoExpr | Iterable[IntoExpr]


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


def from_iter(*values: IterExpr) -> pc.Iter[SqlExpr]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _to_exprs(value: IterExpr) -> pc.Iter[SqlExpr]:
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


def from_args_kwargs(*exprs: IterExpr, **named_exprs: IntoExpr) -> pc.Iter[SqlExpr]:
    """Convert positional and keyword arguments to an iterator of DuckDB Expressions."""
    return from_iter(*exprs).chain(
        pc.Dict.from_ref(named_exprs)
        .items()
        .iter()
        .map_star(lambda name, expr: from_expr(expr).alias(name))
    )


def func(name: str, *args: object) -> SqlExpr:
    """Create a SQL function expression."""
    return FunctionExpression(name, *map(from_expr, args))  # pyright: ignore[reportArgumentType]
