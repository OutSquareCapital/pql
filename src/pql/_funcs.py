from collections.abc import Callable, Iterable
from functools import partial
from typing import final

import pyochain as pc

from . import sql
from ._expr import Expr
from ._meta import (
    SENTINEL_COL,
    ExprKind,
    ExprMeta,
    MultiExpansion,
    replay_transform,
)
from .selectors import all_columns_resolver, exclude_resolver, fixed_resolver
from .sql.typing import IntoExpr, IntoExprColumn, PythonLiteral
from .sql.utils import TryIter, try_iter


@final
class Col:
    __slots__ = ()

    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name), pc.Some(ExprMeta(name)))

    def __getattr__(self, name: str) -> Expr:
        return self(name)


col: Col = Col()


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression."""
    return Expr(sql.lit(value))


def len() -> Expr:
    """Return the number of rows."""
    return Expr(sql.lit(1), pc.Some(ExprMeta("len"))).count()


def _agg_expr(
    agg: Callable[[sql.SqlExpr], sql.SqlExpr], columns: tuple[str, ...]
) -> Expr:
    cols = pc.Seq(columns)
    resolver = cols.then(fixed_resolver).unwrap_or(all_columns_resolver)
    inner = agg(SENTINEL_COL)
    return Expr(
        inner,
        pc.Some(
            ExprMeta(
                cols.then(lambda c: c.first()).unwrap_or("all"),
                kind=ExprKind.SCALAR,
                expansion=pc.Some(
                    MultiExpansion(resolver, partial(replay_transform, inner))
                ),
            )
        ),
    )


def sum(*columns: str) -> Expr:
    return _agg_expr(sql.SqlExpr.sum, columns)


def mean(*columns: str) -> Expr:
    return _agg_expr(sql.SqlExpr.mean, columns)


def median(*columns: str) -> Expr:
    return _agg_expr(sql.SqlExpr.median, columns)


def min(*columns: str) -> Expr:
    return _agg_expr(sql.SqlExpr.min, columns)


def max(*columns: str) -> Expr:
    return _agg_expr(sql.SqlExpr.max, columns)


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """Create a coalesce expression."""
    expr_name = (
        try_iter(exprs).next().map(sql.into_expr, as_col=True).unwrap().get_name()
    )
    return Expr(sql.coalesce(exprs, *more_exprs), pc.Some(ExprMeta(expr_name)))


def all(exclude: Iterable[IntoExprColumn] | None = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    resolver = (
        pc.Option(exclude)
        .map(
            lambda exc: (
                pc.Iter(exc)
                .map(lambda value: sql.into_expr(value, as_col=True).get_name())
                .collect(pc.Set)
                .into(exclude_resolver)
            )
        )
        .unwrap_or(all_columns_resolver)
    )
    return Expr(
        SENTINEL_COL,
        pc.Some(
            ExprMeta(
                "all",
                expansion=pc.Some(
                    MultiExpansion(resolver, partial(replay_transform, SENTINEL_COL))
                ),
            )
        ),
    )


def _horizontal_meta(exprs: TryIter[IntoExpr]) -> pc.Option[ExprMeta]:
    return (
        try_iter(exprs)
        .next()
        .map(lambda v: sql.into_expr(v, as_col=True).get_name())
        .map(ExprMeta)
    )


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.sum_horizontal(exprs, *more_exprs), _horizontal_meta(exprs))


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.min_horizontal(exprs, *more_exprs), _horizontal_meta(exprs))


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.max_horizontal(exprs, *more_exprs), _horizontal_meta(exprs))


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.mean_horizontal(exprs, *more_exprs), _horizontal_meta(exprs))


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.all_horizontal(exprs, *more_exprs), _horizontal_meta(exprs))


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.any_horizontal(exprs, *more_exprs), _horizontal_meta(exprs))


_ELEMENT = Expr(sql.element())


def element() -> Expr:
    """Alias for an element being evaluated in a list context."""
    return _ELEMENT
