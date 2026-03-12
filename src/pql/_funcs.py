from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc

from . import sql
from ._expr import Expr
from ._meta import SENTINEL_COL, ExprMeta
from .selectors import all_columns_resolver, exclude_resolver, fixed_resolver
from .sql.typing import IntoExpr, IntoExprColumn, PythonLiteral
from .sql.utils import TryIter, try_iter


@final
class Col:
    __slots__ = ()

    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name), ExprMeta(name))

    def __getattr__(self, name: str) -> Expr:
        return self(name)


col: Col = Col()


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression."""
    return Expr(sql.lit(value), ExprMeta("literal"))


def len() -> Expr:
    """Return the number of rows."""
    return Expr(sql.lit(1), ExprMeta("len")).count()


def _agg_expr(
    agg: Callable[[sql.SqlExpr], sql.SqlExpr], columns: Iterable[str]
) -> Expr:
    meta = (
        pc.Seq(columns)
        .then_some()
        .into(
            lambda cols: ExprMeta.from_agg_expr(
                cols, cols.map(fixed_resolver).unwrap_or(all_columns_resolver)
            )
        )
    )
    return Expr(agg(SENTINEL_COL), meta)


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
    return Expr(sql.coalesce(exprs, *more_exprs), ExprMeta(expr_name))


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
    return Expr(SENTINEL_COL, ExprMeta.from_all(resolver))


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.sum_horizontal(exprs, *more_exprs), ExprMeta.from_horizontal(exprs))


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.min_horizontal(exprs, *more_exprs), ExprMeta.from_horizontal(exprs))


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.max_horizontal(exprs, *more_exprs), ExprMeta.from_horizontal(exprs))


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(
        sql.mean_horizontal(exprs, *more_exprs), ExprMeta.from_horizontal(exprs)
    )


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.all_horizontal(exprs, *more_exprs), ExprMeta.from_horizontal(exprs))


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.any_horizontal(exprs, *more_exprs), ExprMeta.from_horizontal(exprs))


_ELEMENT = Expr(sql.element(), ExprMeta("element"))


def element() -> Expr:
    """Alias for an element being evaluated in a list context."""
    return _ELEMENT
