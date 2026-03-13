from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc

from . import sql
from ._expr import Expr
from ._meta import SENTINEL_COL, ExprMeta, agg_expr_resolver, all_fn_resolver
from .sql.typing import IntoExpr, IntoExprColumn, PythonLiteral
from .sql.utils import TryIter, try_chain, try_iter


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
    agg: Callable[[sql.SqlExpr], sql.SqlExpr],
    cols: TryIter[str],
    more_cols: Iterable[str],
) -> Expr:
    meta = (
        try_chain(cols, more_cols)
        .collect()
        .then_some()
        .into(lambda cols: ExprMeta.from_agg_expr(cols, agg_expr_resolver(cols)))
    )
    return Expr(agg(SENTINEL_COL), meta)


def sum(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.SqlExpr.sum, cols, more_cols)


def mean(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.SqlExpr.mean, cols, more_cols)


def median(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.SqlExpr.median, cols, more_cols)


def min(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.SqlExpr.min, cols, more_cols)


def max(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.SqlExpr.max, cols, more_cols)


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """Create a coalesce expression."""
    expr_name = (
        try_iter(exprs).next().map(sql.into_expr, as_col=True).unwrap().get_name()
    )
    return Expr(sql.coalesce(exprs, *more_exprs), ExprMeta(expr_name))


def all(exclude: Iterable[IntoExprColumn] | None = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    return Expr(SENTINEL_COL, ExprMeta.from_all(all_fn_resolver(pc.Option(exclude))))


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
