from collections.abc import Callable, Iterable

import pyochain as pc

from . import sql
from ._args_iter import TryIter, try_iter
from ._expr import Expr, ExprMeta
from .sql.typing import IntoExpr, IntoExprColumn, PythonLiteral


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name))

    def __getattr__(self, name: str) -> Expr:
        return self(name)


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression."""
    return Expr(sql.lit(value))


def len() -> Expr:
    """Return the number of rows."""
    return Expr(sql.lit(1), pc.Some(ExprMeta("len"))).count()


def _agg_expr(
    agg: Callable[[sql.SqlExpr], sql.SqlExpr], columns: tuple[str, ...]
) -> Expr:
    selected_names = pc.Seq(columns).then_some()
    return Expr(
        selected_names.map(lambda vals: sql.col(vals.first())).unwrap_or_else(
            lambda: sql.lit(0)
        ),
        pc.Some(
            ExprMeta(
                selected_names.map(lambda vals: vals.first()).unwrap_or("all"),
                is_scalar_like=True,
                is_multi=True,
                selected_names=selected_names,
                multi_agg=pc.Some(agg),
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


col: Col = Col()


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """Create a coalesce expression."""
    return Expr(sql.coalesce(exprs, *more_exprs))


def all(exclude: Iterable[IntoExprColumn] | None = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    inner = sql.all(exclude)
    excluded_names = (
        pc.Option(exclude)
        .map(
            lambda exc: (
                pc.Iter(exc)
                .map(lambda value: sql.into_expr(value, as_col=True).get_name())
                .collect(pc.Set)
            )
        )
        .unwrap_or_else(pc.Set[str].new)
    )
    meta = pc.Some(
        ExprMeta(inner.get_name(), is_multi=True, excluded_names=excluded_names)
    )
    return Expr(inner, meta)


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
