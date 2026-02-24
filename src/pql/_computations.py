from collections.abc import Callable

import pyochain as pc

from . import sql
from ._typing import FillNullStrategy, RoundMode
from .sql.typing import IntoExpr


def round(expr: sql.SqlExpr, decimals: int, mode: RoundMode) -> sql.SqlExpr:
    match mode:
        case "half_to_even":
            return expr.round_even(sql.lit(decimals))
        case "half_away_from_zero":
            return expr.round(decimals)


def shift(expr: sql.SqlExpr, n: int = 1) -> sql.SqlExpr:
    match n:
        case 0:
            return expr
        case n_val if n_val > 0:
            expr = expr.lag(n_val)
        case _:
            expr = expr.lead(-n)
    return expr.over()


def fill_nulls(  # noqa: PLR0911
    expr: sql.SqlExpr,
    value: IntoExpr | None = None,
    strategy: FillNullStrategy | None = None,
    limit: int | None = None,
) -> pc.Result[sql.SqlExpr, ValueError]:

    match (pc.Option(value), pc.Option(strategy), pc.Option(limit)):
        case (pc.Some(_), pc.Some(_), _):
            msg = "cannot specify both `value` and `strategy`"
            return pc.Err(ValueError(msg))
        case (_, _, pc.Some(lim)) if lim < 0:
            msg = "Can't process negative `limit` value for fill_null"
            return pc.Err(ValueError(msg))
        case (_, pc.Some("forward") | pc.Some("backward"), pc.Some(lim)):
            iterator = pc.Iter(range(1, lim + 1))
            match strategy:
                case "forward":
                    exprs = iterator.map(lambda offset: shift(expr, offset))
                case _:
                    exprs = iterator.map(lambda offset: shift(expr, -offset))
            return pc.Ok(exprs.insert(expr).reduce(sql.coalesce))
        case (_, _, pc.Some(_)):
            msg = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
            return pc.Err(ValueError(msg))
        case (pc.Some(val), pc.NONE, pc.NONE):
            return pc.Ok(sql.coalesce(expr, val))
        case (_, pc.Some(strat), pc.NONE):
            return (
                FILL_STRATEGY.get_item(strat)
                .map(lambda f: f(expr))
                .ok_or_else(lambda: ValueError(f"invalid fill_null strategy: {strat}"))
            )
        case _:
            msg = "must specify either a fill `value` or `strategy`"
            return pc.Err(ValueError(msg))


FILL_STRATEGY: pc.Dict[str, Callable[[sql.SqlExpr], sql.SqlExpr]] = pc.Dict.from_ref(
    {
        "forward": lambda expr: expr.last_value().over(rows_end=0, ignore_nulls=True),
        "backward": lambda expr: expr.any_value().over(rows_start=0),
        "min": lambda expr: sql.coalesce(expr, expr.min().over()),
        "max": lambda expr: sql.coalesce(expr, expr.max().over()),
        "mean": lambda expr: sql.coalesce(expr, expr.mean().over()),
        "zero": lambda expr: sql.coalesce(expr, 0),
        "one": lambda expr: sql.coalesce(expr, 1),
    }
)
"""Computation strategies for `fill_null` when ."""
