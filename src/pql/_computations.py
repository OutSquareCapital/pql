from collections.abc import Callable

import pyochain as pc

from . import sql
from ._typing import FillNullStrategy
from .sql.typing import IntoExpr


def fill_nulls(  # noqa: PLR0911
    expr: sql.SqlExpr,
    value: pc.Option[IntoExpr],
    strategy: pc.Option[FillNullStrategy],
    limit: pc.Option[int],
) -> pc.Result[sql.SqlExpr, ValueError]:

    match (value, strategy, limit):
        case (pc.Some(_), pc.Some(_), _):
            msg = "cannot specify both `value` and `strategy`"
            return pc.Err(ValueError(msg))
        case (_, _, pc.Some(lim)) if lim < 0:
            msg = "Can't process negative `limit` value for fill_null"
            return pc.Err(ValueError(msg))
        case (_, pc.Some("forward") | pc.Some("backward") as strat, pc.Some(lim)):
            iterator = pc.Iter(range(1, lim + 1))
            match strat.value:
                case "forward":
                    exprs = iterator.map(expr.shift)
                case _:
                    exprs = iterator.map(lambda offset: expr.shift(-offset))
            return pc.Ok(exprs.insert(expr).reduce(sql.coalesce))
        case (_, _, pc.Some(_)):
            msg = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
            return pc.Err(ValueError(msg))
        case (pc.Some(val), pc.NONE, pc.NONE):
            return pc.Ok(sql.coalesce(expr, val))
        case (_, pc.Some(strat), pc.NONE):
            return pc.Ok(expr.pipe(FILL_STRATEGY[strat]))
        case _:
            msg = "must specify either a fill `value` or `strategy`"
            return pc.Err(ValueError(msg))


FILL_STRATEGY: pc.Dict[FillNullStrategy, Callable[[sql.SqlExpr], sql.SqlExpr]] = (
    pc.Dict.from_ref(
        {
            "forward": lambda expr: expr.last_value().over(
                frame_end=pc.Some(0), ignore_nulls=True
            ),
            "backward": lambda expr: expr.any_value().over(frame_start=pc.Some(0)),
            "min": lambda expr: sql.coalesce(expr, expr.min().over()),
            "max": lambda expr: sql.coalesce(expr, expr.max().over()),
            "mean": lambda expr: sql.coalesce(expr, expr.mean().over()),
            "zero": lambda expr: sql.coalesce(expr, 0),
            "one": lambda expr: sql.coalesce(expr, 1),
        }
    )
)
"""Computation strategies for `fill_null` when ."""
