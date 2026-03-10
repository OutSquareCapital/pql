from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import duckdb
import pyochain as pc

from .utils import try_iter

if TYPE_CHECKING:
    from ._expr import SqlExpr
    from .typing import FrameMode, IntoExprColumn, WindowExclude

type FrameBound = int | str

_EXCLUDE_CLAUSE: pc.Dict[WindowExclude, str] = pc.Dict.from_ref(
    {
        "current_row": "EXCLUDE CURRENT ROW",
        "group": "EXCLUDE GROUP",
        "ties": "EXCLUDE TIES",
        "no_others": "EXCLUDE NO OTHERS",
    }
)


class Kword:
    @classmethod
    def frame_clause(
        cls,
        mode: FrameMode,
        frame_start: pc.Option[FrameBound],
        frame_end: pc.Option[FrameBound],
        *,
        has_order_by: bool,
    ) -> str:
        match (frame_start, frame_end):
            case (pc.Some(start), pc.Some(end)):
                return f"{mode} BETWEEN {_bound(start, 'PRECEDING')} AND {_bound(end, 'FOLLOWING')}"
            case (pc.Some(start), pc.NONE):
                return f"{mode} BETWEEN {_bound(start, 'PRECEDING')} AND UNBOUNDED FOLLOWING"
            case (pc.NONE, pc.Some(end)):
                return (
                    f"{mode} BETWEEN UNBOUNDED PRECEDING AND {_bound(end, 'FOLLOWING')}"
                )
            case _ if has_order_by:
                return f"{mode} BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
            case _:
                return ""

    @classmethod
    def exclude_clause(cls, exclude: WindowExclude) -> str:
        return _EXCLUDE_CLAUSE.get_item(exclude).expect("invalid exclude option")

    @classmethod
    def null_order(cls, *, last: bool) -> str:
        match last:
            case True:
                return "NULLS LAST"
            case False:
                return "NULLS FIRST"

    @classmethod
    def sort_order(cls, *, desc: bool) -> str:
        match desc:
            case True:
                return "DESC"
            case False:
                return "ASC"

    @classmethod
    def partition_by(cls, by: str) -> str:
        return f"""
        PARTITION BY {by}"""

    @classmethod
    def order_by(cls, by: str) -> str:
        return f"""
        ORDER BY {by}"""

    @classmethod
    def sort_strat(cls, item: object, *, desc: bool, nulls_last: bool) -> str:
        return f"{item} {cls.sort_order(desc=desc)} {cls.null_order(last=nulls_last)}"


def _bound(value: FrameBound, direction: str) -> str:
    """Convert a frame bound value into SQL syntax.

    - 0 -> CURRENT ROW
    - positive int -> N {direction}
    - negative int for PRECEDING -> becomes positive FOLLOWING, etc.
    - str -> raw SQL (e.g. "INTERVAL 3 DAYS") + direction
    """
    match value:
        case 0:
            return "CURRENT ROW"
        case int(n):
            return f"{abs(n)} {direction}"
        case str(raw):
            return f"{raw} {direction}"


def over_expr(  # noqa: PLR0913
    expr: SqlExpr,
    partition_by: pc.Option[IntoExprColumn | Iterable[IntoExprColumn]],
    order_by: pc.Option[IntoExprColumn | Iterable[IntoExprColumn]],
    frame_start: pc.Option[FrameBound],
    frame_end: pc.Option[FrameBound],
    frame_mode: FrameMode = "ROWS",
    exclude: pc.Option[WindowExclude] = pc.NONE,
    filter_cond: pc.Option[IntoExprColumn] = pc.NONE,
    *,
    descending: Iterable[bool] | bool = False,
    nulls_last: Iterable[bool] | bool = False,
    ignore_nulls: bool = False,
) -> duckdb.Expression:
    return duckdb.SQLExpression(
        _build_over(
            _handle_filter(
                handle_nulls(expr, ignore_nulls=ignore_nulls),
                filter_cond,
            ),
            partition_by.map(lambda x: try_iter(x).collect()).into(get_partition_by),
            order_by.map(lambda x: try_iter(x).collect()).into(
                get_order_by, descending=descending, nulls_last=nulls_last
            ),
            Kword.frame_clause(
                frame_mode,
                frame_start,
                frame_end,
                has_order_by=order_by.is_some(),
            ),
            exclude.map(Kword.exclude_clause).unwrap_or(""),
        )
    )


def _build_over(
    expr: str, partition_by: str, order_by: str, frame: str, exclude: str
) -> str:
    clauses = pc.Iter((partition_by, order_by, frame, exclude)).filter(bool).join(" ")
    return f"{expr} OVER ({clauses})"


def _handle_filter(expr: str, filter_cond: pc.Option[IntoExprColumn]) -> str:
    return filter_cond.map(lambda c: f"{expr} FILTER (WHERE {c})").unwrap_or(expr)


def get_partition_by(partition_by: pc.Option[pc.Seq[IntoExprColumn]]) -> str:
    return (
        partition_by.map(lambda x: x.iter().map(str).join(", "))
        .map(Kword.partition_by)
        .unwrap_or("")
    )


def handle_nulls(expr: SqlExpr, *, ignore_nulls: bool) -> str:
    match ignore_nulls:
        case True:
            return f"{str(expr).removesuffix(')')} IGNORE NULLS)"
        case False:
            return str(expr)


def get_order_by(
    order_by: pc.Option[pc.Seq[IntoExprColumn]],
    *,
    descending: Iterable[bool] | bool,
    nulls_last: Iterable[bool] | bool,
) -> str:

    def _get_clauses(*, clauses: Iterable[bool] | bool) -> pc.Seq[bool]:
        match clauses:
            case bool() as val:
                return (
                    pc.Iter.once(val)
                    .cycle()
                    .take(order_by.map(lambda x: x.length()).unwrap_or(0))
                    .collect()
                )
            case Iterable() as seq:
                return pc.Seq(seq)

    return (
        order_by.map(
            lambda x: (
                x.iter()
                .zip(_get_clauses(clauses=descending), _get_clauses(clauses=nulls_last))
                .map_star(
                    lambda item, desc, nl: Kword.sort_strat(
                        item, desc=desc, nulls_last=nl
                    )
                )
                .join(", ")
            )
        )
        .map(Kword.order_by)
        .unwrap_or("")
    )
