from enum import StrEnum
from typing import Any

import duckdb
import pyochain as pc

from ._core import ExprHandler


def over[T: ExprHandler[Any]](  # noqa: PLR0913
    expr: T,
    partition_by: pc.Seq[T] | None,
    order_by: pc.Seq[T] | None,
    rows_start: int | None = None,
    rows_end: int | None = None,
    *,
    descending: pc.Seq[bool] | bool = False,
    nulls_last: pc.Seq[bool] | bool = False,
    ignore_nulls: bool = False,
) -> duckdb.Expression:
    return build_over(
        handle_nulls(expr, ignore_nulls=ignore_nulls),
        get_partition_by(partition_by or pc.Seq.new()),  # pyright: ignore[reportArgumentType]
        get_order_by(
            order_by or pc.Seq.new(),  # pyright: ignore[reportArgumentType]
            descending=descending,
            nulls_last=nulls_last,
        ),
        Kword.rows_clause(pc.Option(rows_start), pc.Option(rows_end)),
    )


def build_over(
    expr: str, partition_by: str, order_by: str, row_between: str
) -> duckdb.Expression:
    return duckdb.SQLExpression(
        f"{expr} {Kword.OVER} ({partition_by} {order_by} {row_between})"
    )


class Kword(StrEnum):
    PARTITION_BY = "PARTITION BY"
    ORDER_BY = "ORDER BY"
    DESC = "DESC"
    ASC = "ASC"
    NULLS_LAST = "NULLS LAST"
    NULLS_FIRST = "NULLS FIRST"
    ROWS_BETWEEN = "ROWS BETWEEN"
    OVER = "OVER"

    @classmethod
    def sort_strat[T](
        cls, item: ExprHandler[T], *, desc: bool, nulls_last: bool
    ) -> str:
        return f"{item} {cls.DESC if desc else cls.ASC} {cls.NULLS_LAST if nulls_last else cls.NULLS_FIRST}"

    @classmethod
    def rows_clause(cls, row_start: pc.Option[int], row_end: pc.Option[int]) -> str:
        match (row_start, row_end):
            case (pc.Some(start), pc.Some(end)):
                return f"{cls.ROWS_BETWEEN} {-start} PRECEDING AND {end} FOLLOWING"
            case (pc.Some(start), pc.NONE):
                return f"{cls.ROWS_BETWEEN} {-start} PRECEDING AND UNBOUNDED FOLLOWING"
            case (pc.NONE, pc.Some(end)):
                return f"{cls.ROWS_BETWEEN} UNBOUNDED PRECEDING AND {end} FOLLOWING"
            case _:
                return ""

    @classmethod
    def partition_by(cls, by: str) -> str:
        return f"{cls.PARTITION_BY} {by}"

    @classmethod
    def order_by(cls, by: str) -> str:
        return f"{cls.ORDER_BY} {by}"


def get_partition_by[T](partition_by: pc.Seq[ExprHandler[T]]) -> str:
    return (
        partition_by.then_some()
        .map(lambda x: x.iter().map(str).join(", "))
        .map(Kword.partition_by)
        .unwrap_or("")
    )


def handle_nulls[T](expr: ExprHandler[T], *, ignore_nulls: bool) -> str:
    match ignore_nulls:
        case True:
            return f"{str(expr).removesuffix(')')} ignore nulls)"
        case False:
            return str(expr)


def get_order_by[T](
    order_by: pc.Seq[ExprHandler[T]],
    *,
    descending: pc.Seq[bool] | bool,
    nulls_last: pc.Seq[bool] | bool,
) -> str:
    def _get_clauses(*, clauses: pc.Seq[bool] | bool) -> pc.Seq[bool]:
        match clauses:
            case bool() as val:
                return pc.Iter.once(val).cycle().take(order_by.length()).collect()
            case pc.Seq() as seq:
                return seq

    return (
        order_by.then_some()
        .map(
            lambda x: (
                x.iter()
                .zip(_get_clauses(clauses=descending), _get_clauses(clauses=nulls_last))
                .map_star(
                    lambda item, desc, nl: Kword.sort_strat(
                        item=item, desc=desc, nulls_last=nl
                    )
                )
                .join(", ")
            )
        )
        .map(Kword.order_by)
        .unwrap_or("")
    )
