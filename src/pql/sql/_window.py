from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import duckdb
import pyochain as pc

from ._core import try_iter
from ._raw import Kword

if TYPE_CHECKING:
    from ._expr import SqlExpr


def over_expr(  # noqa: PLR0913
    expr: SqlExpr,
    partition_by: pc.Option[SqlExpr | Iterable[SqlExpr]],
    order_by: pc.Option[SqlExpr | Iterable[SqlExpr]],
    rows_start: pc.Option[int],
    rows_end: pc.Option[int],
    *,
    descending: Iterable[bool] | bool = False,
    nulls_last: Iterable[bool] | bool = False,
    ignore_nulls: bool = False,
) -> duckdb.Expression:
    return _build_over(
        handle_nulls(expr, ignore_nulls=ignore_nulls),
        partition_by.map(lambda x: try_iter(x).collect()).into(get_partition_by),
        order_by.map(lambda x: try_iter(x).collect()).into(
            get_order_by, descending=descending, nulls_last=nulls_last
        ),
        Kword.rows_clause(row_start=rows_start, row_end=rows_end),
    )


def _build_over(
    expr: str, partition_by: str, order_by: str, row_between: str
) -> duckdb.Expression:
    return duckdb.SQLExpression(
        f"{expr} {Kword.OVER} ({partition_by} {order_by} {row_between})"
    )


def get_partition_by(partition_by: pc.Option[pc.Seq[SqlExpr]]) -> str:
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
    order_by: pc.Option[pc.Seq[SqlExpr]],
    *,
    descending: Iterable[bool] | bool,
    nulls_last: Iterable[bool] | bool,
) -> str:

    def _sort_strat(item: SqlExpr, *, desc: bool, nulls_last: bool) -> str:
        return f"{item} {Kword.DESC if desc else Kword.ASC} {Kword.NULLS_LAST if nulls_last else Kword.NULLS_FIRST}"

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
                    lambda item, desc, nl: _sort_strat(item, desc=desc, nulls_last=nl)
                )
                .join(", ")
            )
        )
        .map(Kword.order_by)
        .unwrap_or("")
    )
