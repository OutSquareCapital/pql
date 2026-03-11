from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import duckdb
import pyochain as pc

from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from duckdb import Expression

    from .typing import FrameMode, IntoExprColumn, WindowExclude

type FrameBound = int | str
type NullOrder = Literal["NULLS FIRST", "NULLS LAST"]
type SortOrder = Literal["ASC", "DESC"]
type OptExpr = pc.Option[IntoExprColumn | Iterable[IntoExprColumn]]
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
    def null_order(cls, *, last: bool) -> NullOrder:
        match last:
            case True:
                return "NULLS LAST"
            case False:
                return "NULLS FIRST"

    @classmethod
    def sort_order(cls, *, desc: bool) -> SortOrder:
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


@dataclass(slots=True)
class OverBuilder:
    expr: str

    def handle_nulls(self, *, ignore_nulls: bool) -> Self:
        match ignore_nulls:
            case True:
                return self.__class__(f"{self.expr.removesuffix(')')} IGNORE NULLS)")
            case False:
                return self

    def handle_distinct(self, *, distinct: bool) -> Self:
        match distinct:
            case True:
                return self.__class__(self.expr.replace("(", "(DISTINCT ", 1))
            case False:
                return self

    def handle_fn_order_by(
        self,
        fn_order_by: OptExpr,
        *,
        fn_descending: TryIter[bool],
        fn_nulls_last: TryIter[bool],
    ) -> Self:
        def _build(cols: pc.Seq[IntoExprColumn]) -> str:
            n = cols.length()

            def _expand(*, clauses: TryIter[bool]) -> pc.Seq[bool]:
                match clauses:
                    case Iterable() as seq:
                        return pc.Seq(seq)
                    case _ as val:
                        return try_iter(val).cycle().take(n).collect()

            items = (
                cols.iter()
                .zip(_expand(clauses=fn_descending), _expand(clauses=fn_nulls_last))
                .map_star(
                    lambda item, desc, nl: Kword.sort_strat(
                        item, desc=desc, nulls_last=nl
                    )
                )
                .join(", ")
            )
            return f"{self.expr.removesuffix(')')} ORDER BY {items})"

        return (
            fn_order_by.map(lambda x: try_iter(x).collect().into(_build))
            .map(self.__class__)
            .unwrap_or(self)
        )

    def handle_filter(self, filter_cond: pc.Option[IntoExprColumn]) -> Self:
        return (
            filter_cond.map(lambda c: f"{self.expr} FILTER (WHERE {c})")
            .map(self.__class__)
            .unwrap_or(self)
        )

    def join_clauses(
        self, partition_by: str, order_by: str, frame: str, exclude: str
    ) -> Self:
        clauses = (
            pc.Iter((partition_by, order_by, frame, exclude)).filter(bool).join(" ")
        )
        return self.__class__(f"{self.expr} OVER ({clauses})")

    def build(self) -> Expression:
        return duckdb.SQLExpression(self.expr)

    def build_fn(
        self,
        *,
        order_by: OptExpr,
        ignore_nulls: bool = False,
        fn_descending: TryIter[bool] = False,
        fn_nulls_last: TryIter[bool] = False,
    ) -> Expression:
        return (
            self.handle_fn_order_by(
                order_by, fn_descending=fn_descending, fn_nulls_last=fn_nulls_last
            )
            .handle_nulls(ignore_nulls=ignore_nulls)
            .build()
        )


def get_partition_by(partition_by: pc.Option[pc.Seq[IntoExprColumn]]) -> str:
    return (
        partition_by.map(lambda x: x.iter().map(str).join(", "))
        .map(Kword.partition_by)
        .unwrap_or("")
    )


def get_order_by(
    order_by: pc.Option[pc.Seq[IntoExprColumn]],
    *,
    descending: TryIter[bool],
    nulls_last: TryIter[bool],
) -> str:

    def _get_clauses(*, clauses: TryIter[bool]) -> pc.Seq[bool]:
        match clauses:
            case Iterable() as seq:
                return pc.Seq(seq)
            case _ as val:
                return (
                    try_iter(val)
                    .cycle()
                    .take(order_by.map(lambda x: x.length()).unwrap_or(0))
                    .collect()
                )

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
