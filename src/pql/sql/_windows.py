from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pyochain as pc

from ._core import SqlExpr, raw

type BoolClause = pc.Option[pc.Seq[bool]] | pc.Option[bool]


class Kword(StrEnum):
    PARTITION_BY = "PARTITION BY"
    ORDER_BY = "ORDER BY"
    DESC = "DESC"
    ASC = "ASC"
    NULLS_LAST = "NULLS LAST"
    NULLS_FIRST = "NULLS FIRST"
    ROWS_BETWEEN = "ROWS BETWEEN"

    @classmethod
    def sort_strat(cls, *, desc: bool, nulls_last: bool) -> str:
        return f"{cls.DESC if desc else cls.ASC} {cls.NULLS_LAST if nulls_last else cls.NULLS_FIRST}"

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


@dataclass(slots=True)
class Over:
    """A window function expression builder."""

    partition_by: pc.Seq[SqlExpr] = field(default_factory=pc.Seq[SqlExpr].new)
    order_by: pc.Seq[SqlExpr] = field(default_factory=pc.Seq[SqlExpr].new)
    rows_start: pc.Option[int] = field(default_factory=lambda: pc.NONE)
    rows_end: pc.Option[int] = field(default_factory=lambda: pc.NONE)
    descending: BoolClause = field(default_factory=lambda: pc.NONE)
    nulls_last: BoolClause = field(default_factory=lambda: pc.NONE)
    ignore_nulls: bool = False

    def _on_scalar(self, *, val: bool) -> pc.Seq[bool]:
        return pc.Iter.once(val).cycle().take(self.order_by.length()).collect()

    def _get_clauses(self, clauses: BoolClause) -> pc.Seq[bool]:
        match clauses:
            case pc.Some(bool(val)):
                return self._on_scalar(val=val)
            case pc.Some(pc.Seq()) as seq:
                return seq.unwrap()
            case _:
                return self._on_scalar(val=False)

    def _get_partition_by(self) -> str:
        return (
            self.partition_by.then_some()
            .map(lambda x: x.iter().map(str).join(", "))
            .map(Kword.partition_by)
            .unwrap_or("")
        )

    def _get_order_by(self) -> str:
        return (
            self.order_by.then_some()
            .map(
                lambda x: (
                    x.iter()
                    .zip(
                        self._get_clauses(self.descending),
                        self._get_clauses(self.nulls_last),
                    )
                    .map_star(
                        lambda item, desc, nl: (
                            f"{item} {Kword.sort_strat(desc=desc, nulls_last=nl)}"
                        )
                    )
                    .join(", ")
                )
            )
            .map(Kword.order_by)
            .unwrap_or("")
        )

    def _get_func(self, expr: SqlExpr) -> str:
        match self.ignore_nulls:
            case True:
                return f"{str(expr).removesuffix(')')} ignore nulls)"
            case False:
                return str(expr)

    def call(self, expr: SqlExpr) -> SqlExpr:
        """Generate the full window function SQL expression."""
        return raw(
            f"{self._get_func(expr)} OVER ({self._get_partition_by()} {self._get_order_by()} {Kword.rows_clause(self.rows_start, self.rows_end)})".strip()
        )
