from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import duckdb
import polars as pl
import pyochain as pc
from polars._typing import FrameInitTypes

if TYPE_CHECKING:
    from ._expr import Expr

type PyLiteral = str | int | float | bool | None
type FrameInit = (
    duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes
)
type IntoExpr = PyLiteral | Expr | duckdb.Expression


def data_to_rel(data: FrameInit) -> duckdb.DuckDBPyRelation:
    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case pl.DataFrame():
            return duckdb.from_arrow(data)
        case pl.LazyFrame():
            from sqlglot import exp

            _ = data
            return duckdb.sql(exp.select(exp.Star()).from_("_").sql(dialect="duckdb"))

        case None:
            return duckdb.from_arrow(pl.DataFrame({"_": []}))
        case _:
            return duckdb.from_arrow(pl.DataFrame(data))


def to_expr(value: IntoExpr) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case duckdb.Expression():
            return value
        case str():
            return duckdb.ColumnExpression(value)
        case _:
            return duckdb.ConstantExpression(value)


def to_value(value: IntoExpr) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case duckdb.Expression():
            return value
        case _:
            return duckdb.ConstantExpression(value)


def iter_to_exprs(
    *values: IntoExpr | Iterable[IntoExpr],
) -> pc.Iter[duckdb.Expression]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _to_exprs(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[duckdb.Expression]:
        match value:
            case str():
                return pc.Iter.once(to_expr(value))
            case Iterable():
                return pc.Iter(value).map(to_expr)
            case _:
                return pc.Iter.once(to_expr(value))

    match values:
        case (single,):
            return _to_exprs(single)
        case _:
            return pc.Iter(values).map(_to_exprs).flatten()


type ByClause = (
    pc.Seq[str] | pc.Seq[duckdb.Expression] | pc.Seq[str | duckdb.Expression]
)
type BoolClause = pc.Option[pc.Seq[bool]] | pc.Option[bool]


@dataclass(slots=True)
class WindowSpec:
    partition_by: ByClause = field(default_factory=pc.Seq[str | duckdb.Expression].new)
    order_by: ByClause = field(default_factory=pc.Seq[str | duckdb.Expression].new)
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

    def get_partition_by(self) -> str:
        return (
            self.partition_by.then_some()
            .map(lambda x: x.iter().map(lambda item: str(to_expr(item))).join(", "))
            .map(lambda s: "partition by " + s)
            .unwrap_or("")
        )

    def get_order_by(self) -> str:
        return (
            self.order_by.then_some()
            .map(
                lambda x: x.iter()
                .zip(
                    self._get_clauses(self.descending),
                    self._get_clauses(self.nulls_last),
                )
                .map_star(
                    lambda item,
                    desc,
                    nl: f"{to_expr(item)} {'desc' if desc else 'asc'} {'nulls last' if nl else 'nulls first'}"
                )
                .join(", ")
            )
            .map(lambda s: "order by " + s)
            .unwrap_or("")
        )

    def get_rows_clause(self) -> str:
        match (self.rows_start, self.rows_end):
            case (pc.Some(start), pc.Some(end)):
                return f"rows between {-start} preceding and {end} following"
            case (pc.Some(start), pc.NONE):
                return f"rows between {-start} preceding and unbounded following"
            case (pc.NONE, pc.Some(end)):
                return f"rows between unbounded preceding and {end} following"
            case _:
                return ""

    def get_func(self, expr: duckdb.Expression) -> str:
        match self.ignore_nulls:
            case True:
                return f"{str(expr).removesuffix(')')} ignore nulls)"
            case False:
                return str(expr)

    def into_expr(self, expr: duckdb.Expression) -> duckdb.Expression:
        return duckdb.SQLExpression(
            f"{self.get_func(expr)} over ({self.get_partition_by()} {self.get_order_by()} {self.get_rows_clause()})"
        )
