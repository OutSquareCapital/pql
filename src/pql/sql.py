"""Utility functions for working with DuckDB SQL expressions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import duckdb
import pyochain as pc

if TYPE_CHECKING:
    from ._types import BoolClause, ByClause, IntoExpr, SqlExpr

col = duckdb.ColumnExpression
lit = duckdb.ConstantExpression
when = duckdb.CaseExpression
all = duckdb.StarExpression
func = duckdb.FunctionExpression
fn_once = duckdb.LambdaExpression
coalesce = duckdb.CoalesceOperator
raw = duckdb.SQLExpression
from_arrow = duckdb.from_arrow
from_query = duckdb.from_query
Relation = duckdb.DuckDBPyRelation


def from_expr(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case str():
            return col(value)
        case _:
            return lit(value)


def from_value(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    from ._expr import Expr

    match value:
        case Expr():
            return value.expr
        case _:
            return lit(value)


def from_iter(
    *values: IntoExpr | Iterable[IntoExpr],
) -> pc.Iter[SqlExpr]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _to_exprs(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
        match value:
            case str():
                return pc.Iter.once(from_expr(value))
            case Iterable():
                return pc.Iter(value).map(from_expr)
            case _:
                return pc.Iter.once(from_expr(value))

    match values:
        case (single,):
            return _to_exprs(single)
        case _:
            return pc.Iter(values).map(_to_exprs).flatten()


def from_args_kwargs(
    *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
) -> pc.Iter[SqlExpr]:
    """Convert positional and keyword arguments to an iterator of DuckDB Expressions."""
    return from_iter(*exprs).chain(
        pc.Dict.from_ref(named_exprs)
        .items()
        .iter()
        .map_star(lambda name, expr: from_expr(expr).alias(name))
    )


@dataclass(slots=True)
class WindowExpr:
    """A window function expression builder."""

    partition_by: ByClause = field(default_factory=pc.Seq.new)  # pyright: ignore[reportUnknownVariableType]
    order_by: ByClause = field(default_factory=pc.Seq.new)  # pyright: ignore[reportUnknownVariableType]
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
            .map(lambda x: x.iter().map(lambda item: str(from_expr(item))).join(", "))
            .map(lambda s: "partition by " + s)
            .unwrap_or("")
        )

    def _get_order_by(self) -> str:
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
                    nl: f"{from_expr(item)} {'desc' if desc else 'asc'} {'nulls last' if nl else 'nulls first'}"
                )
                .join(", ")
            )
            .map(lambda s: "order by " + s)
            .unwrap_or("")
        )

    def _get_rows_clause(self) -> str:
        match (self.rows_start, self.rows_end):
            case (pc.Some(start), pc.Some(end)):
                return f"rows between {-start} preceding and {end} following"
            case (pc.Some(start), pc.NONE):
                return f"rows between {-start} preceding and unbounded following"
            case (pc.NONE, pc.Some(end)):
                return f"rows between unbounded preceding and {end} following"
            case _:
                return ""

    def _get_func(self, expr: SqlExpr) -> str:
        match self.ignore_nulls:
            case True:
                return f"{str(expr).removesuffix(')')} ignore nulls)"
            case False:
                return str(expr)

    def call(self, expr: SqlExpr) -> SqlExpr:
        """Generate the full window function SQL expression."""
        return raw(
            f"{self._get_func(expr)} over ({self._get_partition_by()} {self._get_order_by()} {self._get_rows_clause()})"
        )
