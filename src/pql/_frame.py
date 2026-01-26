"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Self

import duckdb
import polars as pl
import pyochain as pc
from polars._typing import FrameInitTypes  # pyright: ignore[reportUnknownVariableType]

from ._expr import Expr, to_expr

type FrameInit = (
    duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None | FrameInitTypes
)


def _data_to_rel(data: FrameInit) -> duckdb.DuckDBPyRelation:  # pyright: ignore[reportUnknownParameterType]
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
            return duckdb.from_arrow(pl.DataFrame({"_": [None]}))
        case _:  # pyright: ignore[reportUnknownVariableType]
            return duckdb.from_arrow(pl.DataFrame(data))


class LazyFrame:
    """LazyFrame providing Polars-like API over DuckDB relations."""

    _rel: duckdb.DuckDBPyRelation
    __slots__ = ("_rel",)

    def __init__(self, data: FrameInit = None) -> None:  # pyright: ignore[reportUnknownParameterType]
        self._rel = _data_to_rel(data)

    def __repr__(self) -> str:
        return f"LazyFrame\n{self._rel}\n"

    def __from_lf__(self, rel: duckdb.DuckDBPyRelation) -> Self:
        instance = self.__class__.__new__(self.__class__)
        instance._rel = rel
        return instance

    @property
    def relation(self) -> duckdb.DuckDBPyRelation:
        """Get the underlying DuckDB relation."""
        return self._rel

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame."""
        return self._rel.pl()

    def select(self, *exprs: Expr | str) -> Self:
        """Select columns or expressions."""
        return self.__from_lf__(self._rel.select(*pc.Iter(exprs).map(to_expr)))

    def with_columns(self, *exprs: Expr) -> Self:
        """Add or replace columns."""
        return self.__from_lf__(
            self._rel.select(
                duckdb.StarExpression(),
                *pc.Iter(exprs).map(lambda e: e.expr),
            )
        )

    def filter(self, *predicates: Expr) -> Self:
        """Filter rows based on predicates."""
        return pc.Iter(predicates).fold(
            self, lambda lf, p: lf.__from_lf__(lf._rel.filter(p.expr))
        )

    def group_by(self, *by: str | Expr) -> GroupBy:
        """Group by columns."""
        return GroupBy(self, by)

    def sort(
        self,
        *by: str | Expr,
        descending: bool | Iterable[bool] = False,
        nulls_last: bool | Iterable[bool] = False,
    ) -> Self:
        """Sort by columns."""

        def _args_iter(*, arg: bool | Iterable[bool]) -> pc.Iter[bool]:
            return (
                pc.Iter.once(arg).cycle().take(len(by))
                if isinstance(arg, bool)
                else pc.Iter(arg)
            )

        def _make_order(col: str | Expr, desc: bool, nl: bool) -> duckdb.Expression:  # noqa: FBT001
            expr = to_expr(col)
            return (
                expr.desc().nulls_last()
                if desc and nl
                else (
                    expr.desc().nulls_first()
                    if desc
                    else (expr.asc().nulls_last() if nl else expr.asc())
                )
            )

        order_exprs = (
            pc.Iter(by)
            .zip(_args_iter(arg=descending), _args_iter(arg=nulls_last))
            .map_star(_make_order)
        )
        return self.__from_lf__(self._rel.sort(*order_exprs))

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self.__from_lf__(self._rel.limit(n))

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self.limit(n)

    def tail(self, n: int = 5) -> Self:
        """Get the last n rows."""
        return self.__from_lf__(
            self._rel.limit(n, offset=self._rel.count("*").fetchone()[0] - n)
        )

    def distinct(self) -> Self:
        """Get distinct rows."""
        return self.__from_lf__(self._rel.distinct())

    def unique(self, subset: str | Iterable[str] | None = None) -> Self:
        """Get unique rows based on subset of columns."""
        if subset is None:
            return self.distinct()
        cols = pc.Iter.once(subset) if isinstance(subset, str) else pc.Iter(subset)
        return self.__from_lf__(self._rel.distinct(*cols.map(duckdb.ColumnExpression)))

    def drop(self, *columns: str) -> Self:
        """Drop columns from the frame."""
        return self.__from_lf__(
            self._rel.select(duckdb.StarExpression(exclude=columns))
        )

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_map = pc.Dict(mapping)

        exprs = pc.Iter(self._rel.columns).map(
            lambda c: duckdb.ColumnExpression(c).alias(
                rename_map.get_item(c).unwrap_or(c)
            )
        )
        return self.__from_lf__(self._rel.select(*exprs))

    def join(  # noqa: PLR0913
        self,
        other: LazyFrame,
        on: str | Expr | Iterable[str] | None = None,
        *,
        left_on: str | Expr | Iterable[str] | None = None,
        right_on: str | Expr | Iterable[str] | None = None,
        how: Literal[
            "inner", "left", "right", "outer", "cross", "semi", "anti"
        ] = "inner",
        suffix: str = "_right",
    ) -> Self:
        """Join with another LazyFrame."""
        match how:
            case "cross":
                return self.__from_lf__(self._rel.cross(other._rel))
            case _:
                lhs = self._rel.set_alias("lhs")
                rhs = other._rel.set_alias("rhs")
                rel = lhs.join(
                    rhs, condition=_build_join_condition(on, left_on, right_on), how=how
                )

                return self.__from_lf__(
                    rel
                    if how in {"semi", "anti"}
                    else _apply_join_suffix(rel, lhs.columns, rhs.columns, on, suffix)
                )

    def explain(self) -> str:
        """Generate SQL string."""
        kwords = (
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "INNER JOIN",
            "FULL JOIN",
            "ON",
            "AND",
            "OR",
            "LIMIT",
            "OFFSET",
        )
        qry = (
            pc.Iter(kwords)
            .fold(
                self._rel.sql_query(),
                lambda acc, kw: acc.replace(f" {kw} ", f"\n{kw} "),
            )
            .split("\n")
        )

        return (
            pc.Iter(qry)
            .map(lambda line: line.rstrip())
            .filter(lambda line: bool(line.strip()))
            .join("\n")
        )


def _build_join_condition(
    on: str | Expr | Iterable[str] | None,
    left_on: str | Expr | Iterable[str] | None,
    right_on: str | Expr | Iterable[str] | None,
) -> str:
    """Build DuckDB join condition string."""
    match (on, left_on, right_on):
        case (str(), _, _):
            return on
        case (None, str(), str()):
            return f"lhs.{left_on} = rhs.{right_on}"
        case (None, Iterable(), Iterable()):
            return (
                pc.Iter(left_on)
                .zip(right_on)
                .map_star(lambda lk, rk: f"lhs.{lk} = rhs.{rk}")
                .join(" AND ")
            )
        case _:
            msg = "Join requires 'on' or both 'left_on' and 'right_on'"
            raise ValueError(msg)


def _apply_join_suffix(
    rel: duckdb.DuckDBPyRelation,
    lhs_cols: Iterable[str],
    rhs_cols: Iterable[str],
    on: str | Expr | Iterable[str] | None,
    suffix: str,
) -> duckdb.DuckDBPyRelation:
    """Apply suffix to duplicate columns from right side."""
    join_keys = pc.Set((on,) if isinstance(on, str) else ())
    lhs_set = pc.Set(lhs_cols)

    def _col_expr(name: str) -> duckdb.Expression:
        col = duckdb.ColumnExpression(f"rhs.{name}")
        return (
            col.alias(f"{name}{suffix}")
            if name in lhs_set and name not in join_keys
            else col
        )

    return rel.select(
        *pc.Iter(lhs_cols).map(lambda c: duckdb.ColumnExpression(f"lhs.{c}")),
        *pc.Iter(rhs_cols).filter(lambda c: c not in join_keys).map(_col_expr),
    )


@dataclass(slots=True)
class GroupBy:
    """GroupBy object for aggregation operations."""

    _lf: LazyFrame
    _by: tuple[str | Expr, ...]

    def agg(self, *exprs: Expr) -> LazyFrame:
        """Aggregate the grouped data."""
        by_strs = (
            pc.Iter(self._by)
            .map(lambda b: b if isinstance(b, str) else str(b.expr))
            .join(", ")
        )
        all_exprs = pc.Seq(
            (
                *pc.Iter(self._by).map(to_expr),
                *pc.Iter(exprs).map(lambda e: e.expr),
            )
        )
        return self._lf.__from_lf__(self._lf.relation.aggregate(all_exprs, by_strs))
