from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Self

import duckdb
import polars as pl
import pyochain as pc
from sqlglot import exp

from ._expr import Expr, exprs_to_nodes, to_node

type FrameInit = duckdb.DuckDBPyRelation | pl.DataFrame | pl.LazyFrame | None


def _data_to_rel(data: FrameInit) -> duckdb.DuckDBPyRelation:
    match data:
        case duckdb.DuckDBPyRelation():
            return data
        case pl.DataFrame() | pl.LazyFrame():
            return duckdb.from_arrow(data.lazy().collect())
        case None:
            return duckdb.from_arrow(pl.DataFrame({"_": [None]}))


class LazyFrame:
    """LazyFrame providing Polars-like API for SQL generation."""

    __ast__: exp.Select
    rel: duckdb.DuckDBPyRelation
    __slots__ = ("__ast__", "rel")

    def __init__(self, data: FrameInit = None) -> None:
        self.__ast__ = exp.select("*").from_(exp.Table(this=exp.Identifier(this="_")))
        self.rel = _data_to_rel(data)

    def __repr__(self) -> str:
        return f"LazyFrame(\n{self.sql()}\n)"

    def _new(self, ast: exp.Select) -> Self:
        instance = self.__class__(self.rel)
        instance.__ast__ = ast
        return instance

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame."""
        return self.rel.query("_", self.sql(pretty=False)).pl()

    def select(self, *exprs: Expr | str) -> Self:
        """Select columns or expressions."""
        nodes = exprs_to_nodes(exprs)
        return self._new(self.__ast__.copy().select(*nodes, append=False, copy=False))

    def with_columns(self, *exprs: Expr) -> Self:
        """Add or replace columns."""
        nodes = exprs_to_nodes(exprs)
        return self._new(
            self.__ast__.copy().select("*", *nodes, append=False, copy=False)
        )

    def filter(self, *predicates: Expr) -> Self:
        """Filter rows based on predicates."""
        new_ast = self.__ast__.copy()
        for p in predicates:
            new_ast = new_ast.where(p.__node__, copy=False)
        return self._new(new_ast)

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
        desc_iter = (
            pc.Iter.once(descending).cycle().take(len(by))
            if isinstance(descending, bool)
            else pc.Iter(descending)
        )
        nulls_iter = (
            pc.Iter.once(nulls_last).cycle().take(len(by))
            if isinstance(nulls_last, bool)
            else pc.Iter(nulls_last)
        )

        order_terms = (
            pc.Iter(by)
            .zip(desc_iter)
            .zip(nulls_iter)
            .map(lambda args: (args[0][0], args[0][1], args[1]))
            .map_star(
                lambda col, desc, nl: exp.Ordered(
                    this=to_node(col), desc=desc, nulls_first=None if nl else True
                )
            )
        )
        return self._new(self.__ast__.copy().order_by(*order_terms, copy=False))

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self._new(self.__ast__.copy().limit(n, copy=False))

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self._new(self.__ast__.copy().limit(n, copy=False))

    def tail(self, n: int = 5) -> Self:
        """Get the last n rows."""
        return self._new(self.__ast__.copy().limit(n, copy=False))

    def distinct(self) -> Self:
        """Get distinct rows."""
        return self._new(self.__ast__.copy().distinct(copy=False))

    def unique(self, subset: str | Iterable[str] | None = None) -> Self:
        """Get unique rows based on subset of columns."""
        if subset is None:
            return self.distinct()
        cols = pc.Iter.once(subset) if isinstance(subset, str) else pc.Iter(subset)
        new_ast = self.__ast__.copy()
        new_ast.set(
            "distinct",
            exp.Distinct(on=exp.Tuple(expressions=exprs_to_nodes(cols).collect())),
        )
        return self._new(new_ast)

    def drop(self, *columns: str) -> Self:
        """Drop columns from the frame."""
        exclude_star = exp.Star(except_=pc.Iter(columns).map(exp.column))
        return self._new(
            self.__ast__.copy().select(exclude_star, append=False, copy=False)
        )

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_exprs = (
            pc.Dict(mapping)
            .items()
            .iter()
            .map_star(lambda old, new: exp.alias_(exp.column(old), new))
        )
        return self._new(
            self.__ast__.copy().select(*rename_exprs, append=True, copy=False)
        )

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
        if how == "cross":
            return self.__class__(self.rel.cross(other.rel))
        lhs = self.rel.set_alias("lhs")
        rhs = other.rel.set_alias("rhs")
        rel = lhs.join(
            rhs, condition=_build_join_condition(on, left_on, right_on), how=how
        )

        return self.__class__(
            rel
            if how in {"semi", "anti"}
            else _apply_join_suffix(rel, lhs.columns, rhs.columns, on, suffix)
        )

    def sql(self, *, pretty: bool = True) -> str:
        """Generate SQL string."""
        ast = self.__ast__.copy()
        if not ast.expressions:
            ast = ast.select("*", append=False, copy=False)
        return ast.sql(dialect="duckdb", pretty=pretty)


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
            return " AND ".join(
                pc.Iter(left_on)
                .zip(right_on)
                .map_star(lambda lk, rk: f"lhs.{lk} = rhs.{rk}")
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
        by_nodes = exprs_to_nodes(self._by).collect()
        agg_nodes = exprs_to_nodes(exprs)
        new_ast = (
            self._lf.__ast__.copy()
            .select(*by_nodes, *agg_nodes, append=False, copy=False)
            .group_by(*by_nodes, copy=False)
        )
        return self._lf._new(new_ast)  # type: ignore[private-access]
