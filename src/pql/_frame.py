from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Self

import pyochain as pc
from sqlglot import exp

from ._expr import Expr, exprs_to_nodes, to_node, val_to_iter


class LazyFrame:
    """LazyFrame providing Polars-like API for SQL generation."""

    __slots__ = ("__ast__",)

    def __init__(self, ast: exp.Select) -> None:
        self.__ast__ = ast

    def __repr__(self) -> str:
        return f"LazyFrame(\n{self.sql()}\n)"

    @classmethod
    def scan_table(cls, name: str) -> Self:
        """Create a LazyFrame from a table name."""
        return cls(exp.select("*").from_(name, dialect="duckdb"))

    # ==================== Transformations ====================

    def select(self, *exprs: Expr | str) -> Self:
        """Select columns or expressions."""
        nodes = exprs_to_nodes(exprs)
        return self.__class__(
            self.__ast__.copy().select(*nodes, append=False, copy=False)
        )

    def with_columns(self, *exprs: Expr) -> Self:
        """Add or replace columns."""
        nodes = exprs_to_nodes(exprs)
        return self.__class__(
            self.__ast__.copy().select("*", *nodes, append=False, copy=False)
        )

    def filter(self, *predicates: Expr) -> Self:
        """Filter rows based on predicates."""
        new_ast = self.__ast__.copy()
        for p in predicates:
            new_ast = new_ast.where(p.__node__, copy=False)
        return self.__class__(new_ast)

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
        desc_flags = (
            [descending] * len(by) if isinstance(descending, bool) else descending
        )
        nulls_flags = (
            [nulls_last] * len(by) if isinstance(nulls_last, bool) else nulls_last
        )

        order_terms = (
            pc.Iter(by)
            .zip(desc_flags)
            .zip(nulls_flags)
            .map(lambda args: (args[0][0], args[0][1], args[1]))
            .map_star(
                lambda col, desc, nl: exp.Ordered(
                    this=to_node(col), desc=desc, nulls_first=None if nl else True
                )
            )
        )
        return self.__class__(self.__ast__.copy().order_by(*order_terms, copy=False))

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self.__class__(self.__ast__.copy().limit(n, copy=False))

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self.limit(n)

    def tail(self, n: int = 5) -> Self:
        """Get the last n rows (requires ORDER BY to be meaningful)."""
        return self.limit(n)

    def distinct(self) -> Self:
        """Get distinct rows."""
        return self.__class__(self.__ast__.copy().distinct(copy=False))

    def unique(self, subset: str | Iterable[str] | None = None) -> Self:
        """Get unique rows based on subset of columns."""
        if subset is None:
            return self.distinct()
        cols = [subset] if isinstance(subset, str) else subset
        new_ast = self.__ast__.copy()
        new_ast.set(
            "distinct",
            exp.Distinct(on=exp.Tuple(expressions=exprs_to_nodes(cols).collect())),
        )
        return self.__class__(new_ast)

    def drop(self, *columns: str) -> Self:
        """Drop columns from the frame."""
        exclude_star = exp.Star(except_=pc.Set(columns).iter().map(exp.column))
        return self.__class__(
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
        return self.__class__(
            self.__ast__.copy().select(*rename_exprs, append=True, copy=False)
        )

    def join(
        self,
        other: LazyFrame,
        on: str | Expr | Iterable[str] | None = None,
        *,
        left_on: str | Expr | Iterable[str] | None = None,
        right_on: str | Expr | Iterable[str] | None = None,
        how: Literal[
            "inner", "left", "right", "outer", "cross", "semi", "anti"
        ] = "inner",
    ) -> Self:
        """Join with another LazyFrame."""
        subquery = exp.Subquery(this=other.__ast__, alias="_r")
        if on is not None:
            on_list = val_to_iter(on)
            new_ast = self.__ast__.copy().join(
                subquery,
                using=exprs_to_nodes(on_list).collect(tuple),
                join_type=how,
                copy=False,
            )
        elif left_on is not None and right_on is not None:
            on_expr = (
                val_to_iter(left_on)
                .zip(val_to_iter(right_on))
                .map_star(
                    lambda lc, rc: exp.EQ(this=to_node(lc), expression=to_node(rc))
                )
                .reduce(lambda acc, eq: exp.And(this=acc, expression=eq))  # type: ignore[arg-type]
            )
            new_ast = self.__ast__.copy().join(
                subquery,
                on=on_expr,
                join_type=how,
                copy=False,
            )
        else:
            new_ast = self.__ast__.copy().join(subquery, join_type=how, copy=False)

        return self.__class__(new_ast)

    # ==================== SQL Output ====================

    def sql(self, *, pretty: bool = True) -> str:
        """Generate SQL string."""
        return self.__ast__.sql(dialect="duckdb", pretty=pretty)

    def explain(self) -> str:
        """Generate EXPLAIN SQL."""
        return f"EXPLAIN {self.sql(pretty=False)}"


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
        return LazyFrame(new_ast)
