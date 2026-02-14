"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import pyochain as pc

from . import sql
from .sql import (
    ExprHandler,
    Relation,
    SqlExpr,
    args_into_exprs,
    into_expr,
    iter_into_exprs,
    try_iter,
)

if TYPE_CHECKING:
    import polars as pl

    from ._expr import Expr
    from .sql import FrameInit, IntoExpr, IntoExprColumn
TEMP_NAME = "__pql_temp__"
TEMP_COL = sql.col(TEMP_NAME)


class LazyFrame(ExprHandler[Relation]):
    """LazyFrame providing Polars-like API over DuckDB relations."""

    __slots__ = ()

    def __init__(self, data: FrameInit) -> None:
        self._expr = Relation(data)

    def __repr__(self) -> str:
        return f"LazyFrame\n{self._expr}\n"

    def __from_lf__(self, rel: Relation) -> Self:
        instance = self.__class__.__new__(self.__class__)
        instance._expr = rel
        return instance

    def _select(self, exprs: IntoExprColumn, groups: str = "") -> Self:
        return self.__from_lf__(self._expr.select(*sql.from_cols(exprs), groups=groups))

    def _filter(self, *predicates: IntoExprColumn) -> Self:
        return pc.Iter(predicates).fold(
            self,
            lambda lf, p: sql.from_cols(p).fold(
                lf,
                lambda inner_lf, pred: inner_lf.__from_lf__(
                    inner_lf._expr.filter(pred)
                ),
            ),
        )

    def _agg(self, exprs: IntoExprColumn, group_expr: SqlExpr | str = "") -> Self:
        """Note: duckdb relation.aggregate has type issues, it does accept iterables of expressions but the type signature is wrong."""
        return self.__from_lf__(self._expr.aggregate(sql.from_cols(exprs), group_expr))

    def _iter_slct(self, func: Callable[[str], SqlExpr]) -> Self:
        return self.columns.iter().map(func).into(self._select)

    def _iter_agg(self, func: Callable[[SqlExpr], SqlExpr]) -> Self:
        return (
            self.columns.iter().map(lambda c: func(sql.col(c)).alias(c)).into(self._agg)
        )

    @property
    def relation(self) -> Relation:
        """Get the underlying DuckDB relation."""
        return self._expr

    def lazy(self) -> pl.LazyFrame:
        """Get a Polars LazyFrame."""
        return self._expr.pl(lazy=True)

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame."""
        return self._expr.pl()

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        """Select columns or expressions."""
        return self._select(args_into_exprs(*exprs, **named_exprs))

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        """Add or replace columns."""
        return (
            args_into_exprs(*exprs, **named_exprs).insert(sql.all()).into(self._select)
        )

    def filter(
        self,
        *predicates: IntoExpr | Iterable[IntoExpr],
        **constraints: Any,  # noqa: ANN401
    ) -> Self:
        """Filter rows based on predicates and equality constraints."""
        return (
            args_into_exprs(*predicates)
            .chain(
                pc.Dict.from_ref(constraints)
                .items()
                .iter()
                .map_star(lambda name, value: sql.col(name).eq(into_expr(value)))
            )
            .into(self._filter)
        )

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
    ) -> Self:
        """Sort by columns."""
        sort_exprs = try_iter(by).chain(more_by).into(iter_into_exprs).collect()

        def _args_iter(
            *, arg: bool | Sequence[bool], name: str
        ) -> pc.Result[pc.Iter[bool], ValueError]:
            match arg:
                case bool():
                    return pc.Ok(pc.Iter.once(arg).cycle().take(len(sort_exprs)))
                case Sequence():
                    if len(arg) != sort_exprs.length():
                        msg = (
                            f"the length of `{name}` ({len(arg)}) does not match "
                            f"the length of `by` ({sort_exprs.length()})"
                        )
                        return pc.Err(ValueError(msg))
                    return pc.Ok(pc.Iter(arg))

        def _make_order(col: SqlExpr, desc: bool, nl: bool) -> SqlExpr:  # noqa: FBT001
            match (desc, nl):
                case (True, True):
                    return col.desc().nulls_last()
                case (True, False):
                    return col.desc().nulls_first()
                case (False, True):
                    return col.asc().nulls_last()
                case (False, False):
                    return col.asc()

        return self.__from_lf__(
            self._expr.sort(
                *sort_exprs.iter()
                .zip(
                    _args_iter(arg=descending, name="descending").unwrap(),
                    _args_iter(arg=nulls_last, name="nulls_last").unwrap(),
                )
                .map_star(_make_order)
            )
        )

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self.__from_lf__(self._expr.limit(n))

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self.limit(n)

    def drop(self, *columns: str) -> Self:
        """Drop columns from the frame."""
        return self._select(sql.all(exclude=columns))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_map = pc.Dict(mapping)

        return self._iter_slct(
            lambda c: sql.col(c).alias(rename_map.get_item(c).unwrap_or(c))
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
                self._expr.sql_query(),
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

    def first(self) -> Self:
        """Get the first row."""
        return self.head(1)

    def count(self) -> Self:
        """Return the count of each column."""
        return self._iter_agg(lambda c: c.count())

    def sum(self) -> Self:
        """Aggregate the sum of each column."""
        return self._iter_agg(lambda c: c.sum())

    def mean(self) -> Self:
        """Aggregate the mean of each column."""
        return self._iter_agg(lambda c: c.avg())

    def median(self) -> Self:
        """Aggregate the median of each column."""
        return self._iter_agg(lambda c: c.median())

    def min(self) -> Self:
        """Aggregate the minimum of each column."""
        return self._iter_agg(lambda c: c.min())

    def max(self) -> Self:
        """Aggregate the maximum of each column."""
        return self._iter_agg(lambda c: c.max())

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column."""
        return self._iter_agg(SqlExpr.stddev_samp if ddof == 1 else SqlExpr.stddev_pop)

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column."""
        return self._iter_agg(SqlExpr.var_samp if ddof == 1 else SqlExpr.var_pop)

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self._iter_agg(lambda c: c.is_null().count_if())

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values."""
        return self._iter_slct(
            lambda c: (
                sql.when(sql.col(c).isnan(), into_expr(value, as_col=True))
                .otherwise(sql.col(c))
                .alias(c)
            )
        )

    def _col_or_subset(self, subset: str | Iterable[str] | None) -> pc.Iter[str]:
        match subset:
            case None:
                return self.columns.iter()
            case str():
                return pc.Iter.once(subset)
            case Iterable():
                return pc.Iter(subset)

    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Self:
        """Shift values by n positions."""
        abs_n = abs(n)
        match n:
            case n_val if n_val < 0:

                def _shift_fn(c: str) -> SqlExpr:
                    return sql.col(c).lead(sql.lit(abs_n)).alias(c)

            case _:

                def _shift_fn(c: str) -> SqlExpr:
                    return sql.col(c).lag(sql.lit(abs_n)).alias(c)

        return self._iter_slct(
            lambda c: sql.coalesce(_shift_fn(c).over(), into_expr(fill_value)).alias(c)
        )

    def clone(self) -> Self:
        """Create a copy of the LazyFrame."""
        return self.__from_lf__(self._expr)

    @property
    def columns(self) -> pc.Vec[str]:
        """Get column names."""
        return self._expr.columns

    @property
    def dtypes(self) -> pc.Vec[sql.datatypes.DataType]:
        """Get column data types."""
        return self._expr.dtypes

    @property
    def width(self) -> int:
        """Get number of columns."""
        return len(self._expr.columns)

    @property
    def schema(self) -> pc.Dict[str, sql.datatypes.DataType]:
        """Get the schema as a dictionary."""
        return self.columns.iter().zip(self._expr.dtypes).collect(pc.Dict)

    def collect_schema(self) -> pc.Dict[str, sql.datatypes.DataType]:
        """Collect the schema (same as schema property for lazy)."""
        return self.schema

    def quantile(self, quantile: float) -> Self:
        """Compute quantile for each column."""
        return self._iter_agg(lambda c: c.quantile_cont(sql.lit(quantile)))

    def top_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        """Return top k rows by column(s)."""
        return self.sort(
            by,
            descending=(
                not reverse
                if isinstance(reverse, bool)
                else pc.Iter(reverse).map(lambda x: not x).collect()
            ),
        ).head(k)

    def bottom_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        """Return bottom k rows by column(s)."""
        return self.sort(by, descending=reverse).head(k)

    def cast(
        self, dtypes: Mapping[str, sql.datatypes.DataType] | sql.datatypes.DataType
    ) -> Self:
        """Cast columns to specified dtypes."""
        match dtypes:
            case Mapping():
                dtype_map = pc.Dict(dtypes)
                return self._iter_slct(
                    lambda c: (
                        sql.col(c).cast(dtype_map.get_item(c).unwrap()).alias(c)
                        if c in dtypes
                        else sql.col(c)
                    )
                )
            case _:
                return self._iter_slct(lambda c: sql.col(c).cast(dtypes).alias(c))

    def sink_parquet(self, path: str | Path, *, compression: str = "zstd") -> None:
        """Write to Parquet file."""
        self._expr.write_parquet(str(path), compression=compression)

    def sink_csv(
        self, path: str | Path, *, separator: str = ",", include_header: bool = True
    ) -> None:
        """Write to CSV file."""
        self._expr.write_csv(str(path), sep=separator, header=include_header)

    def sink_ndjson(self, path: str | Path) -> None:
        """Write to newline-delimited JSON file."""
        self._expr.pl(lazy=True).sink_ndjson(path)
