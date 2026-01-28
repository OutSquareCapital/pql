"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, Literal, Self

import polars as pl
import pyochain as pc

from . import sql

if TYPE_CHECKING:
    from . import datatypes
    from ._expr import Expr
    from ._types import FrameInit, IntoExpr


class LazyFrame:
    """LazyFrame providing Polars-like API over DuckDB relations."""

    _rel: sql.Relation
    __slots__ = ("_rel",)

    def __init__(self, data: FrameInit = None) -> None:
        match data:
            case sql.Relation():
                self._rel = data
            case pl.DataFrame():
                self._rel = sql.from_arrow(data)
            case pl.LazyFrame():
                from sqlglot import exp

                _ = data
                self._rel = sql.from_query(
                    exp.select(exp.Star()).from_("_").sql(dialect="duckdb")
                )

            case None:
                self._rel = sql.from_arrow(pl.DataFrame({"_": []}))
            case _:
                self._rel = sql.from_arrow(pl.DataFrame(data))

    def __repr__(self) -> str:
        return f"LazyFrame\n{self._rel}\n"

    def __from_lf__(self, rel: sql.Relation) -> Self:
        instance = self.__class__.__new__(self.__class__)
        instance._rel = rel
        return instance

    @property
    def relation(self) -> sql.Relation:
        """Get the underlying DuckDB relation."""
        return self._rel

    def lazy(self) -> pl.LazyFrame:
        """Get a Polars LazyFrame."""
        return self._rel.pl(lazy=True)

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame."""
        return self._rel.pl()

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        """Select columns or expressions."""
        return self.__from_lf__(
            self._rel.select(*sql.from_args_kwargs(*exprs, **named_exprs))
        )

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        """Add or replace columns."""
        return self.__from_lf__(
            self._rel.select(
                *sql.from_args_kwargs(*exprs, **named_exprs).insert(sql.all())
            )
        )

    def filter(self, *predicates: Expr) -> Self:
        """Filter rows based on predicates."""
        return pc.Iter(predicates).fold(
            self, lambda lf, p: lf.__from_lf__(lf._rel.filter(p.expr))
        )

    def sort(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        descending: bool | Iterable[bool] = False,
        nulls_last: bool | Iterable[bool] = False,
    ) -> Self:
        """Sort by columns."""

        def _args_iter(*, arg: bool | Iterable[bool]) -> pc.Iter[bool]:
            match arg:
                case bool():
                    return pc.Iter.once(arg).cycle().take(len(by))
                case Iterable():
                    return pc.Iter(arg)

        def _make_order(col: sql.SqlExpr, desc: bool, nl: bool) -> sql.SqlExpr:  # noqa: FBT001
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
            self._rel.sort(
                *sql.from_iter(*by)
                .zip(_args_iter(arg=descending), _args_iter(arg=nulls_last))
                .map_star(_make_order)
            )
        )

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self.__from_lf__(self._rel.limit(n))

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self.limit(n)

    def unique(self, subset: str | Iterable[str] | None = None) -> Self:
        """Get unique rows based on subset of columns."""

        def _on_subset(cols: pc.Iter[str]) -> Self:
            return self.__from_lf__(
                self._rel.select(
                    sql.all(),
                    sql.WindowExpr(partition_by=cols.collect())
                    .call(sql.fns.row_number())
                    .alias("__rn__"),
                )
                .filter(sql.col("__rn__").__eq__(sql.lit(1)))
                .project("* EXCLUDE (__rn__)")
            )

        match subset:
            case None:
                return self.__from_lf__(self._rel.distinct())
            case str():
                return pc.Iter.once(subset).into(_on_subset)
            case Iterable():
                return pc.Iter(subset).into(_on_subset)

    def drop(self, *columns: str) -> Self:
        """Drop columns from the frame."""
        return self.__from_lf__(self._rel.select(sql.all(exclude=columns)))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_map = pc.Dict(mapping)

        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: sql.col(c).alias(rename_map.get_item(c).unwrap_or(c))
                )
            )
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

    def pipe[**P, T](
        self,
        function: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Apply a function to the LazyFrame."""
        return function(self, *args, **kwargs)

    def reverse(self) -> Self:
        """Reverse the LazyFrame rows."""
        return self.__from_lf__(
            self._rel.row_number("over () as __rn__", "*").project("* EXCLUDE (__rn__)")
        )

    def first(self) -> Self:
        """Get the first row."""
        return self.head(1)

    def count(self) -> Self:
        """Return the count of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: sql.fns.count(sql.col(c)).alias(c)),
            )
        )

    def sum(self) -> Self:
        """Aggregate the sum of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: sql.fns.sum_func(sql.col(c)).alias(c))
            )
        )

    def mean(self) -> Self:
        """Aggregate the mean of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: sql.fns.avg(sql.col(c)).alias(c)),
            )
        )

    def median(self) -> Self:
        """Aggregate the median of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: sql.fns.median(sql.col(c)).alias(c))
            )
        )

    def min(self) -> Self:
        """Aggregate the minimum of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: sql.fns.min(sql.col(c)).alias(c))
            )
        )

    def max(self) -> Self:
        """Aggregate the maximum of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: sql.fns.max(sql.col(c)).alias(c))
            )
        )

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column."""
        std_func = sql.fns.stddev_samp if ddof == 1 else sql.fns.stddev_pop
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: std_func(sql.col(c)).alias(c))
            )
        )

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column."""
        var_func = sql.fns.var_samp if ddof == 1 else sql.fns.var_pop
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(lambda c: var_func(sql.col(c)).alias(c))
            )
        )

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: sql.fns.count_if(
                        sql.col(c).isnull(),
                    ).alias(c)
                )
            )
        )

    def fill_null(
        self,
        value: IntoExpr = None,
        *,
        strategy: Literal["forward", "backward"] | None = None,
    ) -> Self:
        """Fill null values."""
        match (value, strategy):
            case (None, "forward"):
                return self._fill_strategy("LAST_VALUE")
            case (None, "backward"):
                return self._fill_strategy("FIRST_VALUE")
            case (val, None) if val is not None:
                return self.__from_lf__(
                    self._rel.select(
                        *self.columns.iter().map(
                            lambda c: sql.coalesce(
                                sql.col(c), sql.from_expr(val)
                            ).alias(c)
                        )
                    )
                )
            case _:
                msg = "Either `value` or `strategy` must be provided."
                raise ValueError(msg)

    def _fill_strategy(self, func_name: Literal["LAST_VALUE", "FIRST_VALUE"]) -> Self:
        """Internal fill strategy using window functions."""
        match func_name:
            case "FIRST_VALUE":
                rows_start = pc.Some(0)
                rows_end = pc.NONE
                fill_func = sql.fns.first_value
            case "LAST_VALUE":
                rows_start = pc.NONE
                rows_end = pc.Some(0)
                fill_func = sql.fns.last_value
        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: sql.coalesce(
                        sql.col(c),
                        sql.WindowExpr(
                            rows_start=rows_start, rows_end=rows_end, ignore_nulls=True
                        ).call(
                            fill_func(sql.col(c)),
                        ),
                    ).alias(c)
                )
            )
        )

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values."""
        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: sql.when(sql.fns.isnan(sql.col(c)), sql.from_expr(value))
                    .otherwise(sql.col(c))
                    .alias(c)
                )
            )
        )

    def drop_nulls(self, subset: str | Iterable[str] | None = None) -> Self:
        """Drop rows with null values."""
        return self._col_or_subset(subset).fold(
            self,
            lambda lf, c: lf.__from_lf__(lf._rel.filter(sql.col(c).isnotnull())),
        )

    def drop_nans(self, subset: str | Iterable[str] | None = None) -> Self:
        """Drop rows with NaN values."""
        return self._col_or_subset(subset).fold(
            self,
            lambda lf, c: lf.__from_lf__(lf._rel.filter(~sql.fns.isnan(sql.col(c)))),
        )

    def _col_or_subset(self, subset: str | Iterable[str] | None) -> pc.Iter[str]:
        match subset:
            case None:
                return self.columns.iter()
            case str():
                return pc.Iter.once(subset)
            case Iterable():
                return pc.Iter(subset)

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        """Add a row index column."""
        rel = self._rel.row_number("over () as __rn__", "*")
        match offset:
            case 0:
                return self.__from_lf__(
                    rel.project(f'(__rn__ - 1)::BIGINT AS "{name}", * EXCLUDE (__rn__)')
                )
            case _:
                return self.__from_lf__(
                    rel.project(
                        f'(__rn__ - 1 + {offset})::BIGINT AS "{name}", * EXCLUDE (__rn__)'
                    )
                )

    # Deprecated alias
    with_row_count = with_row_index

    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Self:
        """Shift values by n positions."""
        shift_func = sql.fns.lag if n > 0 else sql.fns.lead
        abs_n = abs(n)
        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: sql.coalesce(
                        sql.WindowExpr().call(
                            shift_func(
                                sql.col(c),
                                sql.lit(abs_n),
                            ),
                        ),
                        sql.from_value(fill_value),
                    ).alias(c)
                )
            )
        )

    def clone(self) -> Self:
        """Create a copy of the LazyFrame."""
        return self.__from_lf__(self._rel)

    @property
    def columns(self) -> pc.Vec[str]:
        """Get column names."""
        return pc.Vec.from_ref(self._rel.columns)

    @property
    def dtypes(self) -> pc.Vec[str]:
        """Get column data types."""
        return pc.Vec.from_ref(self._rel.dtypes)

    @property
    def width(self) -> int:
        """Get number of columns."""
        return len(self._rel.columns)

    @property
    def schema(self) -> pc.Dict[str, str]:
        """Get the schema as a dictionary."""
        return self.columns.iter().zip(self._rel.dtypes).collect(pc.Dict)

    def collect_schema(self) -> pc.Dict[str, str]:
        """Collect the schema (same as schema property for lazy)."""
        return self.schema

    def quantile(self, quantile: float) -> Self:
        """Compute quantile for each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: sql.fns.quantile_cont(
                        sql.col(c),
                        sql.lit(quantile),
                    ).alias(c)
                )
            )
        )

    def gather_every(self, n: int, offset: int = 0) -> Self:
        """Take every nth row."""
        return self.__from_lf__(
            self._rel.row_number("over () as __rn__", "*")
            .filter(f"(__rn__ - 1 - {offset}) % {n} = 0 AND __rn__ > {offset}")
            .project("* EXCLUDE (__rn__)")
        )

    def top_k(
        self, k: int, *, by: IntoExpr | Iterable[IntoExpr], reverse: bool = False
    ) -> Self:
        """Return top k rows by column(s)."""
        return self.sort(by, descending=not reverse).head(k)

    def bottom_k(
        self, k: int, *, by: IntoExpr | Iterable[IntoExpr], reverse: bool = False
    ) -> Self:
        """Return bottom k rows by column(s)."""
        return self.sort(by, descending=reverse).head(k)

    def cast(
        self, dtypes: Mapping[str, datatypes.DataType] | datatypes.DataType
    ) -> Self:
        """Cast columns to specified dtypes."""
        match dtypes:
            case Mapping():
                dtype_map = pc.Dict(dtypes)
                exprs = self.columns.iter().map(
                    lambda c: (
                        sql.col(c).cast(dtype_map.get_item(c).unwrap()).alias(c)
                        if c in dtypes
                        else sql.col(c)
                    )
                )
            case _:
                exprs = self.columns.iter().map(
                    lambda c: sql.col(c).cast(dtypes).alias(c)
                )
        return self.__from_lf__(self._rel.select(*exprs))

    def sink_parquet(
        self,
        path: str | Path,
        *,
        compression: str = "zstd",
    ) -> None:
        """Write to Parquet file."""
        self._rel.write_parquet(str(path), compression=compression)

    def sink_csv(
        self,
        path: str | Path,
        *,
        separator: str = ",",
        include_header: bool = True,
    ) -> None:
        """Write to CSV file."""
        self._rel.write_csv(str(path), sep=separator, header=include_header)

    def sink_ndjson(self, path: str | Path) -> None:
        """Write to newline-delimited JSON file."""
        self._rel.pl(lazy=True).sink_ndjson(str(path))
