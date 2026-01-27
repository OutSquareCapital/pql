"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, Literal, Self

import duckdb
import pyochain as pc

from ._ast import (
    FrameInit,
    IntoExpr,
    WindowSpec,
    data_to_rel,
    iter_to_exprs,
    to_expr,
    to_value,
)

if TYPE_CHECKING:
    import polars as pl

    from . import datatypes
    from ._expr import Expr


class LazyFrame:
    """LazyFrame providing Polars-like API over DuckDB relations."""

    _rel: duckdb.DuckDBPyRelation
    __slots__ = ("_rel",)

    def __init__(self, data: FrameInit = None) -> None:
        self._rel = data_to_rel(data)

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

    def select(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Self:
        """Select columns or expressions."""
        return self.__from_lf__(self._rel.select(*iter_to_exprs(*exprs)))

    def with_columns(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> Self:
        """Add or replace columns."""
        return self.__from_lf__(
            self._rel.select(duckdb.StarExpression(), *iter_to_exprs(*exprs))
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

        def _make_order(col: IntoExpr, desc: bool, nl: bool) -> duckdb.Expression:  # noqa: FBT001
            match (desc, nl):
                case (True, True):
                    return to_expr(col).desc().nulls_last()
                case (True, False):
                    return to_expr(col).desc().nulls_first()
                case (False, True):
                    return to_expr(col).asc().nulls_last()
                case (False, False):
                    return to_expr(col).asc()

        return self.__from_lf__(
            self._rel.sort(
                *iter_to_exprs(*by)
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
                    duckdb.StarExpression(),
                    WindowSpec(partition_by=cols.map(to_expr).collect())
                    .into_expr(
                        duckdb.FunctionExpression("row_number"),
                    )
                    .alias("__rn__"),
                )
                .filter(
                    duckdb.ColumnExpression("__rn__") == duckdb.ConstantExpression(1)
                )
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
        return self.__from_lf__(
            self._rel.select(duckdb.StarExpression(exclude=columns))
        )

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_map = pc.Dict(mapping)

        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: duckdb.ColumnExpression(c).alias(
                        rename_map.get_item(c).unwrap_or(c)
                    )
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
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "count", duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def sum(self) -> Self:
        """Aggregate the sum of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "sum", duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def mean(self) -> Self:
        """Aggregate the mean of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "avg", duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def median(self) -> Self:
        """Aggregate the median of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "median", duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def min(self) -> Self:
        """Aggregate the minimum of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "min", duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def max(self) -> Self:
        """Aggregate the maximum of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "max", duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column."""
        func = "stddev_samp" if ddof == 1 else "stddev_pop"
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        func, duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column."""
        func = "var_samp" if ddof == 1 else "var_pop"
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        func, duckdb.ColumnExpression(c)
                    ).alias(c)
                ),
                "",
            )
        )

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self.__from_lf__(
            self._rel.aggregate(
                self.columns.iter().map(
                    lambda c: duckdb.FunctionExpression(
                        "count_if",
                        duckdb.ColumnExpression(c).isnull(),
                    ).alias(c)
                ),
                "",
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
                            lambda c: duckdb.CoalesceOperator(
                                duckdb.ColumnExpression(c), to_expr(val)
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
            case "LAST_VALUE":
                rows_start = pc.NONE
                rows_end = pc.Some(0)
        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: duckdb.CoalesceOperator(
                        duckdb.ColumnExpression(c),
                        WindowSpec(
                            rows_start=rows_start, rows_end=rows_end, ignore_nulls=True
                        ).into_expr(
                            duckdb.FunctionExpression(
                                func_name.lower(), duckdb.ColumnExpression(c)
                            ),
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
                    lambda c: duckdb.CaseExpression(
                        duckdb.FunctionExpression("isnan", duckdb.ColumnExpression(c)),
                        to_expr(value),
                    )
                    .otherwise(duckdb.ColumnExpression(c))
                    .alias(c)
                )
            )
        )

    def drop_nulls(self, subset: str | Iterable[str] | None = None) -> Self:
        """Drop rows with null values."""
        return self._col_or_subset(subset).fold(
            self,
            lambda lf, c: lf.__from_lf__(
                lf._rel.filter(duckdb.ColumnExpression(c).isnotnull())
            ),
        )

    def drop_nans(self, subset: str | Iterable[str] | None = None) -> Self:
        """Drop rows with NaN values."""
        return self._col_or_subset(subset).fold(
            self,
            lambda lf, c: lf.__from_lf__(
                lf._rel.filter(
                    ~duckdb.FunctionExpression("isnan", duckdb.ColumnExpression(c))
                )
            ),
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
        if offset == 0:
            return self.__from_lf__(
                rel.project(f'(__rn__ - 1)::BIGINT AS "{name}", * EXCLUDE (__rn__)')
            )
        return self.__from_lf__(
            rel.project(
                f'(__rn__ - 1 + {offset})::BIGINT AS "{name}", * EXCLUDE (__rn__)'
            )
        )

    # Deprecated alias
    with_row_count = with_row_index

    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Self:
        """Shift values by n positions."""
        func = "LAG" if n > 0 else "LEAD"
        abs_n = abs(n)
        return self.__from_lf__(
            self._rel.select(
                *self.columns.iter().map(
                    lambda c: duckdb.CoalesceOperator(
                        WindowSpec().into_expr(
                            duckdb.FunctionExpression(
                                func.lower(),
                                duckdb.ColumnExpression(c),
                                duckdb.ConstantExpression(abs_n),
                            ),
                        ),
                        to_value(fill_value),
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
                    lambda c: duckdb.FunctionExpression(
                        "quantile_cont",
                        duckdb.ColumnExpression(c),
                        duckdb.ConstantExpression(quantile),
                    ).alias(c)
                ),
                "",
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
                        duckdb.ColumnExpression(c)
                        .cast(dtype_map.get_item(c).unwrap())
                        .alias(c)
                        if c in dtypes
                        else duckdb.ColumnExpression(c)
                    )
                )
            case _:
                exprs = self.columns.iter().map(
                    lambda c: duckdb.ColumnExpression(c).cast(dtypes).alias(c)
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
