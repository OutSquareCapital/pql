"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import pyochain as pc

from . import sql
from ._args_iter import check_by_arg, try_chain, try_flatten, try_iter
from ._datatypes import DataType
from ._expr import Expr, ExprPlan

if TYPE_CHECKING:
    import polars as pl
    from narwhals.typing import IntoFrame

    from ._typing import (
        AsofJoinStrategy,
        FillNullStrategy,
        JoinKeysRes,
        JoinStrategy,
        OptIter,
        RollingInterpolationMethod,
        UniqueKeepStrategy,
    )
    from .sql.typing import (
        IntoDict,
        IntoExpr,
        IntoExprColumn,
        IntoRel,
        NPArrayLike,
        SeqIntoVals,
    )

TEMP_NAME = "__pql_temp__"
TEMP_COL = sql.col(TEMP_NAME)
MAX_I64 = 9_223_372_036_854_775_807


class JoinKeys[T: pc.Seq[str] | str](NamedTuple):
    left: T
    right: T

    @staticmethod
    def from_on(
        on: pc.Option[str], left_on: pc.Option[str], right_on: pc.Option[str]
    ) -> JoinKeysRes[str]:
        match (on, left_on, right_on):
            case (pc.Some(on_key), pc.NONE, pc.NONE):
                return pc.Ok(JoinKeys(on_key, on_key))
            case (pc.NONE, pc.Some(lk), pc.Some(rk)):
                return pc.Ok(JoinKeys(lk, rk))
            case (pc.NONE, _, _):
                msg = "Either (`left_on` and `right_on`) or `on` keys should be specified."
                return pc.Err(ValueError(msg))
            case _:
                msg = "If `on` is specified, `left_on` and `right_on` should be None."
                return pc.Err(ValueError(msg))

    @staticmethod
    def from_by(
        by: OptIter[str], by_left: OptIter[str], by_right: OptIter[str]
    ) -> JoinKeysRes[pc.Seq[str]]:
        match (by, by_left, by_right):
            case (pc.Some(by_key), pc.NONE, pc.NONE):
                vals = try_iter(by_key).collect()
                return pc.Ok(JoinKeys(vals, vals))
            case (pc.NONE, pc.Some(bl), pc.Some(br)):
                left_vals = try_iter(bl).collect()
                right_vals = try_iter(br).collect()
                match left_vals.length() == right_vals.length():
                    case True:
                        return pc.Ok(JoinKeys(left_vals, right_vals))
                    case False:
                        msg = "`by_left` and `by_right` must have the same length."
                        return pc.Err(ValueError(msg))
            case (pc.NONE, pc.NONE, pc.NONE):
                return pc.Ok(JoinKeys(pc.Seq[str].new(), pc.Seq[str].new()))
            case (pc.NONE, _, _):
                msg = "Can not specify only `by_left` or `by_right`, you need to specify both."
                return pc.Err(ValueError(msg))
            case _:
                msg = "If `by` is specified, `by_left` and `by_right` should be None."
                return pc.Err(ValueError(msg))

    @staticmethod
    def from_how(
        how: JoinStrategy,
        on_opt: pc.Option[pc.Seq[str]],
        left_on_opt: pc.Option[pc.Seq[str]],
        right_on_opt: pc.Option[pc.Seq[str]],
    ) -> JoinKeysRes[pc.Seq[str]]:
        match (on_opt, left_on_opt, right_on_opt):
            case (pc.Some(on_vals), pc.NONE, pc.NONE):
                return pc.Ok(JoinKeys(on_vals, on_vals))
            case (pc.NONE, pc.Some(lv), pc.Some(rv)) if lv.length() == rv.length():
                return pc.Ok(JoinKeys(lv, rv))
            case (pc.NONE, pc.Some(_), pc.Some(_)):
                msg = "`left_on` and `right_on` must have the same length."
                return pc.Err(ValueError(msg))
            case (pc.Some(_), _, _):
                msg = f"If `on` is specified, `left_on` and `right_on` should be None for {how}."
                return pc.Err(ValueError(msg))
            case _:
                msg = f"Either (`left_on` and `right_on`) or `on` keys should be specified for {how}."
                return pc.Err(ValueError(msg))


class LazyGroupBy:
    __slots__ = ("_frame", "_group_expr", "_keys")

    def __init__(self, frame: LazyFrame, keys: pc.Seq[sql.SqlExpr]) -> None:
        self._frame = frame
        self._keys = keys
        self._group_expr = self._keys.iter().map(str).join(", ")

    def _agg_columns(self, func: Callable[[Expr], Expr]) -> LazyFrame:
        key_names = (
            self._keys.iter().map(lambda key: key.inner().get_name()).collect(pc.Set)
        )

        return (
            self._frame.columns.iter()
            .filter(lambda name: name not in key_names)
            .map(lambda name: func(Expr(sql.col(name))).alias(name))
            .into(self.agg)
        )

    def len(self, name: str = "len") -> LazyFrame:
        return self.agg(sql.lit(1).count().alias(name))

    def all(self) -> LazyFrame:
        return self._agg_columns(Expr.implode)

    def sum(self) -> LazyFrame:
        return self._agg_columns(Expr.sum)

    def mean(self) -> LazyFrame:
        return self._agg_columns(Expr.mean)

    def median(self) -> LazyFrame:
        return self._agg_columns(Expr.median)

    def min(self) -> LazyFrame:
        return self._agg_columns(Expr.min)

    def max(self) -> LazyFrame:
        return self._agg_columns(Expr.max)

    def first(self) -> LazyFrame:
        return self._agg_columns(Expr.first)

    def last(self) -> LazyFrame:
        return self._agg_columns(Expr.last)

    def n_unique(self) -> LazyFrame:
        return self._agg_columns(Expr.n_unique)

    def quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> LazyFrame:
        return self._agg_columns(
            lambda expr: expr.quantile(quantile, interpolation=interpolation)
        )

    def agg(
        self,
        aggregate: IntoExpr | Iterable[IntoExpr],
        *more_aggregates: IntoExpr,
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
        return (
            self._keys.iter()
            .chain(
                ExprPlan.from_inputs(
                    self._frame.columns,
                    try_chain(aggregate, more_aggregates),
                    named_exprs=named_aggs,
                ).aliased_sql()
            )
            .into(self._frame.inner().aggregate, self._group_expr)
            .pipe(self._frame.__class__)
        )


class LazyFrame(sql.CoreHandler[sql.Relation]):
    """LazyFrame providing Polars-like API over DuckDB relations."""

    __slots__ = ()

    def __init__(self, data: IntoRel) -> None:
        match data:
            case sql.Relation():
                self._inner = data
            case _:
                self._inner = sql.Relation(data)

    @classmethod
    def from_query(cls, query: str, **relations: IntoRel) -> Self:
        return cls(pc.Dict.from_ref(relations).into(sql.from_query, query))

    @classmethod
    def from_table(cls, table: str) -> Self:
        return cls(sql.from_table(table))

    @classmethod
    def from_table_function(cls, function: str) -> Self:
        return cls(sql.from_table_function(function))

    @classmethod
    def from_df(cls, df: IntoFrame) -> Self:
        return cls(sql.from_df(df))

    @classmethod
    def from_numpy(cls, arr: NPArrayLike[Any, Any]) -> Self:
        return cls(sql.from_numpy(arr))

    @classmethod
    def from_mapping(cls, mapping: IntoDict[str, Any]) -> Self:
        return cls(sql.from_mapping(mapping))

    @classmethod
    def from_sequence(cls, data: SeqIntoVals) -> Self:
        return cls(sql.from_sequence(data))

    def _select(
        self, exprs: IntoExprColumn | Iterable[IntoExprColumn], groups: str = ""
    ) -> Self:
        return (
            self.inner()
            .select(
                *try_flatten(exprs).map(lambda v: sql.into_expr(v, as_col=True)),
                groups=groups,
            )
            .pipe(self._new)
        )

    def _filter(self, predicates: IntoExprColumn | Iterable[IntoExprColumn]) -> Self:
        return try_iter(predicates).fold(
            self,
            lambda lf, p: (
                try_flatten(p)
                .map(lambda v: sql.into_expr(v, as_col=True))
                .fold(
                    lf,
                    lambda inner_lf, pred: inner_lf._new(inner_lf.inner().filter(pred)),
                )
            ),
        )

    def _agg(
        self,
        exprs: IntoExprColumn | Iterable[IntoExprColumn],
        group_expr: sql.SqlExpr | str = "",
    ) -> Self:
        return (
            self.inner()
            .aggregate(
                try_flatten(exprs).map(lambda v: sql.into_expr(v, as_col=True)),
                group_expr,
            )
            .pipe(self._new)
        )

    def _iter_slct(self, func: Callable[[str], sql.SqlExpr]) -> Self:
        return self.columns.iter().map(func).into(self._select)

    def _iter_agg(self, func: Callable[[sql.SqlExpr], sql.SqlExpr]) -> Self:
        return (
            self.columns.iter().map(lambda c: func(sql.col(c)).alias(c)).into(self._agg)
        )

    def lazy(self) -> pl.LazyFrame:
        """Get a Polars LazyFrame."""
        return self.inner().pl(lazy=True)

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame."""
        return self.inner().pl()

    def select(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        **named_exprs: IntoExpr,
    ) -> Self:
        """Select columns or expressions."""
        plan = ExprPlan.from_inputs(
            self.columns, try_chain(expr, more_exprs), named_exprs
        )
        match plan.can_use_unique():
            case True:
                return self.inner().unique(plan.unique().join(", ")).pipe(self._new)
            case False:
                match plan.is_scalar_select():
                    case True:
                        return plan.aliased_sql().into(self._agg)
                    case False:
                        return plan.aliased_sql().into(self._select)

    def with_columns(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        **named_exprs: IntoExpr,
    ) -> Self:
        """Add or replace columns."""
        cols = self.columns
        return (
            ExprPlan.from_inputs(self.columns, try_chain(expr, more_exprs), named_exprs)
            .to_updates()
            .into(
                lambda updates: (
                    cols.iter()
                    .map(
                        lambda name: updates.get_item(name).map_or(
                            sql.col(name), lambda c: c.alias(name)
                        )
                    )
                    .chain(
                        updates.items()
                        .iter()
                        .filter_star(lambda name, _expr: name not in cols)
                        .map_star(lambda name, expr: expr.alias(name))
                    )
                )
            )
            .into(self._select)
        )

    def filter(
        self,
        predicate: IntoExprColumn | Iterable[IntoExprColumn],
        *more_predicates: IntoExprColumn,
        **constraints: IntoExpr,
    ) -> Self:
        """Filter rows based on predicates and equality constraints."""
        return (
            try_chain(predicate, more_predicates)
            .map(lambda value: sql.into_expr(value, as_col=True))
            .chain(
                pc.Option(constraints)
                .map(
                    lambda mapping: (
                        pc.Dict.from_ref(mapping)
                        .items()
                        .iter()
                        .map_star(
                            lambda name, value: sql.col(name).eq(sql.into_expr(value))
                        )
                    )
                )
                .unwrap_or(pc.Iter[sql.SqlExpr].new())
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
        return (
            try_chain(by, more_by)
            .map(lambda v: sql.into_expr(v, as_col=True))
            .collect()
            .into(
                lambda sort_exprs: sort_exprs.iter().zip(
                    check_by_arg(sort_exprs, "descending", arg=descending).unwrap(),
                    check_by_arg(sort_exprs, "nulls_last", arg=nulls_last).unwrap(),
                )
            )
            .map_star(lambda expr, desc, nls: expr.set_order(desc=desc, nulls_last=nls))
            .into(lambda x: self.inner().sort(*x))
            .pipe(self._new)
        )

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self.inner().limit(n).pipe(self._new)

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self.limit(n)

    def slice(self, offset: int, length: int | None = None) -> Self:
        """Get a slice of rows."""
        slice_idx_col = "__pql_slice_idx__"
        slice_len_col = "__pql_slice_len__"

        def _with_idx_and_len() -> Self:
            return self.with_columns(
                sql.row_number().over().sub(sql.lit(1)).alias(slice_idx_col),
                sql.lit(1).count().over().alias(slice_len_col),
            )

        def _from_end_start(off: int) -> sql.SqlExpr:
            return sql.col(slice_idx_col).ge(sql.col(slice_len_col).add(sql.lit(off)))

        match (pc.Option(length), offset):
            case (pc.Some(length), _) if length < 0:
                msg = f"negative slice lengths ({length}) are invalid for LazyFrame"
                raise ValueError(msg)
            case (len_val, offset) if offset >= 0:
                return (
                    self.inner()
                    .limit(len_val.unwrap_or(MAX_I64), offset=offset)
                    .pipe(self._new)
                )
            case (pc.Some(0), _):
                return self.limit(0)
            case (pc.Some(length), offset):
                return (
                    _with_idx_and_len()
                    .filter(
                        _from_end_start(offset).and_(
                            sql.col(slice_idx_col).lt(
                                sql.col(slice_len_col)
                                .add(sql.lit(offset))
                                .add(sql.lit(length))
                            )
                        )
                    )
                    .drop(slice_idx_col, slice_len_col)
                )
            case (_, offset):
                return (
                    _with_idx_and_len()
                    .filter(_from_end_start(offset))
                    .drop(slice_idx_col, slice_len_col)
                )

    def tail(self, n: int = 5) -> Self:
        """Get the last n rows."""
        match n:
            case val if val < 0:
                msg = "`n` must be greater than or equal to 0"
                raise ValueError(msg)
            case 0:
                return self.limit(0)
            case _:
                return self.slice(-n)

    def drop(self, *columns: str) -> Self:
        """Drop columns from the frame."""
        return self._select(sql.all(exclude=columns))

    def drop_nulls(self, subset: str | Iterable[str] | None = None) -> Self:
        """Drop rows that contain null values."""
        return (
            pc.Option(subset)
            .map(try_iter)
            .unwrap_or_else(self.columns.iter)
            .map(lambda name: sql.col(name).is_not_null())
            .into(self._filter)
        )

    def explode(self, columns: str | Sequence[str], *more_columns: str) -> Self:
        """Explode list-like columns."""
        to_explode_names = try_chain(columns, more_columns).collect()
        to_explode = to_explode_names.iter().map(sql.col).collect()
        target = (
            to_explode.first()
            if to_explode.length() == 1
            else (
                to_explode.first().list.zip(
                    *to_explode.iter().skip(1),
                    sql.lit(1).eq(sql.lit(1)),
                )
            )
        )

        zipped_index = (
            to_explode_names.iter()
            .enumerate()
            .map_star(lambda idx, name: (name, idx + 1))
            .collect(pc.Dict)
        )
        is_single_explode = to_explode.length() == 1

        def _explode_expr(name: str, replace: sql.SqlExpr) -> sql.SqlExpr:
            match is_single_explode:
                case True:
                    return replace.alias(name)
                case False:
                    return replace.struct.extract(
                        sql.lit(zipped_index.get_item(name).unwrap()),
                    ).alias(name)

        def _project_col(
            name: str, *, unnest: bool, replace: sql.SqlExpr
        ) -> sql.SqlExpr:
            match (unnest, name in to_explode_names):
                case (True, True):
                    return _explode_expr(name, replace)
                case (False, True):
                    return sql.lit(None).alias(name)
                case _:
                    return sql.col(name)

        def _proj(*, unnest: bool) -> pc.Iter[sql.SqlExpr]:
            replace = sql.unnest(target) if unnest else sql.lit(None)
            return self.columns.iter().map(
                lambda name: _project_col(name, unnest=unnest, replace=replace)
            )

        return (
            target.is_not_null()
            .and_(target.len().gt(sql.lit(0)))
            .pipe(
                lambda cond: (
                    self.inner()
                    .filter(cond)
                    .select(*_proj(unnest=True))
                    .union(
                        self.inner().filter(cond.not_()).select(*_proj(unnest=False))
                    )
                )
            )
            .pipe(self._new)
        )

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_map = pc.Dict(mapping)

        return self._iter_slct(
            lambda c: sql.col(c).alias(rename_map.get_item(c).unwrap_or(c))
        )

    def explain(self) -> str:
        """Generate SQL string.

        If `sqlparse` is installed, the SQL output will be formatted for better readability.
        """

        def _try_import() -> pc.Option[Any]:
            try:
                import sqlparse

                return pc.Some(sqlparse)
            except ImportError:
                return pc.NONE

        qry = self.inner().sql_query()
        return (
            _try_import()
            .map(lambda sp: sp.format(qry, reindent_aligned=True, keyword_case="upper"))
            .unwrap_or(qry)
        )

    def unnest(
        self,
        columns: IntoExprColumn | Iterable[IntoExprColumn],
        *more_columns: IntoExprColumn,
    ) -> Self:
        return (
            try_chain(columns, more_columns)
            .map(lambda v: sql.into_expr(v, as_col=True))
            .collect()
            .into(
                lambda unnest_cols: (
                    unnest_cols.iter()
                    .map(sql.unnest)
                    .insert(sql.all(exclude=unnest_cols))
                )
            )
            .into(self._select)
        )

    def first(self) -> Self:
        """Get the first row."""
        return self.head(1)

    def last(self) -> Self:
        """Get the last row."""
        return self.tail(1)

    def count(self) -> Self:
        """Return the count of each column."""
        return self._iter_agg(sql.SqlExpr.count)

    def describe(self) -> Self:
        """Return descriptive statistics."""
        return self.inner().describe().pipe(self._new)

    def sum(self) -> Self:
        """Aggregate the sum of each column."""
        return self._iter_agg(sql.SqlExpr.sum)

    def mean(self) -> Self:
        """Aggregate the mean of each column."""
        return self._iter_agg(sql.SqlExpr.avg)

    def median(self) -> Self:
        """Aggregate the median of each column."""
        return self._iter_agg(sql.SqlExpr.median)

    def min(self) -> Self:
        """Aggregate the minimum of each column."""
        return self._iter_agg(sql.SqlExpr.min)

    def max(self) -> Self:
        """Aggregate the maximum of each column."""
        return self._iter_agg(sql.SqlExpr.max)

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column."""
        return self._iter_agg(
            sql.SqlExpr.stddev_samp if ddof == 1 else sql.SqlExpr.stddev_pop
        )

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column."""
        return self._iter_agg(
            sql.SqlExpr.var_samp if ddof == 1 else sql.SqlExpr.var_pop
        )

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self._iter_agg(lambda c: c.is_null().count_if())

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values."""
        return self._iter_slct(
            lambda c: sql.when(sql.col(c).isnan()).then(value).otherwise(c).alias(c)
        )

    def fill_null(
        self,
        value: IntoExpr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        """Fill null values."""
        return self._iter_slct(
            lambda c: (
                Expr(sql.col(c))
                .fill_null(value=value, strategy=strategy, limit=limit)
                .inner()
                .alias(c)
            )
        )

    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Self:
        """Shift values by n positions."""
        abs_n = abs(n)
        match n:
            case n_val if n_val < 0:

                def _shift_fn(c: str) -> sql.SqlExpr:
                    return sql.col(c).lead(abs_n).alias(c)

            case _:

                def _shift_fn(c: str) -> sql.SqlExpr:
                    return sql.col(c).lag(abs_n).alias(c)

        fill_expr = sql.into_expr(fill_value)
        return self._iter_slct(
            lambda c: sql.coalesce(_shift_fn(c).over(), fill_expr).alias(c)
        )

    def clone(self) -> Self:
        """Create a copy of the LazyFrame."""
        return self._new(self.inner())

    def gather_every(self, n: int, offset: int = 0) -> Self:
        """Take every nth row starting from offset."""
        return (
            self.with_row_index(name=TEMP_NAME, order_by=self.columns)
            .filter(
                TEMP_COL.ge(sql.lit(offset)).and_(
                    TEMP_COL.sub(sql.lit(offset)).mod(sql.lit(n)).eq(sql.lit(0))
                )
            )
            .drop(TEMP_NAME)
        )

    @property
    def columns(self) -> pc.Vec[str]:
        """Get column names."""
        return self.inner().columns

    @property
    def width(self) -> int:
        """Get number of columns."""
        return self.columns.length()

    @property
    def schema(self) -> pc.Dict[str, DataType]:
        return (
            self.columns.iter()
            .zip(self.inner().dtypes, strict=True)
            .map_star(
                lambda name, dtype: (
                    name,
                    DataType.__from_sql__(sql.parse_dtype(dtype)),
                )
            )
            .collect(pc.Dict)
        )

    def collect_schema(self) -> pc.Dict[str, DataType]:
        """Collect the schema (same as schema property for lazy)."""
        return self.schema

    def group_by(
        self, *keys: IntoExpr | Iterable[IntoExpr], drop_null_keys: bool = False
    ) -> LazyGroupBy:
        """Start a group by operation."""
        key_exprs = (
            try_iter(keys)
            .flat_map(try_iter)
            .map(lambda key: sql.into_expr(key, as_col=True))
            .collect()
        )
        grouped_frame = (
            key_exprs.iter().map(lambda key: key.is_not_null()).into(self._filter)
            if drop_null_keys
            else self
        )
        return LazyGroupBy(grouped_frame, key_exprs)

    def join(  # noqa: PLR0913
        self,
        other: Self,
        on: str | Iterable[str] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Iterable[str] | None = None,
        right_on: str | Iterable[str] | None = None,
        suffix: str = "_right",
    ) -> Self:
        """Join with another LazyFrame."""
        on_opt = pc.Option(on).map(lambda value: try_iter(value).collect())
        left_on_opt = pc.Option(left_on).map(lambda value: try_iter(value).collect())
        right_on_opt = pc.Option(right_on).map(lambda value: try_iter(value).collect())

        def _validate_cross() -> pc.Result[None, ValueError]:
            match (on_opt, left_on_opt, right_on_opt):
                case (pc.NONE, pc.NONE, pc.NONE):
                    return pc.Ok(None)
                case _:
                    msg = (
                        "Can not pass `left_on`, `right_on` or `on` keys for cross join"
                    )

                    return pc.Err(ValueError(msg))

        match how:
            case "cross":
                _validate_cross().unwrap()
                right_on_set = pc.Set[str].new()
                rel = (
                    self.inner().set_alias("lhs").cross(other.inner().set_alias("rhs"))
                )
            case _:
                join_keys = JoinKeys.from_how(
                    how, on_opt, left_on_opt, right_on_opt
                ).unwrap()
                right_on_set = join_keys.right.iter().collect(pc.Set)
                rel = (
                    self.inner()
                    .set_alias("lhs")
                    .join(
                        other.inner().set_alias("rhs"),
                        condition=join_keys.left.iter()
                        .zip(join_keys.right)
                        .map_star(
                            lambda left, right: sql.col(f'lhs."{left}"').eq(
                                sql.col(f'rhs."{right}"')
                            )
                        )
                        .reduce(sql.SqlExpr.and_),
                        how=how,
                    )
                )

        def _rhs_expr(name: str) -> pc.Option[sql.SqlExpr]:
            col_in_lhs = name in self.columns
            is_join_key = name in right_on_set
            match (how == "outer", col_in_lhs, is_join_key):
                case (False, _, True):
                    return pc.NONE
                case (False, True, False) | (True, True, _):
                    return pc.Some(sql.col(f'rhs."{name}"').alias(f"{name}{suffix}"))
                case _:
                    return pc.Some(sql.col(f'rhs."{name}"'))

        match how:
            case "inner" | "left" | "cross" | "outer":
                return (
                    self.columns.iter()
                    .map(lambda name: sql.col(f'lhs."{name}"'))
                    .chain(
                        other.columns.iter()
                        .map(_rhs_expr)
                        .filter_map(lambda expr: expr)
                    )
                    .into(lambda x: rel.select(*x))
                    .set_alias(self.inner().alias)
                    .pipe(self._new)
                )
            case _:
                return rel.select("lhs.*").set_alias(self.inner().alias).pipe(self._new)

    def join_asof(  # noqa: PLR0913
        self,
        other: Self,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | Iterable[str] | None = None,
        by_right: str | Iterable[str] | None = None,
        by: str | Iterable[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self:
        """Perform an asof join."""
        rhs = "_rhs"
        lhs = "_lhs"
        on_keys = JoinKeys.from_on(
            pc.Option(on), pc.Option(left_on), pc.Option(right_on)
        ).unwrap()
        by_keys = JoinKeys.from_by(
            pc.Option(by), pc.Option(by_left), pc.Option(by_right)
        ).unwrap()

        left_columns = self.columns.iter().collect(pc.Set)
        drop_keys = by_keys.right.iter().collect(pc.SetMut)
        if on is not None:
            drop_keys.add(on_keys.right)

        def _rhs_expr(name: str) -> pc.Option[sql.SqlExpr]:
            match (name in drop_keys, name in left_columns):
                case (True, _):
                    return pc.NONE
                case (False, True):
                    return pc.Some(sql.col(f'{rhs}."{name}"').alias(f"{name}{suffix}"))
                case _:
                    return pc.Some(sql.col(f'{rhs}."{name}"'))

        lhs_select = self.columns.iter().map(lambda c: sql.col(f'{lhs}."{c}"'))
        rhs_select = other.columns.iter().map(_rhs_expr).filter_map(lambda x: x)

        def _expr_sql(expr: sql.SqlExpr) -> str:
            base_sql = str(expr)
            alias_name = expr.inner().get_name()
            return base_sql if alias_name == base_sql else f"{base_sql} AS {alias_name}"

        by_cond = (
            by_keys.left.iter()
            .zip(by_keys.right)
            .map_star(
                lambda left, right: sql.col(f'{lhs}."{left}"').eq(
                    sql.col(f'{rhs}."{right}"')
                )
            )
        )
        lhs_key = sql.col(f'{lhs}."{on_keys.left}"')
        rhs_key = sql.col(f'{rhs}."{on_keys.right}"')
        lhs_cols = lhs_select.chain(rhs_select).map(_expr_sql).join(", ")

        def _simple_asof_join(comparison: Callable[[sql.SqlExpr], sql.SqlExpr]) -> Self:
            return self.from_query(
                f"""--sql
                        SELECT {lhs_cols}
                        FROM _lhs
                        ASOF LEFT JOIN _rhs ON {by_cond.chain(pc.Iter.once(comparison(rhs_key))).reduce(sql.SqlExpr.and_)}
                        """,
                _lhs=self.inner().set_alias(lhs).inner(),
                _rhs=other.inner().set_alias(rhs).inner(),
            )

        match strategy:
            case "backward":
                return _simple_asof_join(lhs_key.ge)
            case "forward":
                return _simple_asof_join(lhs_key.le)
            case "nearest":
                asof_order = "__pql_asof_order__"
                asof_rank = "__pql_asof_rank__"
                condition = by_cond.chain(pc.Iter.once(lhs_key.ge(rhs_key))).reduce(
                    sql.SqlExpr.and_
                )

                return (
                    self.with_row_index(name=TEMP_NAME, order_by=self.columns)
                    .inner()
                    .set_alias(lhs)
                    .join(other.inner().set_alias(rhs), condition=condition, how="left")
                    .select(
                        sql.all(exclude=drop_keys),
                        sql.col(f'{rhs}."{on_keys.right}"').alias(asof_order),
                    )
                    .select(
                        sql.all(),
                        sql.row_number()
                        .over(
                            partition_by=TEMP_COL,
                            order_by=sql.col(asof_order)
                            .sub(sql.col(on_keys.left))
                            .abs(),
                        )
                        .alias(asof_rank),
                    )
                    .filter(sql.col(asof_rank).eq(sql.lit(1)))
                    .select(sql.all(exclude=(asof_rank, asof_order, TEMP_NAME)))
                    .pipe(self._new)
                )

    def quantile(self, quantile: float) -> Self:
        """Compute quantile for each column."""
        return self._iter_agg(lambda c: c.quantile_cont(sql.lit(quantile)))

    def unique(
        self,
        subset: str | list[str] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        order_by: str | Sequence[str] | None = None,
    ) -> Self:
        """Drop duplicate rows from this LazyFrame."""
        order_by_opt = pc.Option(order_by).map(lambda value: try_iter(value).collect())
        (
            pc.Err(
                ValueError(
                    "`order_by` must be specified when `keep` is 'first' or 'last' "
                    "because LazyFrame makes no assumptions about row order."
                )
            )
            if keep in {"first", "last"} and order_by_opt is pc.NONE
            else pc.Ok(None)
        ).unwrap()

        def _marker(subset_cols: Iterable[IntoExprColumn]) -> sql.SqlExpr:
            match keep:
                case "none":
                    return sql.all().count().over(partition_by=subset_cols)
                case "first":
                    return sql.row_number().over(
                        partition_by=subset_cols,
                        order_by=order_by_opt.unwrap(),
                    )
                case "last":
                    return sql.row_number().over(
                        partition_by=subset_cols,
                        order_by=order_by_opt.unwrap(),
                        descending=True,
                        nulls_last=True,
                    )
                case _:
                    return sql.row_number().over(partition_by=subset_cols)

        return (
            pc.Option(subset)
            .map(try_iter)
            .unwrap_or(self.columns.iter())
            .into(_marker)
            .alias(TEMP_NAME)
            .pipe(self.with_columns)
            .filter(TEMP_COL.eq(1))
            .drop(TEMP_NAME)
        )

    def unpivot(
        self,
        on: str | list[str] | None = None,
        *,
        index: str | list[str] | None = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> Self:
        """Unpivot from wide to long format."""
        index_cols = (
            pc.Option(index)
            .map(lambda value: try_iter(value).collect())
            .unwrap_or(pc.Seq[str].new())
        )
        on_cols = (
            pc.Option(on)
            .map(lambda value: try_iter(value).collect())
            .unwrap_or(
                self.columns.iter()
                .filter(lambda name: name not in index_cols)
                .collect()
            )
            .join(", ")
        )
        return self.from_query(
            f"""--sql
                    SELECT {index_cols.iter().chain((variable_name, value_name)).join(", ")}
                    FROM (UNPIVOT _rel ON {on_cols}
                    INTO NAME {variable_name} VALUE {value_name})
                    """,
            _rel=self.inner().inner(),
        )

    def with_row_index(
        self,
        name: str,
        *,
        order_by: str | Sequence[str],
    ) -> Self:
        """Insert row index based on order_by."""
        return self._select(
            (
                sql.row_number()
                .over(order_by=try_iter(order_by))
                .sub(sql.lit(1))
                .alias(name),
                sql.all(),
            )
        )

    def top_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        *,
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        """Return top k rows by column(s)."""

        def _descending() -> bool | Sequence[bool]:
            match reverse:
                case bool():
                    return not reverse
                case _:
                    return pc.Iter(reverse).map(lambda x: not x).collect()

        return self.sort(by, descending=_descending()).head(k)

    def bottom_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        *,
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        """Return bottom k rows by column(s)."""
        return self.sort(by, descending=reverse).head(k)

    def cast(self, dtypes: Mapping[str, DataType] | DataType) -> Self:
        """Cast columns to specified dtypes."""
        match dtypes:
            case Mapping():
                dtype_map = pc.Dict(dtypes)
                return self._iter_slct(
                    lambda c: (
                        dtype_map.get_item(c)
                        .map(
                            lambda dtype: (
                                sql.col(c).cast(dtype.raw.to_duckdb()).alias(c)
                            )
                        )
                        .unwrap_or(sql.col(c))
                    )
                )
            case _:
                return self._iter_slct(
                    lambda c: sql.col(c).cast(dtypes.raw.to_duckdb()).alias(c)
                )

    def sink_parquet(self, path: str | Path, *, compression: str = "zstd") -> None:
        """Write to Parquet file."""
        self.inner().write_parquet(str(path), compression=compression)

    def sink_csv(
        self, path: str | Path, *, separator: str = ",", include_header: bool = True
    ) -> None:
        """Write to CSV file."""
        self.inner().write_csv(str(path), sep=separator, header=include_header)

    def sink_ndjson(self, path: str | Path) -> None:
        """Write to newline-delimited JSON file."""
        self.inner().pl(lazy=True).sink_ndjson(path)
