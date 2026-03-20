from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, ClassVar, Self

import pyochain as pc

from pql.sql.typing import IntoExprColumn

from ._code_gen import (
    ArrayFns,
    DateTimeFns,
    EnumFns,
    Expression,
    Fns,
    GeoSpatialFns,
    JsonFns,
    ListFns,
    MapFns,
    RegexFns,
    StringFns,
    StructFns,
)
from ._core import func
from ._window import FrameBound, OverBuilder, get_order, get_partition, make_spec

if TYPE_CHECKING:
    from .typing import (
        ClosedInterval,
        FrameMode,
        IntoExpr,
        IntoExprColumn,
        RoundMode,
        WindowExclude,
    )
    from .utils import TryIter


class SqlExpr(Expression, Fns):
    """A wrapper around duckdb.Expression that provides operator overloading and SQL function methods."""

    __slots__: ClassVar[Iterable[str]] = ()

    def to_sql(self) -> str:
        """Serialize expression to a SQL fragment, including AS alias when needed."""
        base = str(self)
        name = self.get_name()
        return base if name == base else f"{base} AS {name}"

    def _reversed(self, expr: Self, *, reverse: bool = False) -> Self:
        match reverse:
            case True:
                return expr.over(frame_start=pc.Some(0))
            case False:
                return expr.over(frame_end=pc.Some(0))

    @property
    def arr(self) -> SqlExprArrayNameSpace:
        """Access array functions."""
        return SqlExprArrayNameSpace(self)

    @property
    def str(self) -> SqlExprStringNameSpace:
        """Access string functions."""
        return SqlExprStringNameSpace(self)

    @property
    def list(self) -> SqlExprListNameSpace:
        """Access list functions."""
        return SqlExprListNameSpace(self)

    @property
    def struct(self) -> SqlExprStructNameSpace:
        """Access struct functions."""
        return SqlExprStructNameSpace(self)

    @property
    def dt(self) -> SqlExprDateTimeNameSpace:
        """Access datetime functions."""
        return SqlExprDateTimeNameSpace(self)

    @property
    def json(self) -> SqlExprJsonNameSpace:
        """Access JSON functions."""
        return SqlExprJsonNameSpace(self)

    @property
    def re(self) -> SqlExprRegexNameSpace:
        """Access regex functions."""
        return SqlExprRegexNameSpace(self)

    @property
    def map(self) -> SqlExprMapNameSpace:
        """Access map functions."""
        return SqlExprMapNameSpace(self)

    @property
    def enum(self) -> SqlExprEnumNameSpace:
        """Access enum functions."""
        return SqlExprEnumNameSpace(self)

    @property
    def geo(self) -> SqlExprGeoSpatialNameSpace:
        """Access geospatial functions."""
        return SqlExprGeoSpatialNameSpace(self)

    def cum_count(self, *, reverse: bool = False) -> Self:
        """Cumulative non-null count."""
        return self._reversed(self.count(), reverse=reverse)

    def cum_sum(self, *, reverse: bool = False) -> Self:
        """Cumulative sum."""
        return self._reversed(self.sum(), reverse=reverse)

    def cum_prod(self, *, reverse: bool = False) -> Self:
        """Cumulative product."""
        return self._reversed(self.product(), reverse=reverse)

    def cum_min(self, *, reverse: bool = False) -> Self:
        """Cumulative minimum."""
        return self._reversed(self.min(), reverse=reverse)

    def cum_max(self, *, reverse: bool = False) -> Self:
        """Cumulative maximum."""
        return self._reversed(self.max(), reverse=reverse)

    def var(self, ddof: int) -> Self:
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def std(self, ddof: int) -> Self:
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def kurtosis(self, *, bias: bool = True) -> Self:
        match bias:
            case True:
                return self.kurtosis_pop()
            case False:
                return self.kurtosis_samp()

    def skew(self, *, bias: bool) -> Self:
        adjusted = self.skewness()
        match bias:
            case False:
                return adjusted
            case True:
                n = self.count()
                factor = n.sub(2).truediv(n.mul(n.sub(1)).sqrt())
                return adjusted.mul(factor)

    def shift(self, n: int = 1) -> Self:
        match n:
            case 0:
                return self
            case n_val if n_val > 0:
                return self.lag(n_val, None).over()
            case _:
                return self.lead(-n, None).over()

    def round(self, decimals: int, mode: RoundMode) -> Self:
        match mode:
            case "half_to_even":
                return self.round_even(decimals)
            case "half_away_from_zero":
                return self.round_from_zero(decimals)

    def quantile(self, quantile: float, *, interpolation: bool = True) -> Self:
        match interpolation:
            case True:
                return self.quantile_cont(quantile)
            case False:
                return self.quantile_disc(quantile)

    def is_between(
        self, lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval
    ) -> Self:
        match closed:
            case "both":
                return self.ge(lower_bound).and_(self.le(upper_bound))
            case "left":
                return self.ge(lower_bound).and_(self.lt(upper_bound))
            case "right":
                return self.gt(lower_bound).and_(self.le(upper_bound))
            case "none":
                return self.gt(lower_bound).and_(self.lt(upper_bound))

    def clip(self, lower_bound: IntoExpr = None, upper_bound: IntoExpr = None) -> Self:
        match (lower_bound, upper_bound):
            case (None, None):
                return self
            case (None, upper):
                return self.least(upper)
            case (lower, None):
                return self.greatest(lower)
            case (lower, upper):
                return self.greatest(lower).least(upper)

    def n_unique(self) -> Self:
        """Count distinct values."""
        return self._new(self.implode().list.distinct().list.length().inner())

    def has_nulls(self) -> Self:
        """Return whether the expression contains nulls."""
        return self.is_null().any()

    def repeat_by(self, by: IntoExprColumn | int) -> Self:
        """Repeat values by count, returning a list."""
        from ._funcs import into_expr

        expr = into_expr(by, as_col=True).list.range().list.eval(self).inner()
        return self._new(expr)

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values."""
        from ._when import when

        return self._new(when(self.eq(old)).then(new).otherwise(self).inner())

    def is_close(
        self,
        other: IntoExpr,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-5,
        *,
        nans_equal: bool = False,
    ) -> Self:
        """Check if two floating point values are close."""
        from ._funcs import into_expr, lit
        from ._when import when

        other_expr = into_expr(other)
        threshold = lit(abs_tol).add(lit(rel_tol).mul(other_expr.abs()))
        close = self.sub(other_expr).abs().le(threshold)
        match nans_equal:
            case False:
                return close
            case True:
                return self._new(
                    when(self.is_nan().and_(other_expr.is_nan()))
                    .then(value=True)
                    .otherwise(close)
                    .inner()
                )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self._new(self.row_number().over(pc.Some(self)).eq(1).inner())

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return (
            self.row_number()
            .over(pc.Some(self), pc.Some(self), descending=True, nulls_last=True)
            .eq(1)
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        from ._funcs import all

        return self._new(all().count().over(pc.Some(self)).gt(1).inner())

    def is_unique(self) -> Self:
        """Check if value is unique."""
        from ._funcs import all

        return self._new(all().count().over(pc.Some(self)).eq(1).inner())

    def arg_sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Return indices that would sort the expression."""
        return (
            self.row_number()
            .over(order_by=pc.Some(self), descending=descending, nulls_last=nulls_last)
            .sub(1)
        )

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value."""
        return self.last_value().over(frame_end=pc.Some(0), ignore_nulls=True)

    def backward_fill(self, limit: int | None) -> Self:
        """Fill null values with the next non-null value."""
        expr = self.any_value()
        return (
            pc.Option(limit)
            .map(lambda lmt: expr.over(frame_start=pc.Some(0), frame_end=pc.Some(lmt)))
            .unwrap_or_else(lambda: expr.over(frame_start=pc.Some(0)))
        )

    def fill_nan(self, value: float | IntoExprColumn | None) -> Self:
        """Fill NaN values."""
        from ._when import when

        return self._new(when(self.is_nan()).then(value).otherwise(self).inner())

    def log(self, x: IntoExprColumn | float | None = None) -> Self:
        """Computes the logarithm of x to base b.

        b may be omitted, in which case the default 10.

        **SQL name**: *log*

        Args:
            x (IntoExprColumn | float | None): `DOUBLE` expression

        Returns:
            Self
        """
        return self._new(func("log", x, self.inner()))

    def greatest(self, *args: IntoExpr) -> Self:
        """Returns the largest value.

        For strings lexicographical ordering is used.

        Note that lowercase characters are considered “larger” than uppercase characters and collations are not supported.

        **SQL name**: *greatest*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            Self
        """
        return self._new(func("greatest", self.inner(), *args))

    def least(self, *args: IntoExpr) -> Self:
        """Returns the smallest value.

        For strings lexicographical ordering is used.

        Note that uppercase characters are considered “smaller” than lowercase characters, and collations are not supported.

        **SQL name**: *least*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            Self
        """
        return self._new(func("least", self.inner(), *args))

    def over(  # noqa: PLR0913
        self,
        partition_by: pc.Option[TryIter[IntoExprColumn]] = pc.NONE,
        order_by: pc.Option[TryIter[IntoExprColumn]] = pc.NONE,
        frame_start: pc.Option[FrameBound] = pc.NONE,
        frame_end: pc.Option[FrameBound] = pc.NONE,
        frame_mode: FrameMode = "ROWS",
        exclude: pc.Option[WindowExclude] = pc.NONE,
        filter_cond: pc.Option[IntoExprColumn] = pc.NONE,
        fn_order_by: pc.Option[TryIter[IntoExprColumn]] = pc.NONE,
        *,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
        ignore_nulls: bool = False,
        distinct: bool = False,
        fn_descending: TryIter[bool] = False,
        fn_nulls_last: TryIter[bool] = False,
    ) -> Self:
        order = get_order(order_by, descending=descending, nulls_last=nulls_last)
        spec = make_spec(
            frame_mode,
            has_order_by=order_by.is_some(),
            frame_start=frame_start,
            frame_end=frame_end,
            exclude=exclude,
        )
        return self.__class__(
            OverBuilder.from_expr(self.inner())
            .handle_nulls(ignore_nulls=ignore_nulls)
            .handle_distinct(distinct=distinct)
            .handle_fn_order_by(
                fn_order_by=fn_order_by,
                fn_descending=fn_descending,
                fn_nulls_last=fn_nulls_last,
            )
            .handle_filter(filter_cond)
            .handle_clauses(
                partition_by=get_partition(partition_by), order=order, spec=spec
            )
            .build()
        )

    def set_order(self, *, desc: bool, nulls_last: bool) -> Self:
        """Set the ordering of the expression. Syntactic sugar for use in parameterized functions."""
        match (desc, nulls_last):
            case (True, True):
                return self.desc().nulls_last()
            case (True, False):
                return self.desc()
            case (False, True):
                return self.asc().nulls_last()
            case (False, False):
                return self.asc()

    def dense_rank(self) -> Self:
        """The rank of the current row without gaps; this function counts peer groups.

        Returns:
            Self
        """
        return self._new(func("dense_rank"))

    def cume_dist(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The cumulative distribution: (number of partition rows preceding or peer with current row) / total partition rows.

        If an `ORDER BY` clause is specified, the distribution is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(
            OverBuilder.from_expr(func("cume_dist")).build_fn(
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def percent_rank(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The relative rank of the current row: (rank() - 1) / (total partition rows - 1).

        If an `ORDER BY` clause is specified, the relative rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(
            OverBuilder.from_expr(func("percent_rank")).build_fn(
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def rank(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The rank of the current row with gaps; same as row_number of its first peer.

        If an `ORDER BY` clause is specified, the rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(
            OverBuilder.from_expr(func("rank")).build_fn(
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def row_number(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The number of the current row within the partition, counting from 1.

        If an `ORDER BY` clause is specified, the row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(
            OverBuilder.from_expr(func("row_number")).build_fn(
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )


@dataclass(slots=True)
class SqlExprStringNameSpace(StringFns[SqlExpr]):
    """String function namespace for SQL expressions."""

    def strftime(self, format_arg: IntoExprColumn | date | datetime | str) -> SqlExpr:
        """Converts a `date` to a string according to the format string.

        **SQL name**: *strftime*

        Args:
            format_arg (IntoExprColumn | date | datetime | str): `DATE | TIMESTAMP | TIMESTAMP_NS | VARCHAR` expression

        Returns:
            SqlExpr
        """
        return self._new(func("strftime", self.inner(), format_arg))

    def strptime(self, format_arg: IntoExprColumn | list[str]) -> SqlExpr:
        """Converts the `string` text to timestamp according to the format string.

        Throws an error on failure.

        To return `NULL` on failure, use try_strptime.

        **SQL name**: *strptime*

        Args:
            format_arg (IntoExprColumn | list[str]): `VARCHAR | VARCHAR[]` expression

        Returns:
            SqlExpr
        """
        return self._new(func("strptime", self.inner(), format_arg))

    def concat(self, *args: IntoExpr) -> SqlExpr:
        """Concatenates multiple strings or lists.

        `NULL` inputs are skipped.

        See also operator `||`.

        **SQL name**: *concat*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            SqlExpr
        """
        return self._new(func("concat", self.inner(), *args))


@dataclass(slots=True)
class SqlExprListNameSpace(ListFns[SqlExpr]):
    """List function namespace for SQL expressions."""

    def eval(self, expr: SqlExpr) -> SqlExpr:
        """Run an expression against each array element."""
        from ._funcs import fn_once

        return self._new(self.transform(fn_once(expr.inner())).inner())

    def std(self, ddof: int = 1) -> SqlExpr:
        """Compute the standard deviation of the lists in the column."""
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def var(self, ddof: int = 1) -> SqlExpr:
        """Compute the variance of the lists in the column."""
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def filter(self, lambda_arg: IntoExprColumn) -> SqlExpr:
        """Constructs a list from those elements of the input `list` for which the `lambda` function returns `true`.

        DuckDB must be able to cast the `lambda` function's return type to `BOOL`.

        The return type of `list_filter` is the same as the input list's.

        **SQL name**: *filter*

        Args:
            lambda_arg (IntoExprColumn): `LAMBDA` expression

        Examples:
            filter([3, 4, 5], lambda x : x > 4)

        Returns:
            T
        """
        from ._funcs import fn_once

        return self._new(func("list_filter", self.inner(), fn_once(lambda_arg)))


@dataclass(slots=True)
class SqlExprStructNameSpace(StructFns[SqlExpr]):
    """Struct function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprDateTimeNameSpace(DateTimeFns[SqlExpr]):
    """Datetime function namespace for SQL expressions."""

    def trunc(self, precision: IntoExprColumn) -> SqlExpr:
        """Truncate to specified precision.

        **SQL name**: *date_trunc*

        Args:
            precision (IntoExprColumn): `VARCHAR` expression

        Examples:
            date_trunc('hour', TIMESTAMPTZ '1992-09-20 20:38:40')

        Returns:
            T
        """
        return self._new(func("date_trunc", precision, self.inner()))


@dataclass(slots=True)
class SqlExprArrayNameSpace(ArrayFns[SqlExpr]):
    """Array function namespace for SQL expressions."""

    def eval(self, expr: SqlExpr) -> SqlExpr:
        """Run an expression against each array element."""
        from ._funcs import fn_once

        return self._new(self.transform(fn_once(expr.inner())).inner())

    def filter(self, lambda_arg: IntoExprColumn) -> SqlExpr:
        """Constructs a list from those elements of the input `list` for which the `lambda` function returns `true`.

        DuckDB must be able to cast the `lambda` function's return type to `BOOL`.

        The return type of `list_filter` is the same as the input list's.

        **SQL name**: *filter*

        Args:
            lambda_arg (IntoExprColumn): `LAMBDA` expression

        Examples:
            filter([3, 4, 5], lambda x : x > 4)

        Returns:
            T
        """
        from ._funcs import fn_once

        return self._new(func("array_filter", self.inner(), fn_once(lambda_arg)))


@dataclass(slots=True)
class SqlExprJsonNameSpace(JsonFns[SqlExpr]):
    """JSON function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprRegexNameSpace(RegexFns[SqlExpr]):
    """Regex function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprMapNameSpace(MapFns[SqlExpr]):
    """Map function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprEnumNameSpace(EnumFns[SqlExpr]):
    """Enum function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprGeoSpatialNameSpace(GeoSpatialFns[SqlExpr]):
    """Geospatial function namespace for SQL expressions."""
