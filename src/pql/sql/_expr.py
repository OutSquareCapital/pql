from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, ClassVar, Self

import pyochain as pc

from ._code_gen import (
    ArrayFns,
    DateTimeFns,
    EnumFns,
    Expression,
    Fns,
    JsonFns,
    ListFns,
    MapFns,
    RegexFns,
    StringFns,
    StructFns,
)
from ._core import func
from ._window import Kword, OverBuilder, get_order_by, get_partition_by
from .utils import try_iter

if TYPE_CHECKING:
    from .typing import FrameMode, IntoExpr, IntoExprColumn, RoundMode, WindowExclude
    from .utils import TryIter


class SqlExpr(Expression, Fns):
    """A wrapper around duckdb.Expression that provides operator overloading and SQL function methods."""

    __slots__: ClassVar[Iterable[str]] = ()

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

    def shift(self, n: int = 1) -> Self:
        match n:
            case 0:
                return self
            case n_val if n_val > 0:
                return self.lag(n_val).over()
            case _:
                return self.lead(-n).over()

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
        partition_by: TryIter[IntoExprColumn] | None = None,
        order_by: TryIter[IntoExprColumn] | None = None,
        frame_start: int | str | None = None,
        frame_end: int | str | None = None,
        frame_mode: FrameMode = "ROWS",
        exclude: WindowExclude | None = None,
        filter_cond: IntoExprColumn | None = None,
        fn_order_by: TryIter[IntoExprColumn] | None = None,
        *,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
        ignore_nulls: bool = False,
        distinct: bool = False,
        fn_descending: TryIter[bool] = False,
        fn_nulls_last: TryIter[bool] = False,
    ) -> Self:
        ordr = pc.Option(order_by)
        return self.__class__(
            OverBuilder(str(self))
            .handle_nulls(ignore_nulls=ignore_nulls)
            .handle_distinct(distinct=distinct)
            .handle_fn_order_by(
                pc.Option(fn_order_by),
                fn_descending=fn_descending,
                fn_nulls_last=fn_nulls_last,
            )
            .handle_filter(filter_cond=pc.Option(filter_cond))
            .join_clauses(
                pc.Option(partition_by)
                .map(lambda x: try_iter(x).collect())
                .into(get_partition_by),
                ordr.map(lambda x: try_iter(x).collect()).into(
                    get_order_by, descending=descending, nulls_last=nulls_last
                ),
                Kword.frame_clause(
                    frame_mode,
                    pc.Option(frame_start),
                    pc.Option(frame_end),
                    has_order_by=ordr.is_some(),
                ),
                pc.Option(exclude).map(Kword.exclude_clause).unwrap_or(""),  # pyright: ignore[reportArgumentType]
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

    def cume_dist(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
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
            OverBuilder(str(func("cume_dist"))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def dense_rank(self) -> Self:
        """The rank of the current row without gaps; this function counts peer groups.

        Returns:
            Self
        """
        return self._new(func("dense_rank"))

    def fill(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """Replaces NULL values of expr with a linear interpolation based on the closest non-NULL values and the sort values.

        Both values must support arithmetic and there must be only one ordering key.

        For missing values at the ends, linear extrapolation is used.

        Failure to interpolate results in the NULL value being retained.

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("fill", self.inner()))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def first_value(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """Returns expr evaluated at the row that is the first row (with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an `ORDER BY` clause is specified, the first row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("first_value", self.inner()))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def lag(  # noqa: PLR0913
        self,
        offset: IntoExprColumn | int = 1,
        default: IntoExpr | None = None,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """Returns expr evaluated at the row that is offset rows (among rows with a non-null value of expr if IGNORE NULLS is set) before the current row within the window frame.

        If there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row.

        If omitted, offset defaults to 1 and default to NULL.

        If an `ORDER BY` clause is specified, the lagged row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (IntoExprColumn | int): Number of rows to look back (default: 1)
            default (IntoExpr | None): Default value if no such row exists (default: NULL)
            order_by (TryIter[IntoExprColumn] | None): Secondary ordering within the function call
            ignore_nulls (bool): Skip NULL values when looking back
            descending (TryIter[bool]): Descending order for `order_by`
            nulls_last (TryIter[bool]): Nulls-last for `order_by`

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("lag", self.inner(), offset, default))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def last_value(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """Returns expr evaluated at the row that is the last row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an `ORDER BY` clause is specified, the last row is determined within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("last_value", self.inner()))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def lead(  # noqa: PLR0913
        self,
        offset: IntoExprColumn | int = 1,
        default: IntoExpr | None = None,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """Returns expr evaluated at the row that is offset rows after the current row (among rows with a non-null value of expr if IGNORE NULLS is set) within the window frame.

        If there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL.

        If an `ORDER BY` clause is specified, the leading row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (IntoExprColumn | int): Number of rows to look ahead (default: 1)
            default (IntoExpr | None): Default value if no such row exists (default: NULL)
            order_by (TryIter[IntoExprColumn] | None): Secondary ordering within the function call
            ignore_nulls (bool): Skip NULL values when looking ahead
            descending (TryIter[bool]): Descending order for `order_by`
            nulls_last (TryIter[bool]): Nulls-last for `order_by`

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("lead", self.inner(), offset, default))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def nth_value(
        self,
        nth: IntoExprColumn | int,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """Returns expr evaluated at the nth row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame (counting from 1).

        Return `NULL` if no such row.

        If an `ORDER BY` clause is specified, the nth row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            nth (IntoExprColumn | int): The row number to retrieve (1-based)
            order_by (TryIter[IntoExprColumn] | None): Secondary ordering within the function call
            ignore_nulls (bool): Skip NULL values when counting
            descending (TryIter[bool]): Descending order for `order_by`
            nulls_last (TryIter[bool]): Nulls-last for `order_by`

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("nth_value", self.inner(), nth))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def ntile(
        self,
        num_buckets: IntoExprColumn | int,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """An integer ranging from 1 to num_buckets, dividing the partition as equally as possible.

        If an `ORDER BY` clause is specified, the ntile is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            num_buckets (IntoExprColumn | int): Number of buckets to divide into
            order_by (TryIter[IntoExprColumn] | None): Secondary ordering within the function call
            ignore_nulls (bool): Ignored (ntile does not support IGNORE NULLS)
            descending (TryIter[bool]): Descending order for `order_by`
            nulls_last (TryIter[bool]): Nulls-last for `order_by`

        Returns:
            Self
        """
        return self._new(
            OverBuilder(str(func("ntile", num_buckets))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def percent_rank(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
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
            OverBuilder(str(func("percent_rank"))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def rank(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
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
            OverBuilder(str(func("rank"))).build_fn(
                order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def row_number(
        self,
        *,
        order_by: TryIter[IntoExprColumn] | None = None,
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
            OverBuilder(str(func("row_number"))).build_fn(
                order_by=pc.Option(order_by),
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
