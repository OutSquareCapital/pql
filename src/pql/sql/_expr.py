from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Self

import duckdb
import pyochain as pc

from ._code_gen import (
    ArrayFns,
    DateTimeFns,
    Expression,
    Fns,
    JsonFns,
    ListFns,
    MapFns,
    RegexFns,
    StringFns,
    StructFns,
)
from ._core import func, into_duckdb, try_flatten, try_iter
from ._window import over_expr

if TYPE_CHECKING:
    from ._typing import IntoExpr, IntoExprColumn


def row_number() -> SqlExpr:
    """Create a ROW_NUMBER() expression."""
    return SqlExpr(duckdb.FunctionExpression("row_number"))


def fn_once(lhs: Any, rhs: SqlExpr) -> SqlExpr:  # noqa: ANN401
    return SqlExpr(duckdb.LambdaExpression(lhs, rhs.inner()))


def all(*, exclude: Iterable[IntoExprColumn] | None = None) -> SqlExpr:
    return (
        pc.Option(exclude)
        .map(lambda x: pc.Iter(x).map(lambda e: into_expr(e, as_col=True).inner()))
        .map(lambda exc: SqlExpr(duckdb.StarExpression(exclude=exc)))
        .unwrap_or(SqlExpr(duckdb.StarExpression()))
    )


def when(condition: IntoExpr, value: IntoExpr) -> SqlExpr:
    return SqlExpr(
        duckdb.CaseExpression(
            into_expr(condition).inner(), into_expr(value, as_col=True).inner()
        )
    )


def col(name: str) -> SqlExpr:
    """Create a column expression."""
    return SqlExpr(duckdb.ColumnExpression(name))


def lit(value: IntoExpr) -> SqlExpr:
    """Create a literal expression."""
    return SqlExpr(duckdb.ConstantExpression(into_duckdb(value)))


def raw(sql: str) -> SqlExpr:
    """Create a raw SQL expression."""
    return SqlExpr(duckdb.SQLExpression(sql))


def coalesce(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    """Create a COALESCE expression."""
    return SqlExpr(
        duckdb.CoalesceOperator(
            *try_iter(exprs).chain(more_exprs).map(lambda e: into_expr(e).inner())
        )
    )


def into_expr(value: IntoExpr, *, as_col: bool = False) -> SqlExpr:
    """Convert a value to a DuckDB Expression.

    Args:
        value (IntoExpr): The value to convert.
        as_col (bool): Whether to treat `str` values as column names (default: `False`).

    Returns:
        SqlExpr: The resulting DuckDB wrapper Expression.
    """
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case Expr():
            return value.inner()
        case str() if as_col:
            return col(value)
        case _:
            return lit(value)


def args_into_exprs(
    exprs: Iterable[IntoExpr | Iterable[IntoExpr]],
    named_exprs: dict[str, IntoExpr] | None = None,
) -> pc.Iter[SqlExpr]:
    """Convert positional and keyword arguments to an iterator of DuckDB Expressions."""
    return (
        try_flatten(exprs)
        .map(lambda v: into_expr(v, as_col=True))  # pyright: ignore[reportArgumentType]
        .chain(
            pc.Option(named_exprs)
            .map(
                lambda x: (
                    pc.Dict.from_ref(x)
                    .items()
                    .iter()
                    .map_star(
                        lambda name, expr: into_expr(expr, as_col=True).alias(name)
                    )
                )
            )
            .unwrap_or(pc.Iter[SqlExpr].new())
        )
    )


class SqlExpr(Expression, Fns):
    """A wrapper around duckdb.Expression that provides operator overloading and SQL function methods."""

    __slots__ = ()

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

    def log(self, x: Self | float | None = None) -> Self:
        """Computes the logarithm of x to base b.

        b may be omitted, in which case the default 10.

        **SQL name**: *log*

        Args:
            x (Self | float | None): `DOUBLE` expression

        Returns:
            Self
        """
        return self._new(func("log", x, self._expr))

    def implode(self) -> Self:
        """Returns a LIST containing all the values of a column.

        **SQL name**: *list*

        See Also:
            array_agg

        Returns:
            Self
        """
        return self._new(func("list", self._expr))

    def greatest(self, *args: Self) -> Self:
        """Returns the largest value.

        For strings lexicographical ordering is used.

        Note that lowercase characters are considered “larger” than uppercase characters and collations are not supported.

        **SQL name**: *greatest*

        Args:
            *args (Self): `ANY` expression

        Returns:
            Self
        """
        return self._new(func("greatest", self.inner(), *args))

    def least(self, *args: Self) -> Self:
        """Returns the smallest value.

        For strings lexicographical ordering is used.

        Note that uppercase characters are considered “smaller” than lowercase characters, and collations are not supported.

        **SQL name**: *least*

        Args:
            *args (Self): `ANY` expression

        Returns:
            Self
        """
        return self._new(func("least", self.inner(), *args))

    def to_map(self, values: Self) -> Self:
        """Creates a map from a set of keys and values.

        **SQL name**: *map*

        Args:
            values (Self): `V[]` expression

        Returns:
            Self
        """
        return self._new(func("map", self.inner(), values))

    def __str__(self) -> str:
        return self._expr.__str__()

    def add(self, other: Self) -> Self:
        return self.__add__(other)

    def and_(self, other: Self) -> Self:
        return self.__and__(other)

    def div(self, other: Self) -> Self:
        return self.__div__(other)

    def eq(self, other: Self) -> Self:
        return self.__eq__(other)

    def floordiv(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def ge(self, other: Self) -> Self:
        return self.__ge__(other)

    def gt(self, other: Self) -> Self:
        return self.__gt__(other)

    def not_(self) -> Self:
        return self.__invert__()

    def le(self, other: Self) -> Self:
        return self.__le__(other)

    def lt(self, other: Self) -> Self:
        return self.__lt__(other)

    def mod(self, other: Self) -> Self:
        return self.__mod__(other)

    def mul(self, other: Self) -> Self:
        return self.__mul__(other)

    def ne(self, other: Self) -> Self:
        return self.__ne__(other)

    def neg(self) -> Self:
        return self.__neg__()

    def or_(self, other: Self) -> Self:
        return self.__or__(other)

    def pow(self, other: Self) -> Self:
        return self.__pow__(other)

    def radd(self, other: Self) -> Self:
        return self.__radd__(other)

    def rand(self, other: Self) -> Self:
        return self.__rand__(other)

    def rdiv(self, other: Self) -> Self:
        return self.__rdiv__(other)

    def rfloordiv(self, other: Self) -> Self:
        return self.__rfloordiv__(other)

    def rmod(self, other: Self) -> Self:
        return self.__rmod__(other)

    def rmul(self, other: Self) -> Self:
        return self.__rmul__(other)

    def ror(self, other: Self) -> Self:
        return self.__ror__(other)

    def rpow(self, other: Self) -> Self:
        return self.__rpow__(other)

    def rsub(self, other: Self) -> Self:
        return self.__rsub__(other)

    def rtruediv(self, other: Self) -> Self:
        return self.__rtruediv__(other)

    def sub(self, other: Self) -> Self:
        return self.__sub__(other)

    def truediv(self, other: Self) -> Self:
        return self.__truediv__(other)

    def over(  # noqa: PLR0913
        self,
        partition_by: Iterable[SqlExpr] | SqlExpr | None = None,
        order_by: Iterable[SqlExpr] | SqlExpr | None = None,
        rows_start: int | None = None,
        rows_end: int | None = None,
        *,
        descending: Iterable[bool] | bool = False,
        nulls_last: Iterable[bool] | bool = False,
        ignore_nulls: bool = False,
    ) -> Self:
        return self._new(
            over_expr(
                self,
                pc.Option(partition_by),
                pc.Option(order_by),
                pc.Option(rows_start),
                pc.Option(rows_end),
                descending=descending,
                nulls_last=nulls_last,
                ignore_nulls=ignore_nulls,
            )
        )

    def cume_dist(self) -> Self:
        """The cumulative distribution: (number of partition rows preceding or peer with current row) / total partition rows.

        If an `ORDER BY` clause is specified, the distribution is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("cume_dist"))

    def dense_rank(self) -> Self:
        """The rank of the current row without gaps; this function counts peer groups.

        Returns:
            Self
        """
        return self._new(func("dense_rank"))

    def fill(self) -> Self:
        """Replaces NULL values of expr with a linear interpolation based on the closest non-NULL values and the sort values.

        Both values must support arithmetic and there must be only one ordering key.

        For missing values at the ends, linear extrapolation is used.

        Failure to interpolate results in the NULL value being retained.

        Returns:
            Self
        """
        return self._new(func("fill", self._expr))

    def first_value(self) -> Self:
        """Returns expr evaluated at the row that is the first row (with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an `ORDER BY` clause is specified, the first row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("first_value", self._expr))

    def lag(self, offset: SqlExpr | int = 1, default: SqlExpr | None = None) -> Self:
        """Returns expr evaluated at the row that is offset rows (among rows with a non-null value of expr if IGNORE NULLS is set) before the current row within the window frame.

        If there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row.

        If omitted, offset defaults to 1 and default to NULL.

        If an `ORDER BY` clause is specified, the lagged row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (SqlExpr | int): Number of rows to look back (default: 1)
            default (SqlExpr | None): Default value if no such row exists (default: NULL)

        Returns:
            Self
        """
        return self._new(func("lag", self._expr, offset, default))

    def last_value(self) -> Self:
        """Returns expr evaluated at the row that is the last row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an `ORDER BY` clause is specified, the last row is determined within the frame using the provided ordering instead of the frame ordering.


        Returns:
            Self
        """
        return self._new(func("last_value", self._expr))

    def lead(self, offset: SqlExpr | int = 1, default: SqlExpr | None = None) -> Self:
        """Returns expr evaluated at the row that is offset rows after the current row (among rows with a non-null value of expr if IGNORE NULLS is set) within the window frame.

        If there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL.

        If an `ORDER BY` clause is specified, the leading row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (SqlExpr | int): Number of rows to look ahead (default: 1)
            default (SqlExpr | None): Default value if no such row exists (default: NULL)

        Returns:
            Self
        """
        return self._new(func("lead", self._expr, offset, default))

    def nth_value(self, nth: SqlExpr | int) -> Self:
        """Returns expr evaluated at the nth row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame (counting from 1).

        Return `NULL` if no such row.

        If an `ORDER BY` clause is specified, the nth row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            nth (SqlExpr | int): The row number to retrieve (1-based)

        Returns:
            Self
        """
        return self._new(func("nth_value", self._expr, nth))

    def ntile(self) -> Self:
        """An integer ranging from 1 to num_buckets, dividing the partition as equally as possible.

        If an `ORDER BY` clause is specified, the ntile is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            num_buckets (SqlExpr | int): Number of buckets to divide into

        Returns:
            Self
        """
        return self._new(func("ntile", self._expr))

    def percent_rank(self) -> Self:
        """The relative rank of the current row: (rank() - 1) / (total partition rows - 1).

        If an `ORDER BY` clause is specified, the relative rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("percent_rank"))

    def rank(self) -> Self:
        """The rank of the current row with gaps; same as row_number of its first peer.

        If an `ORDER BY` clause is specified, the rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("rank"))

    def row_number(self) -> Self:
        """The number of the current row within the partition, counting from 1.

        If an `ORDER BY` clause is specified, the row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("row_number"))


@dataclass(slots=True)
class SqlExprStringNameSpace(StringFns[SqlExpr]):
    """String function namespace for SQL expressions."""

    def strftime(self, format_arg: SqlExpr | date | datetime | str) -> SqlExpr:
        """Converts a `date` to a string according to the format string.

        **SQL name**: *strftime*

        Args:
            format_arg (SqlExpr | date | datetime | str): `DATE | TIMESTAMP | TIMESTAMP_NS | VARCHAR` expression

        Returns:
            SqlExpr
        """
        return self._new(func("strftime", self.inner(), format_arg))

    def strptime(self, format_arg: SqlExpr | list[str] | str) -> SqlExpr:
        """Converts the `string` text to timestamp according to the format string.

        Throws an error on failure.

        To return `NULL` on failure, use try_strptime.

        **SQL name**: *strptime*

        Args:
            format_arg (SqlExpr | list[str] | str): `VARCHAR | VARCHAR[]` expression

        Returns:
            SqlExpr
        """
        return self._new(func("strptime", self.inner(), format_arg))

    def concat(self, *args: SqlExpr) -> SqlExpr:
        """Concatenates multiple strings or lists.

        `NULL` inputs are skipped.

        See also operator `||`.

        **SQL name**: *concat*

        Args:
            *args (SqlExpr): `ANY` expression

        Returns:
            SqlExpr
        """
        return self._new(func("concat", self.inner(), *args))


@dataclass(slots=True)
class SqlExprListNameSpace(ListFns[SqlExpr]):
    """List function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprStructNameSpace(StructFns[SqlExpr]):
    """Struct function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprDateTimeNameSpace(DateTimeFns[SqlExpr]):
    """Datetime function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprArrayNameSpace(ArrayFns[SqlExpr]):
    """Array function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprJsonNameSpace(JsonFns[SqlExpr]):
    """JSON function namespace for SQL expressions."""

    def parse(self) -> SqlExpr:
        """Parse and minify json.

        **SQL name**: *json*

        Returns:
            Self
        """
        return self._new(func("json", self.inner()))


@dataclass(slots=True)
class SqlExprRegexNameSpace(RegexFns[SqlExpr]):
    """Regex function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprMapNameSpace(MapFns[SqlExpr]):
    """Map function namespace for SQL expressions."""
