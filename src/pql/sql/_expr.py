from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Self

import duckdb
import pyochain as pc

from ._core import func
from ._window import over
from .fns import (
    ArrayFns,
    DateTimeFns,
    Fns,
    JsonFns,
    ListFns,
    StringFns,
    StructFns,
)

if TYPE_CHECKING:
    from .._expr import Expr
    from .datatypes import DataType


type PyLiteral = (
    str
    | int
    | float
    | bool
    | date
    | datetime
    | time
    | timedelta
    | bytes
    | bytearray
    | memoryview
    | list[PyLiteral]
    | dict[Any, PyLiteral]
    | None
)
type IntoExpr = PyLiteral | SqlExpr | duckdb.Expression | Expr
type IntoExprColumn = Iterable[SqlExpr] | SqlExpr | str | duckdb.Expression | Expr


def row_number() -> SqlExpr:
    """Create a ROW_NUMBER() expression."""
    return SqlExpr(duckdb.FunctionExpression("row_number"))


def fn_once(lhs: Any, rhs: SqlExpr) -> SqlExpr:  # noqa: ANN401
    return SqlExpr(duckdb.LambdaExpression(lhs, rhs.inner()))


def all(*, exclude: Any | None = None) -> SqlExpr:  # noqa: ANN401
    return SqlExpr(duckdb.StarExpression(exclude=exclude))


def when(condition: SqlExpr, value: SqlExpr) -> SqlExpr:
    return SqlExpr(duckdb.CaseExpression(condition.inner(), value.inner()))


def col(name: str) -> SqlExpr:
    """Create a column expression."""
    return SqlExpr(duckdb.ColumnExpression(name))


def lit(value: IntoExpr) -> SqlExpr:
    """Create a literal expression."""
    return SqlExpr(duckdb.ConstantExpression(value))


def raw(sql: str) -> SqlExpr:
    """Create a raw SQL expression."""
    return SqlExpr(duckdb.SQLExpression(sql))


def coalesce(*exprs: SqlExpr) -> SqlExpr:
    """Create a COALESCE expression."""
    return SqlExpr(duckdb.CoalesceOperator(*(expr.inner() for expr in exprs)))


def to_duckdb(value: SqlExpr | str | duckdb.Expression) -> duckdb.Expression | str:
    """Convert a SqlExpr to duckdb.Expression, preserving str and duckdb.Expression."""
    match value:
        case str() | duckdb.Expression():
            return value
        case SqlExpr():
            return value.inner()


def from_cols(exprs: IntoExprColumn) -> pc.Iter[duckdb.Expression | str]:
    """Convert one or more values or iterables of values to an iterable of DuckDB Expressions or strings."""
    from .._expr import Expr

    match exprs:
        case duckdb.Expression():
            return pc.Iter.once(exprs)
        case SqlExpr():
            return pc.Iter.once(exprs.inner())
        case Expr():
            return pc.Iter.once(exprs.inner().inner())
        case str():
            return pc.Iter.once(exprs)
        case Iterable():
            return pc.Iter(exprs).map(from_cols).flatten()


class SqlExpr(Fns):  # noqa: PLW1641
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
    def js(self) -> SqlExprJsonNameSpace:
        """Access JSON functions."""
        return SqlExprJsonNameSpace(self)

    @classmethod
    def from_expr(cls, value: IntoExpr) -> SqlExpr:
        """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
        from .._expr import Expr

        match value:
            case SqlExpr():
                return value
            case duckdb.Expression():
                return cls(value)
            case Expr():
                return value.inner()
            case str():
                return col(value)
            case _:
                return lit(value)

    @classmethod
    def from_value(cls, value: IntoExpr) -> SqlExpr:
        """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
        from .._expr import Expr

        match value:
            case SqlExpr():
                return value
            case duckdb.Expression():
                return SqlExpr(value)
            case Expr():
                return value.inner()
            case _:
                return lit(value)

    @classmethod
    def from_iter(cls, *values: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
        """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

        Note:
            We handle this with an external variadic argument, and an internal closure, to
            distinguish between a single iterable argument and multiple arguments.
        """

        def _single_to_expr(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
            match value:
                case str() | bytes() | bytearray():
                    return pc.Iter.once(cls.from_expr(value))
                case Iterable():
                    return pc.Iter(value).map(cls.from_expr)
                case _:
                    return pc.Iter.once(cls.from_expr(value))

        match values:
            case (single,):
                return _single_to_expr(single)
            case _:
                return pc.Iter(values).map(_single_to_expr).flatten()

    @classmethod
    def from_args_kwargs(
        cls, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> pc.Iter[SqlExpr]:
        """Convert positional and keyword arguments to an iterator of DuckDB Expressions."""
        return cls.from_iter(*exprs).chain(
            pc.Dict.from_ref(named_exprs)
            .items()
            .iter()
            .map_star(lambda name, expr: cls.from_expr(expr).alias(name))
        )

    def log(self, x: Self | float | None = None) -> Self:
        """Computes the logarithm of x to base b.

        b may be omitted, in which case the default 10.

        Args:
            x (Self | float | None): `DOUBLE` expression

        Returns:
            Self
        """
        return self._new(func("log", x, self._expr))

    def implode(self) -> Self:
        """Returns a LIST containing all the values of a column.

        See Also:
            array_agg

        Returns:
            Self
        """
        return self._new(func("list", self._expr))

    def __str__(self) -> str:
        return str(self._expr)

    def __add__(self, other: Self) -> Self:
        return self._new(self._expr.__add__(other._expr))

    def __and__(self, other: Self) -> Self:
        return self._new(self._expr.__and__(other._expr))

    def and_(self, other: Self) -> Self:
        return self.__and__(other)

    def __div__(self, other: Self) -> Self:
        return self._new(self._expr.__truediv__(other._expr))

    def div(self, other: Self) -> Self:
        return self.__div__(other)

    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        return self._new(self._expr.__eq__(other._expr))

    def eq(self, other: Self) -> Self:
        return self.__eq__(other)

    def __floordiv__(self, other: Self) -> Self:
        return self._new(self._expr.__floordiv__(other._expr))

    def floordiv(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __ge__(self, other: Self) -> Self:
        return self._new(self._expr.__ge__(other._expr))

    def ge(self, other: Self) -> Self:
        return self.__ge__(other)

    def __gt__(self, other: Self) -> Self:
        return self._new(self._expr.__gt__(other._expr))

    def gt(self, other: Self) -> Self:
        return self.__gt__(other)

    def __invert__(self) -> Self:
        return self._new(self._expr.__invert__())

    def invert(self) -> Self:
        return self.__invert__()

    def __le__(self, other: Self) -> Self:
        return self._new(self._expr.__le__(other._expr))

    def le(self, other: Self) -> Self:
        return self.__le__(other)

    def __lt__(self, other: Self) -> Self:
        return self._new(self._expr.__lt__(other._expr))

    def lt(self, other: Self) -> Self:
        return self.__lt__(other)

    def __mod__(self, other: Self) -> Self:
        return self._new(self._expr.__mod__(other._expr))

    def __mul__(self, other: Self) -> Self:
        return self._new(self._expr.__mul__(other._expr))

    def mul(self, other: Self) -> Self:
        return self.__mul__(other)

    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        return self._new(self._expr.__ne__(other._expr))

    def ne(self, other: Self) -> Self:
        return self.__ne__(other)

    def __neg__(self) -> Self:
        return self._new(self._expr.__neg__())

    def neg(self) -> Self:
        return self.__neg__()

    def __or__(self, other: Self) -> Self:
        return self._new(self._expr.__or__(other._expr))

    def or_(self, other: Self) -> Self:
        return self.__or__(other)

    def __pow__(self, other: Self) -> Self:
        return self._new(self._expr.__pow__(other._expr))

    def __radd__(self, other: Self) -> Self:
        return self._new(self._expr.__radd__(other._expr))

    def radd(self, other: Self) -> Self:
        return self.__radd__(other)

    def __rand__(self, other: Self) -> Self:
        return self._new(self._expr.__rand__(other._expr))

    def rand(self, other: Self) -> Self:
        return self.__rand__(other)

    def __rdiv__(self, other: Self) -> Self:
        return self._new(self._expr.__rtruediv__(other._expr))

    def rdiv(self, other: Self) -> Self:
        return self.__rdiv__(other)

    def __rfloordiv__(self, other: Self) -> Self:
        return self._new(self._expr.__rfloordiv__(other._expr))

    def rfloordiv(self, other: Self) -> Self:
        return self.__rfloordiv__(other)

    def __rmod__(self, other: Self) -> Self:
        return self._new(self._expr.__rmod__(other._expr))

    def rmod(self, other: Self) -> Self:
        return self.__rmod__(other)

    def __rmul__(self, other: Self) -> Self:
        return self._new(self._expr.__rmul__(other._expr))

    def rmul(self, other: Self) -> Self:
        return self.__rmul__(other)

    def __ror__(self, other: Self) -> Self:
        return self._new(self._expr.__ror__(other._expr))

    def ror(self, other: Self) -> Self:
        return self.__ror__(other)

    def __rpow__(self, other: Self) -> Self:
        return self._new(self._expr.__rpow__(other._expr))

    def rpow(self, other: Self) -> Self:
        return self.__rpow__(other)

    def __rsub__(self, other: Self) -> Self:
        return self._new(self._expr.__rsub__(other._expr))

    def rsub(self, other: Self) -> Self:
        return self.__rsub__(other)

    def __rtruediv__(self, other: Self) -> Self:
        return self._new(self._expr.__rtruediv__(other._expr))

    def rtruediv(self, other: Self) -> Self:
        return self.__rtruediv__(other)

    def __sub__(self, other: Self) -> Self:
        return self._new(self._expr.__sub__(other._expr))

    def sub(self, other: Self) -> Self:
        return self.__sub__(other)

    def __truediv__(self, other: Self) -> Self:
        return self._new(self._expr.__truediv__(other._expr))

    def truediv(self, other: Self) -> Self:
        return self.__truediv__(other)

    def alias(self, name: str) -> Self:
        return self._new(self._expr.alias(name))

    def asc(self) -> Self:
        return self._new(self._expr.asc())

    def between(self, lower: Self, upper: Self) -> Self:
        return self._new(self._expr.between(lower._expr, upper._expr))

    def cast(self, dtype: DataType) -> Self:
        return self._new(self._expr.cast(dtype))

    def collate(self, collation: str) -> Self:
        return self._new(self._expr.collate(collation))

    def desc(self) -> Self:
        return self._new(self._expr.desc())

    def get_name(self) -> str:
        return self._expr.get_name()

    def is_in(self, *args: Self) -> Self:
        return self._new(self._expr.isin(*(arg._expr for arg in args)))

    def is_not_in(self, *args: Self) -> Self:
        return self._new(self._expr.isnotin(*(arg._expr for arg in args)))

    def is_not_null(self) -> Self:
        return self._new(self._expr.isnotnull())

    def is_null(self) -> Self:
        return self._new(self._expr.isnull())

    def nulls_first(self) -> Self:
        return self._new(self._expr.nulls_first())

    def nulls_last(self) -> Self:
        return self._new(self._expr.nulls_last())

    def otherwise(self, value: Self) -> Self:
        return self._new(self._expr.otherwise(value._expr))

    def show(self) -> None:
        self._expr.show()

    def when(self, condition: Self, value: Self) -> Self:
        return self._new(self._expr.when(condition._expr, value._expr))

    def over(  # noqa: PLR0913
        self,
        partition_by: pc.Seq[SqlExpr] | None = None,
        order_by: pc.Seq[SqlExpr] | None = None,
        rows_start: int | None = None,
        rows_end: int | None = None,
        *,
        descending: pc.Seq[bool] | bool = False,
        nulls_last: pc.Seq[bool] | bool = False,
        ignore_nulls: bool = False,
    ) -> Self:
        return self._new(
            over(
                self,
                partition_by,
                order_by,
                rows_start,
                rows_end,
                descending=descending,
                nulls_last=nulls_last,
                ignore_nulls=ignore_nulls,
            )
        )

    def cume_dist(self) -> Self:
        """The cumulative distribution: (number of partition rows preceding or peer with current row) / total partition rows.

        If an ORDER BY clause is specified, the distribution is computed within the frame using the provided ordering instead of the frame ordering.

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

        Both values must support arithmetic and there must be only one ordering key. For missing values at the ends, linear extrapolation is used. Failure to interpolate results in the NULL value being retained.

        Returns:
            Self
        """
        return self._new(func("fill", self._expr))

    def first_value(self) -> Self:
        """Returns expr evaluated at the row that is the first row (with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an ORDER BY clause is specified, the first row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("first_value", self._expr))

    def lag(self, offset: SqlExpr | int = 1, default: SqlExpr | None = None) -> Self:
        """Returns expr evaluated at the row that is offset rows (among rows with a non-null value of expr if IGNORE NULLS is set) before the current row within the window frame; if there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL. If an ORDER BY clause is specified, the lagged row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (SqlExpr | int): Number of rows to look back (default: 1)
            default (SqlExpr | None): Default value if no such row exists (default: NULL)

        Returns:
            Self
        """
        return self._new(func("lag", self._expr, offset, default))

    def last_value(self) -> Self:
        """Returns expr evaluated at the row that is the last row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an ORDER BY clause is specified, the last row is determined within the frame using the provided ordering instead of the frame ordering.


        Returns:
            Self
        """
        return self._new(func("last_value", self._expr))

    def lead(self, offset: SqlExpr | int = 1, default: SqlExpr | None = None) -> Self:
        """Returns expr evaluated at the row that is offset rows after the current row (among rows with a non-null value of expr if IGNORE NULLS is set) within the window frame; if there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL. If an ORDER BY clause is specified, the leading row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (SqlExpr | int): Number of rows to look ahead (default: 1)
            default (SqlExpr | None): Default value if no such row exists (default: NULL)

        Returns:
            Self
        """
        return self._new(func("lead", self._expr, offset, default))

    def nth_value(self, nth: SqlExpr | int) -> Self:
        """Returns expr evaluated at the nth row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame (counting from 1); NULL if no such row.

        If an ORDER BY clause is specified, the nth row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            nth (SqlExpr | int): The row number to retrieve (1-based)

        Returns:
            Self
        """
        return self._new(func("nth_value", self._expr, nth))

    def ntile(self) -> Self:
        """An integer ranging from 1 to num_buckets, dividing the partition as equally as possible.

        If an ORDER BY clause is specified, the ntile is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            num_buckets (SqlExpr | int): Number of buckets to divide into

        Returns:
            Self
        """
        return self._new(func("ntile", self._expr))

    def percent_rank(self) -> Self:
        """The relative rank of the current row: (rank() - 1) / (total partition rows - 1).

        If an ORDER BY clause is specified, the relative rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("percent_rank"))

    def rank(self) -> Self:
        """The rank of the current row with gaps; same as row_number of its first peer.

        If an ORDER BY clause is specified, the rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("rank"))

    def row_number(self) -> Self:
        """The number of the current row within the partition, counting from 1.

        If an ORDER BY clause is specified, the row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._new(func("row_number"))


@dataclass(slots=True)
class SqlExprStringNameSpace(StringFns[SqlExpr]):
    """String function namespace for SQL expressions."""


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
