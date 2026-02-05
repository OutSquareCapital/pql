from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

import duckdb
import pyochain as pc

from ._core import func
from .fns import FnsMixin

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
    return SqlExpr(duckdb.LambdaExpression(lhs, rhs.to_duckdb()))


def all(*, exclude: Any | None = None) -> SqlExpr:  # noqa: ANN401
    return SqlExpr(duckdb.StarExpression(exclude=exclude))


def when(condition: SqlExpr, value: SqlExpr) -> SqlExpr:
    return SqlExpr(duckdb.CaseExpression(condition.to_duckdb(), value.to_duckdb()))


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
    return SqlExpr(duckdb.CoalesceOperator(*(expr.to_duckdb() for expr in exprs)))


def from_expr(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case duckdb.Expression():
            return SqlExpr(value)
        case Expr():
            return value.to_sql()
        case str():
            return col(value)
        case _:
            return lit(value)


def from_value(value: IntoExpr) -> SqlExpr:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case duckdb.Expression():
            return SqlExpr(value)
        case Expr():
            return value.to_sql()
        case _:
            return lit(value)


def to_duckdb(value: SqlExpr | str | duckdb.Expression) -> duckdb.Expression | str:
    """Convert a SqlExpr to duckdb.Expression, preserving str and duckdb.Expression."""
    match value:
        case str() | duckdb.Expression():
            return value
        case SqlExpr():
            return value.to_duckdb()


def from_cols(exprs: IntoExprColumn) -> pc.Iter[duckdb.Expression | str]:
    """Convert one or more values or iterables of values to an iterable of DuckDB Expressions or strings."""
    from .._expr import Expr

    match exprs:
        case duckdb.Expression():
            return pc.Iter.once(exprs)
        case Expr() | SqlExpr():
            return pc.Iter.once(exprs.to_duckdb())
        case str():
            return pc.Iter.once(exprs)
        case Iterable():
            return pc.Iter(exprs).map(from_cols).flatten()


def from_iter(*values: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
    """Convert one or more values or iterables of values to an iterator of DuckDB Expressions.

    Note:
        We handle this with an external variadic argument, and an internal closure, to
        distinguish between a single iterable argument and multiple arguments.
    """

    def _single_to_expr(value: IntoExpr | Iterable[IntoExpr]) -> pc.Iter[SqlExpr]:
        match value:
            case str() | bytes() | bytearray():
                return pc.Iter.once(from_expr(value))
            case Iterable():
                return pc.Iter(value).map(from_expr)
            case _:
                return pc.Iter.once(from_expr(value))

    match values:
        case (single,):
            return _single_to_expr(single)
        case _:
            return pc.Iter(values).map(_single_to_expr).flatten()


def from_args_kwargs(
    *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
) -> pc.Iter[SqlExpr]:
    """Convert positional and keyword arguments to an iterator of DuckDB Expressions."""
    return from_iter(*exprs).chain(
        pc.Dict.from_ref(named_exprs)
        .items()
        .iter()
        .map_star(lambda name, expr: from_expr(expr).alias(name))
    )


def build_over(
    expr: str, partition_by: str, order_by: str, row_between: str
) -> duckdb.Expression:
    return duckdb.SQLExpression(
        f"{expr} {Kword.OVER} ({partition_by} {order_by} {row_between})"
    )


class Kword(StrEnum):
    PARTITION_BY = "PARTITION BY"
    ORDER_BY = "ORDER BY"
    DESC = "DESC"
    ASC = "ASC"
    NULLS_LAST = "NULLS LAST"
    NULLS_FIRST = "NULLS FIRST"
    ROWS_BETWEEN = "ROWS BETWEEN"
    OVER = "OVER"

    @classmethod
    def sort_strat(cls, item: SqlExpr, *, desc: bool, nulls_last: bool) -> str:
        return f"{item} {cls.DESC if desc else cls.ASC} {cls.NULLS_LAST if nulls_last else cls.NULLS_FIRST}"

    @classmethod
    def rows_clause(cls, row_start: pc.Option[int], row_end: pc.Option[int]) -> str:
        match (row_start, row_end):
            case (pc.Some(start), pc.Some(end)):
                return f"{cls.ROWS_BETWEEN} {-start} PRECEDING AND {end} FOLLOWING"
            case (pc.Some(start), pc.NONE):
                return f"{cls.ROWS_BETWEEN} {-start} PRECEDING AND UNBOUNDED FOLLOWING"
            case (pc.NONE, pc.Some(end)):
                return f"{cls.ROWS_BETWEEN} UNBOUNDED PRECEDING AND {end} FOLLOWING"
            case _:
                return ""

    @classmethod
    def partition_by(cls, by: str) -> str:
        return f"{cls.PARTITION_BY} {by}"

    @classmethod
    def order_by(cls, by: str) -> str:
        return f"{cls.ORDER_BY} {by}"


def get_partition_by(partition_by: pc.Seq[SqlExpr]) -> str:
    return (
        partition_by.then_some()
        .map(lambda x: x.iter().map(str).join(", "))
        .map(Kword.partition_by)
        .unwrap_or("")
    )


def handle_nulls(expr: SqlExpr, *, ignore_nulls: bool) -> str:
    match ignore_nulls:
        case True:
            return f"{str(expr).removesuffix(')')} ignore nulls)"
        case False:
            return str(expr)


def get_order_by(
    order_by: pc.Seq[SqlExpr],
    *,
    descending: pc.Seq[bool] | bool,
    nulls_last: pc.Seq[bool] | bool,
) -> str:
    def _get_clauses(*, clauses: pc.Seq[bool] | bool) -> pc.Seq[bool]:
        match clauses:
            case bool() as val:
                return pc.Iter.once(val).cycle().take(order_by.length()).collect()
            case pc.Seq() as seq:
                return seq

    return (
        order_by.then_some()
        .map(
            lambda x: (
                x.iter()
                .zip(_get_clauses(clauses=descending), _get_clauses(clauses=nulls_last))
                .map_star(
                    lambda item, desc, nl: Kword.sort_strat(
                        item=item, desc=desc, nulls_last=nl
                    )
                )
                .join(", ")
            )
        )
        .map(Kword.order_by)
        .unwrap_or("")
    )


class SqlExpr(FnsMixin):  # noqa: PLW1641
    """A wrapper around duckdb.Expression that provides operator overloading and SQL function methods."""

    __slots__ = ()

    def to_duckdb(self) -> duckdb.Expression:
        """Get the underlying DuckDB Expression."""
        return self._expr

    def __str__(self) -> str:
        return str(self._expr)

    def __add__(self, other: Self) -> Self:
        return self.__class__(self._expr.__add__(other._expr))

    def __and__(self, other: Self) -> Self:
        return self.__class__(self._expr.__and__(other._expr))

    def and_(self, other: Self) -> Self:
        return self.__and__(other)

    def __div__(self, other: Self) -> Self:
        return self.__class__(self._expr.__truediv__(other._expr))

    def div(self, other: Self) -> Self:
        return self.__div__(other)

    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        return self.__class__(self._expr.__eq__(other._expr))

    def eq(self, other: Self) -> Self:
        return self.__eq__(other)

    def __floordiv__(self, other: Self) -> Self:
        return self.__class__(self._expr.__floordiv__(other._expr))

    def floordiv(self, other: Self) -> Self:
        return self.__floordiv__(other)

    def __ge__(self, other: Self) -> Self:
        return self.__class__(self._expr.__ge__(other._expr))

    def ge(self, other: Self) -> Self:
        return self.__ge__(other)

    def __gt__(self, other: Self) -> Self:
        return self.__class__(self._expr.__gt__(other._expr))

    def gt(self, other: Self) -> Self:
        return self.__gt__(other)

    def __invert__(self) -> Self:
        return self.__class__(self._expr.__invert__())

    def invert(self) -> Self:
        return self.__invert__()

    def __le__(self, other: Self) -> Self:
        return self.__class__(self._expr.__le__(other._expr))

    def le(self, other: Self) -> Self:
        return self.__le__(other)

    def __lt__(self, other: Self) -> Self:
        return self.__class__(self._expr.__lt__(other._expr))

    def lt(self, other: Self) -> Self:
        return self.__lt__(other)

    def __mod__(self, other: Self) -> Self:
        return self.__class__(self._expr.__mod__(other._expr))

    def __mul__(self, other: Self) -> Self:
        return self.__class__(self._expr.__mul__(other._expr))

    def mul(self, other: Self) -> Self:
        return self.__mul__(other)

    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        return self.__class__(self._expr.__ne__(other._expr))

    def ne(self, other: Self) -> Self:
        return self.__ne__(other)

    def __neg__(self) -> Self:
        return self.__class__(self._expr.__neg__())

    def neg(self) -> Self:
        return self.__neg__()

    def __or__(self, other: Self) -> Self:
        return self.__class__(self._expr.__or__(other._expr))

    def or_(self, other: Self) -> Self:
        return self.__or__(other)

    def __pow__(self, other: Self) -> Self:
        return self.__class__(self._expr.__pow__(other._expr))

    def __radd__(self, other: Self) -> Self:
        return self.__class__(self._expr.__radd__(other._expr))

    def radd(self, other: Self) -> Self:
        return self.__radd__(other)

    def __rand__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rand__(other._expr))

    def rand(self, other: Self) -> Self:
        return self.__rand__(other)

    def __rdiv__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rtruediv__(other._expr))

    def rdiv(self, other: Self) -> Self:
        return self.__rdiv__(other)

    def __rfloordiv__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rfloordiv__(other._expr))

    def rfloordiv(self, other: Self) -> Self:
        return self.__rfloordiv__(other)

    def __rmod__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rmod__(other._expr))

    def rmod(self, other: Self) -> Self:
        return self.__rmod__(other)

    def __rmul__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rmul__(other._expr))

    def rmul(self, other: Self) -> Self:
        return self.__rmul__(other)

    def __ror__(self, other: Self) -> Self:
        return self.__class__(self._expr.__ror__(other._expr))

    def ror(self, other: Self) -> Self:
        return self.__ror__(other)

    def __rpow__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rpow__(other._expr))

    def rpow(self, other: Self) -> Self:
        return self.__rpow__(other)

    def __rsub__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rsub__(other._expr))

    def rsub(self, other: Self) -> Self:
        return self.__rsub__(other)

    def __rtruediv__(self, other: Self) -> Self:
        return self.__class__(self._expr.__rtruediv__(other._expr))

    def rtruediv(self, other: Self) -> Self:
        return self.__rtruediv__(other)

    def __sub__(self, other: Self) -> Self:
        return self.__class__(self._expr.__sub__(other._expr))

    def sub(self, other: Self) -> Self:
        return self.__sub__(other)

    def __truediv__(self, other: Self) -> Self:
        return self.__class__(self._expr.__truediv__(other._expr))

    def truediv(self, other: Self) -> Self:
        return self.__truediv__(other)

    def alias(self, name: str) -> Self:
        return self.__class__(self._expr.alias(name))

    def asc(self) -> Self:
        return self.__class__(self._expr.asc())

    def between(self, lower: Self, upper: Self) -> Self:
        return self.__class__(self._expr.between(lower._expr, upper._expr))

    def cast(self, dtype: DataType) -> Self:
        return self.__class__(self._expr.cast(dtype))

    def collate(self, collation: str) -> Self:
        return self.__class__(self._expr.collate(collation))

    def desc(self) -> Self:
        return self.__class__(self._expr.desc())

    def get_name(self) -> str:
        return self._expr.get_name()

    def is_in(self, *args: Self) -> Self:
        return self.__class__(self._expr.isin(*(arg._expr for arg in args)))

    def is_not_in(self, *args: Self) -> Self:
        return self.__class__(self._expr.isnotin(*(arg._expr for arg in args)))

    def is_not_null(self) -> Self:
        return self.__class__(self._expr.isnotnull())

    def is_null(self) -> Self:
        return self.__class__(self._expr.isnull())

    def nulls_first(self) -> Self:
        return self.__class__(self._expr.nulls_first())

    def nulls_last(self) -> Self:
        return self.__class__(self._expr.nulls_last())

    def otherwise(self, value: Self) -> Self:
        return self.__class__(self._expr.otherwise(value._expr))

    def show(self) -> None:
        self._expr.show()

    def when(self, condition: Self, value: Self) -> Self:
        return self.__class__(self._expr.when(condition._expr, value._expr))

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
        return self.__class__(
            build_over(
                handle_nulls(self, ignore_nulls=ignore_nulls),
                get_partition_by(partition_by or pc.Seq[SqlExpr].new()),
                get_order_by(
                    order_by or pc.Seq[SqlExpr].new(),
                    descending=descending,
                    nulls_last=nulls_last,
                ),
                Kword.rows_clause(pc.Option(rows_start), pc.Option(rows_end)),
            )
        )

    def cume_dist(self) -> Self:
        """The cumulative distribution: (number of partition rows preceding or peer with current row) / total partition rows.

        If an ORDER BY clause is specified, the distribution is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self.__class__(func("cume_dist"))

    def dense_rank(self) -> Self:
        """The rank of the current row without gaps; this function counts peer groups.

        Returns:
            Self
        """
        return self.__class__(func("dense_rank"))

    def fill(self) -> Self:
        """Replaces NULL values of expr with a linear interpolation based on the closest non-NULL values and the sort values.

        Both values must support arithmetic and there must be only one ordering key. For missing values at the ends, linear extrapolation is used. Failure to interpolate results in the NULL value being retained.

        Returns:
            Self
        """
        return self.__class__(func("fill", self._expr))

    def first_value(self) -> Self:
        """Returns expr evaluated at the row that is the first row (with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an ORDER BY clause is specified, the first row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self.__class__(func("first_value", self._expr))

    def lag(self, offset: SqlExpr | int = 1, default: SqlExpr | None = None) -> Self:
        """Returns expr evaluated at the row that is offset rows (among rows with a non-null value of expr if IGNORE NULLS is set) before the current row within the window frame; if there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL. If an ORDER BY clause is specified, the lagged row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (SqlExpr | int): Number of rows to look back (default: 1)
            default (SqlExpr | None): Default value if no such row exists (default: NULL)

        Returns:
            Self
        """
        return self.__class__(func("lag", self._expr, offset, default))

    def last_value(self) -> Self:
        """Returns expr evaluated at the row that is the last row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame.

        If an ORDER BY clause is specified, the last row is determined within the frame using the provided ordering instead of the frame ordering.


        Returns:
            Self
        """
        return self.__class__(func("last_value", self._expr))

    def lead(self, offset: SqlExpr | int = 1, default: SqlExpr | None = None) -> Self:
        """Returns expr evaluated at the row that is offset rows after the current row (among rows with a non-null value of expr if IGNORE NULLS is set) within the window frame; if there is no such row, instead return default (which must be of the Same type as expr).

        Both offset and default are evaluated with respect to the current row. If omitted, offset defaults to 1 and default to NULL. If an ORDER BY clause is specified, the leading row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            offset (SqlExpr | int): Number of rows to look ahead (default: 1)
            default (SqlExpr | None): Default value if no such row exists (default: NULL)

        Returns:
            Self
        """
        return self.__class__(func("lead", self._expr, offset, default))

    def nth_value(self, nth: SqlExpr | int) -> Self:
        """Returns expr evaluated at the nth row (among rows with a non-null value of expr if IGNORE NULLS is set) of the window frame (counting from 1); NULL if no such row.

        If an ORDER BY clause is specified, the nth row number is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            nth (SqlExpr | int): The row number to retrieve (1-based)

        Returns:
            Self
        """
        return self.__class__(func("nth_value", self._expr, nth))

    def ntile(self) -> Self:
        """An integer ranging from 1 to num_buckets, dividing the partition as equally as possible.

        If an ORDER BY clause is specified, the ntile is computed within the frame using the provided ordering instead of the frame ordering.

        Args:
            num_buckets (SqlExpr | int): Number of buckets to divide into

        Returns:
            Self
        """
        return self.__class__(func("ntile", self._expr))

    def percent_rank(self) -> Self:
        """The relative rank of the current row: (rank() - 1) / (total partition rows - 1).

        If an ORDER BY clause is specified, the relative rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self.__class__(func("percent_rank"))

    def rank(self) -> Self:
        """The rank of the current row with gaps; same as row_number of its first peer.

        If an ORDER BY clause is specified, the rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self.__class__(func("rank"))

    def row_number(self) -> Self:
        """The number of the current row within the partition, counting from 1.

        If an ORDER BY clause is specified, the row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self.__class__(func("row_number"))
