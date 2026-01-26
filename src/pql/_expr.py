"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import duckdb
import pyochain as pc

from . import datatypes

if TYPE_CHECKING:
    from collections.abc import Iterable


def to_expr(value: object) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become columns for select/group_by)."""
    match value:
        case Expr():
            return value.expr
        case str():
            return duckdb.ColumnExpression(value)
        case _:
            return duckdb.ConstantExpression(value)


def _to_value(value: object) -> duckdb.Expression:
    """Convert a value to a DuckDB Expression (strings become constants for comparisons)."""
    match value:
        case Expr():
            return value.expr
        case _:
            return duckdb.ConstantExpression(value)


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(duckdb.ColumnExpression(name))

    def __getattr__(self, name: str) -> Expr:
        return self.__call__(name)


col: Col = Col()


def lit(value: object) -> Expr:
    """Create a literal expression (equivalent to pl.lit)."""
    return Expr(duckdb.ConstantExpression(value))


class Expr:
    """Expression wrapper providing Polars-like API over DuckDB expressions."""

    __slots__ = ("_expr",)

    def __init__(self, expr: duckdb.Expression) -> None:
        self._expr = expr

    def __repr__(self) -> str:
        return f"Expr({self._expr})"

    @property
    def expr(self) -> duckdb.Expression:
        """Get the underlying DuckDB expression."""
        return self._expr

    def add(self, other: object) -> Self:
        """Add another expression or value."""
        return self.__class__(self._expr + _to_value(other))

    def radd(self, other: object) -> Self:
        return self.__class__(_to_value(other) + self._expr)

    def sub(self, other: object) -> Self:
        return self.__class__(self._expr - _to_value(other))

    def rsub(self, other: object) -> Self:
        return self.__class__(_to_value(other) - self._expr)

    def mul(self, other: object) -> Self:
        return self.__class__(self._expr * _to_value(other))

    def rmul(self, other: object) -> Self:
        return self.__class__(_to_value(other) * self._expr)

    def truediv(self, other: object) -> Self:
        return self.__class__(self._expr / _to_value(other))

    def rtruediv(self, other: object) -> Self:
        return self.__class__(_to_value(other) / self._expr)

    def floordiv(self, other: object) -> Self:
        return self.__class__(self._expr // _to_value(other))

    def rfloordiv(self, other: object) -> Self:
        return self.__class__(_to_value(other) // self._expr)

    def mod(self, other: object) -> Self:
        return self.__class__(self._expr % _to_value(other))

    def rmod(self, other: object) -> Self:
        return self.__class__(_to_value(other) % self._expr)

    def pow(self, other: object) -> Self:
        return self.__class__(self._expr ** _to_value(other))

    def rpow(self, other: object) -> Self:
        return self.__class__(_to_value(other) ** self._expr)

    def neg(self) -> Self:
        return self.__class__(-self._expr)

    def pos(self) -> Self:
        return self

    def abs(self) -> Self:
        return self.__class__(duckdb.FunctionExpression("abs", self._expr))

    def eq(self, other: object) -> Self:
        return self.__class__(self._expr == _to_value(other))

    def ne(self, other: object) -> Self:
        return self.__class__(self._expr != _to_value(other))

    def lt(self, other: object) -> Self:
        return self.__class__(self._expr < _to_value(other))

    def le(self, other: object) -> Self:
        return self.__class__(self._expr <= _to_value(other))

    def gt(self, other: object) -> Self:
        return self.__class__(self._expr > _to_value(other))

    def ge(self, other: object) -> Self:
        return self.__class__(self._expr >= _to_value(other))

    def and_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr & _to_value(others))

    def rand(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(_to_value(others) & self._expr)

    def or_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr | _to_value(others))

    def ror(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(_to_value(others) | self._expr)

    def not_(self) -> Self:
        return self.__class__(~self._expr)

    def alias(self, name: str) -> Self:
        """Rename the expression."""
        return self.__class__(self._expr.alias(name))

    def is_null(self) -> Self:
        """Check if the expression is NULL."""
        return self.__class__(self._expr.isnull())

    def is_not_null(self) -> Self:
        """Check if the expression is not NULL."""
        return self.__class__(self._expr.isnotnull())

    def fill_null(self, value: object) -> Self:
        """Fill NULL values with the given value."""
        return self.__class__(duckdb.CoalesceOperator(self._expr, to_expr(value)))

    def cast(self, dtype: datatypes.DataType) -> Self:
        """Cast to a different data type."""
        return self.__class__(self._expr.cast(dtype))

    def between(self, lower: object, upper: object) -> Self:
        """Check if value is between lower and upper (inclusive)."""
        return self.__class__(self._expr.between(_to_value(lower), _to_value(upper)))

    def is_in(self, values: Iterable[object]) -> Self:
        """Check if value is in an iterable of values."""
        return self.__class__(self._expr.isin(*pc.Iter(values).map(_to_value)))

    def is_not_in(self, values: Iterable[object]) -> Self:
        """Check if value is not in an iterable of values."""
        return self.__class__(~self._expr.isin(*pc.Iter(values).map(_to_value)))

    def sum(self) -> Self:
        """Compute the sum."""
        return self.__class__(duckdb.FunctionExpression("sum", self._expr))

    def mean(self) -> Self:
        """Compute the mean (average)."""
        return self.__class__(duckdb.FunctionExpression("avg", self._expr))

    def min(self) -> Self:
        """Compute the minimum."""
        return self.__class__(duckdb.FunctionExpression("min", self._expr))

    def max(self) -> Self:
        """Compute the maximum."""
        return self.__class__(duckdb.FunctionExpression("max", self._expr))

    def count(self) -> Self:
        """Count non-null values."""
        return self.__class__(
            duckdb.FunctionExpression("count", self._expr).cast(datatypes.UInt32)
        )

    def std(self, ddof: int = 1) -> Self:
        """Compute the standard deviation."""
        func = "stddev_samp" if ddof == 1 else "stddev_pop"
        return self.__class__(duckdb.FunctionExpression(func, self._expr))

    def var(self, ddof: int = 1) -> Self:
        """Compute the variance."""
        func = "var_samp" if ddof == 1 else "var_pop"
        return self.__class__(duckdb.FunctionExpression(func, self._expr))

    def first(self) -> Self:
        """Get the first value."""
        return self.__class__(duckdb.FunctionExpression("first", self._expr))

    def last(self) -> Self:
        """Get the last value."""
        return self.__class__(duckdb.FunctionExpression("last", self._expr))

    def n_unique(self) -> Self:
        """Count unique values."""
        return self.__class__(
            duckdb.SQLExpression(f"COUNT(DISTINCT {self._expr})").cast(datatypes.UInt32)
        )

    @property
    def str(self) -> ExprStringNameSpace:
        """Access string operations."""
        return ExprStringNameSpace(self._expr)

    @property
    def dt(self) -> ExprDateTimeNameSpace:
        """Access datetime operations."""
        return ExprDateTimeNameSpace(self._expr)


class ExprStringNameSpace:
    """String operations namespace (equivalent to pl.Expr.str)."""

    __slots__ = ("_expr",)

    def __init__(self, expr: duckdb.Expression) -> None:
        self._expr = expr

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return Expr(duckdb.FunctionExpression("upper", self._expr))

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return Expr(duckdb.FunctionExpression("lower", self._expr))

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return Expr(duckdb.FunctionExpression("length", self._expr))

    def contains(self, pattern: str, *, literal: bool = True) -> Expr:
        """Check if string contains a pattern."""
        return (
            Expr(
                duckdb.FunctionExpression(
                    "contains", self._expr, duckdb.ConstantExpression(pattern)
                )
            )
            if literal
            else Expr(
                duckdb.FunctionExpression(
                    "regexp_matches", self._expr, duckdb.ConstantExpression(pattern)
                )
            )
        )

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return Expr(
            duckdb.FunctionExpression(
                "starts_with", self._expr, duckdb.ConstantExpression(prefix)
            )
        )

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return Expr(
            duckdb.FunctionExpression(
                "ends_with", self._expr, duckdb.ConstantExpression(suffix)
            )
        )

    def replace(self, pattern: str, replacement: str) -> Expr:
        """Replace occurrences of pattern with replacement."""
        return Expr(
            duckdb.FunctionExpression(
                "replace",
                self._expr,
                duckdb.ConstantExpression(pattern),
                duckdb.ConstantExpression(replacement),
            )
        )

    def strip_chars(self) -> Expr:
        """Strip leading and trailing whitespace."""
        return Expr(duckdb.FunctionExpression("trim", self._expr))

    def strip_chars_start(self) -> Expr:
        """Strip leading whitespace."""
        return Expr(duckdb.FunctionExpression("ltrim", self._expr))

    def strip_chars_end(self) -> Expr:
        """Strip trailing whitespace."""
        return Expr(duckdb.FunctionExpression("rtrim", self._expr))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        args: tuple[duckdb.Expression, ...] = (
            (self._expr, duckdb.ConstantExpression(str(offset + 1)))
            if length is None
            else (
                self._expr,
                duckdb.ConstantExpression(str(offset + 1)),
                duckdb.ConstantExpression(str(length)),
            )
        )
        return Expr(duckdb.FunctionExpression("substring", *args))


class ExprDateTimeNameSpace:
    """Datetime operations namespace (equivalent to pl.Expr.dt)."""

    __slots__ = ("_expr",)

    def __init__(self, expr: duckdb.Expression) -> None:
        self._expr = expr

    def _to_utc(self) -> duckdb.Expression:
        """Convert datetime to UTC timezone."""
        return duckdb.FunctionExpression(
            "timezone", duckdb.ConstantExpression("UTC"), self._expr
        )

    def year(self) -> Expr:
        """Extract the year."""
        return Expr(
            duckdb.FunctionExpression("year", self._to_utc()).cast(datatypes.Int32)
        )

    def month(self) -> Expr:
        """Extract the month."""
        return Expr(
            duckdb.FunctionExpression("month", self._to_utc()).cast(datatypes.Int8)
        )

    def day(self) -> Expr:
        """Extract the day."""
        return Expr(
            duckdb.FunctionExpression("day", self._to_utc()).cast(datatypes.Int8)
        )

    def hour(self) -> Expr:
        """Extract the hour."""
        return Expr(
            duckdb.FunctionExpression("hour", self._to_utc()).cast(datatypes.Int8)
        )

    def minute(self) -> Expr:
        """Extract the minute."""
        return Expr(
            duckdb.FunctionExpression("minute", self._to_utc()).cast(datatypes.Int32)
        )

    def second(self) -> Expr:
        """Extract the second."""
        return Expr(
            duckdb.FunctionExpression("second", self._to_utc()).cast(datatypes.Int32)
        )

    def weekday(self) -> Expr:
        """Extract the day of week (1=Monday, 7=Sunday)."""
        return Expr(
            duckdb.FunctionExpression(
                "dayofweek", self._to_utc().cast(datatypes.Date)
            ).cast(datatypes.Int8)
        )

    def week(self) -> Expr:
        """Extract the week number."""
        return Expr(duckdb.FunctionExpression("week", self._expr).cast(datatypes.Int8))

    def date(self) -> Expr:
        """Extract the date part."""
        return Expr(self._to_utc().cast(datatypes.Date))

    def convert_time_zone(self, time_zone: str) -> Expr:
        """Convert to a different timezone."""
        return Expr(
            duckdb.FunctionExpression(
                "timezone", duckdb.ConstantExpression(time_zone), self._expr
            )
        )
