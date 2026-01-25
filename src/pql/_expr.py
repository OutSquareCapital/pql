from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pyochain as pc
from sqlglot import exp

if TYPE_CHECKING:
    from collections.abc import Iterable


def to_node(value: object) -> exp.Expression:
    """Convert a value to a sqlglot Expression node."""
    match value:
        case Expr():
            return value.__node__
        case str():
            return exp.column(value)
        case _:
            return exp.convert(value)


def exprs_to_nodes(exprs: Iterable[Expr | str]) -> pc.Iter[exp.Expression]:
    """Convert multiple expressions to sqlglot nodes."""
    return pc.Iter(exprs).map(to_node)


def col(name: str) -> Expr:
    """Create a column expression (equivalent to pl.col)."""
    return Expr(exp.column(name))


def lit(value: object) -> Expr:
    """Create a literal expression (equivalent to pl.lit)."""
    return Expr(exp.convert(value))


class Expr:
    """Expression wrapper providing Polars-like API over sqlglot."""

    __slots__ = ("__node__",)

    def __init__(self, node: exp.Expression) -> None:
        self.__node__ = node

    def __repr__(self) -> str:
        return f"Expr({self.__node__.sql(dialect='duckdb')})"

    # ==================== Arithmetic Operators ====================

    def __add__(self, other: object) -> Self:
        return self.__class__(exp.Add(this=self.__node__, expression=to_node(other)))

    def __radd__(self, other: object) -> Self:
        return self.__class__(exp.Add(this=to_node(other), expression=self.__node__))

    def __sub__(self, other: object) -> Self:
        return self.__class__(exp.Sub(this=self.__node__, expression=to_node(other)))

    def __rsub__(self, other: object) -> Self:
        return self.__class__(exp.Sub(this=to_node(other), expression=self.__node__))

    def __mul__(self, other: object) -> Self:
        return self.__class__(exp.Mul(this=self.__node__, expression=to_node(other)))

    def __rmul__(self, other: object) -> Self:
        return self.__class__(exp.Mul(this=to_node(other), expression=self.__node__))

    def __truediv__(self, other: object) -> Self:
        return self.__class__(exp.Div(this=self.__node__, expression=to_node(other)))

    def __rtruediv__(self, other: object) -> Self:
        return self.__class__(exp.Div(this=to_node(other), expression=self.__node__))

    def __floordiv__(self, other: object) -> Self:
        return self.__class__(exp.IntDiv(this=self.__node__, expression=to_node(other)))

    def __rfloordiv__(self, other: object) -> Self:
        return self.__class__(exp.IntDiv(this=to_node(other), expression=self.__node__))

    def __mod__(self, other: object) -> Self:
        return self.__class__(exp.Mod(this=self.__node__, expression=to_node(other)))

    def __rmod__(self, other: object) -> Self:
        return self.__class__(exp.Mod(this=to_node(other), expression=self.__node__))

    def __pow__(self, other: object) -> Self:
        return self.__class__(exp.Pow(this=self.__node__, expression=to_node(other)))

    def __rpow__(self, other: object) -> Self:
        return self.__class__(exp.Pow(this=to_node(other), expression=self.__node__))

    def __neg__(self) -> Self:
        return self.__class__(exp.Neg(this=self.__node__))

    def __pos__(self) -> Self:
        return self

    def __abs__(self) -> Self:
        return self.__class__(exp.Abs(this=self.__node__))

    # ==================== Comparison Operators ====================

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self.__class__(exp.EQ(this=self.__node__, expression=to_node(other)))

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self.__class__(exp.NEQ(this=self.__node__, expression=to_node(other)))

    def __lt__(self, other: object) -> Self:
        return self.__class__(exp.LT(this=self.__node__, expression=to_node(other)))

    def __le__(self, other: object) -> Self:
        return self.__class__(exp.LTE(this=self.__node__, expression=to_node(other)))

    def __gt__(self, other: object) -> Self:
        return self.__class__(exp.GT(this=self.__node__, expression=to_node(other)))

    def __ge__(self, other: object) -> Self:
        return self.__class__(exp.GTE(this=self.__node__, expression=to_node(other)))

    # ==================== Logical Operators ====================

    def __and__(self, other: object) -> Self:
        return self.__class__(exp.And(this=self.__node__, expression=to_node(other)))

    def __rand__(self, other: object) -> Self:
        return self.__class__(exp.And(this=to_node(other), expression=self.__node__))

    def __or__(self, other: object) -> Self:
        return self.__class__(exp.Or(this=self.__node__, expression=to_node(other)))

    def __ror__(self, other: object) -> Self:
        return self.__class__(exp.Or(this=to_node(other), expression=self.__node__))

    def __invert__(self) -> Self:
        return self.__class__(exp.Not(this=self.__node__))

    # ==================== Naming ====================

    def alias(self, name: str) -> Self:
        """Rename the expression."""
        return self.__class__(exp.alias_(self.__node__, name))

    # ==================== Null Handling ====================

    def is_null(self) -> Self:
        """Check if the expression is NULL."""
        return self.__class__(exp.Is(this=self.__node__, expression=exp.Null()))

    def is_not_null(self) -> Self:
        """Check if the expression is not NULL."""
        return self.__class__(
            exp.Not(this=exp.Is(this=self.__node__, expression=exp.Null()))
        )

    def fill_null(self, value: object) -> Self:
        """Fill NULL values with the given value."""
        return self.__class__(
            exp.Coalesce(this=self.__node__, expressions=[to_node(value)])
        )

    # ==================== Aggregations ====================

    def sum(self) -> Self:
        """Compute the sum."""
        return self.__class__(exp.Sum(this=self.__node__))

    def mean(self) -> Self:
        """Compute the mean (average)."""
        return self.__class__(exp.Avg(this=self.__node__))

    def min(self) -> Self:
        """Compute the minimum."""
        return self.__class__(exp.Min(this=self.__node__))

    def max(self) -> Self:
        """Compute the maximum."""
        return self.__class__(exp.Max(this=self.__node__))

    def count(self) -> Self:
        """Count non-null values."""
        return self.__class__(exp.Count(this=self.__node__))

    def std(self, ddof: int = 1) -> Self:
        """Compute the standard deviation."""
        return self.__class__(
            exp.Stddev(this=self.__node__)
            if ddof == 1
            else exp.StddevPop(this=self.__node__)
        )

    def var(self, ddof: int = 1) -> Self:
        """Compute the variance."""
        return self.__class__(
            exp.Variance(this=self.__node__)
            if ddof == 1
            else exp.VariancePop(this=self.__node__)
        )

    def first(self) -> Self:
        """Get the first value."""
        return self.__class__(exp.First(this=self.__node__))

    def last(self) -> Self:
        """Get the last value."""
        return self.__class__(exp.Last(this=self.__node__))

    def n_unique(self) -> Self:
        """Count unique values."""
        return self.__class__(exp.Count(this=exp.Distinct(expressions=[self.__node__])))

    @property
    def str(self) -> _StringNamespace:
        """Access string operations."""
        return _StringNamespace(self.__node__)

    @property
    def dt(self) -> _DatetimeNamespace:
        """Access datetime operations."""
        return _DatetimeNamespace(self.__node__)

    def cast(self, dtype: str) -> Self:
        """Cast to a different data type."""
        return self.__class__(
            exp.Cast(this=self.__node__, to=exp.DataType.build(dtype))
        )

    def between(self, lower: object, upper: object) -> Self:
        """Check if value is between lower and upper (inclusive)."""
        return self.__class__(
            exp.Between(this=self.__node__, low=to_node(lower), high=to_node(upper))
        )

    def is_in(self, values: Iterable[object]) -> Self:
        """Check if value is in a list of values."""
        return self.__class__(
            exp.In(this=self.__node__, expressions=exprs_to_nodes(values))
        )

    def is_not_in(self, values: Iterable[object]) -> Self:
        """Check if value is not in a list of values."""
        return self.__class__(
            exp.Not(this=exp.In(this=self.__node__, expressions=exprs_to_nodes(values)))
        )


class _StringNamespace:
    """String operations namespace (equivalent to pl.Expr.str)."""

    __slots__ = ("__node__",)

    def __init__(self, node: exp.Expression) -> None:
        self.__node__ = node

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return Expr(exp.Upper(this=self.__node__))

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return Expr(exp.Lower(this=self.__node__))

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return Expr(exp.Length(this=self.__node__))

    def contains(self, pattern: str, *, literal: bool = True) -> Expr:
        """Check if string contains a pattern."""
        return (
            Expr(
                exp.Like(
                    this=self.__node__, expression=exp.Literal.string(f"%{pattern}%")
                )
            )
            if literal
            else Expr(
                exp.RegexpLike(
                    this=self.__node__, expression=exp.Literal.string(pattern)
                )
            )
        )

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return Expr(
            exp.Like(this=self.__node__, expression=exp.Literal.string(f"{prefix}%"))
        )

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return Expr(
            exp.Like(this=self.__node__, expression=exp.Literal.string(f"%{suffix}"))
        )

    def replace(self, pattern: str, replacement: str) -> Expr:
        """Replace occurrences of pattern with replacement."""
        return Expr(
            exp.Replace(
                this=self.__node__,
                expression=exp.Literal.string(pattern),
                replacement=exp.Literal.string(replacement),
            )
        )

    def strip_chars(self) -> Expr:
        """Strip leading and trailing whitespace."""
        return Expr(exp.Trim(this=self.__node__))

    def strip_chars_start(self) -> Expr:
        """Strip leading whitespace."""
        return Expr(exp.Trim(this=self.__node__, position="LEADING"))

    def strip_chars_end(self) -> Expr:
        """Strip trailing whitespace."""
        return Expr(exp.Trim(this=self.__node__, position="TRAILING"))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        args: dict[str, exp.Expression] = {
            "this": self.__node__,
            "start": exp.Literal.number(offset + 1),  # SQL is 1-indexed
        }
        if length is not None:
            args["length"] = exp.Literal.number(length)
        return Expr(exp.Substring(**args))


class _DatetimeNamespace:
    """Datetime operations namespace (equivalent to pl.Expr.dt)."""

    __slots__ = ("__node__",)

    def __init__(self, node: exp.Expression) -> None:
        self.__node__ = node

    def year(self) -> Expr:
        """Extract the year."""
        return Expr(exp.Year(this=self.__node__))

    def month(self) -> Expr:
        """Extract the month."""
        return Expr(exp.Month(this=self.__node__))

    def day(self) -> Expr:
        """Extract the day."""
        return Expr(exp.Day(this=self.__node__))

    def hour(self) -> Expr:
        """Extract the hour."""
        return Expr(exp.Hour(this=self.__node__))

    def minute(self) -> Expr:
        """Extract the minute."""
        return Expr(exp.Minute(this=self.__node__))

    def second(self) -> Expr:
        """Extract the second."""
        return Expr(exp.Second(this=self.__node__))

    def weekday(self) -> Expr:
        """Extract the day of week (0=Monday, 6=Sunday)."""
        return Expr(exp.DayOfWeek(this=self.__node__))

    def week(self) -> Expr:
        """Extract the week number."""
        return Expr(exp.Week(this=self.__node__))

    def date(self) -> Expr:
        """Extract the date part."""
        return Expr(exp.Date(this=self.__node__))
