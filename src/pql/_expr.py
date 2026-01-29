"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, Self

import duckdb
import pyochain as pc

from . import datatypes, sql

if TYPE_CHECKING:
    from ._types import IntoExpr, RoundMode


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name))

    def __getattr__(self, name: str) -> Expr:
        return self.__call__(name)


col: Col = Col()


def all() -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    return Expr(sql.all())


@dataclass(slots=True)
class SqlExprHandler:
    """A wrapper for DuckDB expressions."""

    _expr: sql.SqlExpr


@dataclass(slots=True)
class Expr(SqlExprHandler):
    """Expression wrapper providing Polars-like API over DuckDB expressions."""

    def __repr__(self) -> str:
        return f"Expr({self._expr})"

    @property
    def str(self) -> ExprStringNameSpace:
        """Access string operations."""
        return ExprStringNameSpace(self._expr)

    @property
    def list(self) -> ExprListNameSpace:
        """Access list operations."""
        return ExprListNameSpace(self._expr)

    @property
    def struct(self) -> ExprStructNameSpace:
        """Access struct operations."""
        return ExprStructNameSpace(self._expr)

    @property
    def expr(self) -> sql.SqlExpr:
        """Get the underlying DuckDB expression."""
        return self._expr

    def __add__(self, other: IntoExpr) -> Self:
        return self.add(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) + self._expr)

    def __sub__(self, other: IntoExpr) -> Self:
        return self.sub(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) - self._expr)

    def __mul__(self, other: IntoExpr) -> Self:
        return self.mul(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) * self._expr)

    def __truediv__(self, other: IntoExpr) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) / self._expr)

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) // self._expr)

    def __mod__(self, other: IntoExpr) -> Self:
        return self.mod(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) % self._expr)

    def __pow__(self, other: IntoExpr) -> Self:
        return self.pow(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) ** self._expr)

    def __neg__(self) -> Self:
        return self.neg()

    def __eq__(self, other: IntoExpr) -> Self:  # type: ignore[override]
        return self.eq(other)

    def __ne__(self, other: IntoExpr) -> Self:  # type: ignore[override]
        return self.ne(other)

    def __lt__(self, other: IntoExpr) -> Self:
        return self.lt(other)

    def __le__(self, other: IntoExpr) -> Self:
        return self.le(other)

    def __gt__(self, other: IntoExpr) -> Self:
        return self.gt(other)

    def __ge__(self, other: IntoExpr) -> Self:
        return self.ge(other)

    def __and__(self, other: IntoExpr) -> Self:
        return self.and_(other)

    def __rand__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) & self._expr)

    def __or__(self, other: IntoExpr) -> Self:
        return self.or_(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self.__class__(sql.from_value(other) | self._expr)

    def __invert__(self) -> Self:
        return self.not_()

    def __hash__(self) -> int:
        return hash(str(self._expr))

    def add(self, other: Any) -> Self:  # noqa: ANN401
        """Add another expression or value."""
        return self.__class__(self._expr + sql.from_value(other))

    def sub(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr - sql.from_value(other))

    def mul(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr * sql.from_value(other))

    def truediv(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr / sql.from_value(other))

    def floordiv(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr // sql.from_value(other))

    def mod(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr % sql.from_value(other))

    def pow(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr ** sql.from_value(other))

    def neg(self) -> Self:
        return self.__class__(-self._expr)

    def abs(self) -> Self:
        return self.__class__(sql.fns.abs(self._expr))

    def eq(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr == sql.from_value(other))

    def ne(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr != sql.from_value(other))

    def lt(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr < sql.from_value(other))

    def le(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr <= sql.from_value(other))

    def gt(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr > sql.from_value(other))

    def ge(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr >= sql.from_value(other))

    def and_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr & sql.from_value(others))

    def or_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr | sql.from_value(others))

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

    def cast(self, dtype: datatypes.DataType) -> Self:
        """Cast to a different data type."""
        return self.__class__(self._expr.cast(dtype))

    def is_in(self, other: Collection[IntoExpr] | IntoExpr) -> Self:
        """Check if value is in an iterable of values."""
        return self.__class__(self._expr.isin(*sql.from_iter(other)))

    def pipe[**P, T](
        self,
        function: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Apply a function to the expression."""
        return function(self, *args, **kwargs)

    def floor(self) -> Self:
        """Round down to the nearest integer."""
        return self.__class__(sql.fns.floor(self._expr))

    def ceil(self) -> Self:
        """Round up to the nearest integer."""
        return self.__class__(sql.fns.ceil(self._expr))

    def round(self, decimals: int = 0, *, mode: RoundMode = "half_to_even") -> Self:
        """Round to given number of decimal places."""
        match mode:
            case "half_to_even":
                rounding_func = sql.fns.round_even
            case "half_away_from_zero":
                rounding_func = sql.fns.round
        return self.__class__(rounding_func(self._expr, sql.lit(decimals)))

    def sqrt(self) -> Self:
        """Compute the square root."""
        return self.__class__(sql.fns.sqrt(self._expr))

    def cbrt(self) -> Self:
        """Compute the cube root."""
        return self.__class__(sql.fns.cbrt(self._expr))

    def log(self, base: float = 2.718281828459045) -> Self:
        """Compute the logarithm."""
        return self.__class__(sql.fns.log(sql.lit(base), self._expr))

    def log10(self) -> Self:
        """Compute the base 10 logarithm."""
        return self.__class__(sql.fns.log10(self._expr))

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x."""
        return self.__class__(sql.fns.ln(self._expr.__add__(sql.lit(1))))

    def exp(self) -> Self:
        """Compute the exponential."""
        return self.__class__(sql.fns.exp(self._expr))

    def sin(self) -> Self:
        """Compute the sine."""
        return self.__class__(sql.fns.sin(self._expr))

    def cos(self) -> Self:
        """Compute the cosine."""
        return self.__class__(sql.fns.cos(self._expr))

    def tan(self) -> Self:
        """Compute the tangent."""
        return self.__class__(sql.fns.tan(self._expr))

    def arctan(self) -> Self:
        """Compute the arc tangent."""
        return self.__class__(sql.fns.atan(self._expr))

    def sinh(self) -> Self:
        """Compute the hyperbolic sine."""
        return self.__class__(
            sql.fns.func(
                "/",
                sql.fns.exp(self._expr).__sub__(sql.fns.exp(-self._expr)),
                sql.lit("2"),
            )
        )

    def cosh(self) -> Self:
        """Compute the hyperbolic cosine."""
        return self.__class__(
            sql.fns.func(
                "/",
                sql.fns.exp(self._expr).__add__(sql.fns.exp(-self._expr)),
                sql.lit("2"),
            )
        )

    def tanh(self) -> Self:
        """Compute the hyperbolic tangent."""
        exp_x = sql.fns.exp(self._expr)
        exp_neg_x = sql.fns.exp(-self._expr)
        return self.__class__(
            (exp_x.__sub__(exp_neg_x)).__truediv__(exp_x.__add__(exp_neg_x))
        )

    def degrees(self) -> Self:
        """Convert radians to degrees."""
        return self.__class__(sql.fns.degrees(self._expr))

    def radians(self) -> Self:
        """Convert degrees to radians."""
        return self.__class__(sql.fns.radians(self._expr))

    def sign(self) -> Self:
        """Get the sign of the value."""
        return self.__class__(sql.fns.sign(self._expr))

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value."""
        return self.__class__(
            sql.WindowExpr(rows_end=pc.Some(0), ignore_nulls=True).call(
                sql.fns.last_value(self._expr),
            )
        )

    def backward_fill(self) -> Self:
        """Fill null values with the next non-null value."""
        return self.__class__(
            sql.WindowExpr(rows_start=pc.Some(0), ignore_nulls=True).call(
                sql.fns.first_value(self._expr),
            )
        )

    def interpolate(self) -> Self:
        """Interpolate null values using linear interpolation."""
        last_value = sql.WindowExpr(rows_end=pc.Some(0), ignore_nulls=True).call(
            sql.fns.last_value(self._expr),
        )
        return self.__class__(
            sql.coalesce(
                self._expr,
                last_value.__add__(
                    sql.WindowExpr(rows_start=pc.Some(0), ignore_nulls=True)
                    .call(sql.fns.first_value(self._expr))
                    .__sub__(last_value)
                ).__truediv__(sql.lit(2)),
            )
        )

    def is_nan(self) -> Self:
        """Check if value is NaN."""
        return self.__class__(sql.fns.isnan(self._expr))

    def is_not_nan(self) -> Self:
        """Check if value is not NaN."""
        return self.__class__(~sql.fns.isnan(self._expr))

    def is_finite(self) -> Self:
        """Check if value is finite."""
        return self.__class__(sql.fns.isfinite(self._expr))

    def is_infinite(self) -> Self:
        """Check if value is infinite."""
        return self.__class__(sql.fns.isinf(self._expr))

    def fill_nan(self, value: int | float | Expr | None) -> Self:  # noqa: PYI041
        """Fill NaN values."""
        return self.__class__(
            sql.when(sql.fns.isnan(self._expr), sql.from_value(value)).otherwise(
                self._expr
            )
        )

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash."""
        return self.__class__(sql.fns.hash(self._expr, sql.lit(seed)))

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values."""
        return self.__class__(
            sql.when(
                self._expr.__eq__(sql.from_value(old)), sql.from_value(new)
            ).otherwise(self._expr)
        )

    def repeat_by(self, by: Expr | int) -> Self:
        """Repeat values by count, returning a list."""
        return self.__class__(
            sql.fns.list_transform(
                sql.fns.range(sql.from_value(by)),
                sql.fn_once("_", self._expr),
            )
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        return self.__class__(
            sql.WindowExpr(partition_by=pc.Seq((self._expr,)))
            .call(sql.fns.count(sql.all()))
            .__gt__(sql.lit(1)),
        )

    def is_unique(self) -> Self:
        """Check if value is unique."""
        return self.__class__(
            sql.WindowExpr(partition_by=pc.Seq((self._expr,)))
            .call(sql.fns.count(sql.all()))
            .__eq__(sql.lit(1)),
        )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self.__class__(
            sql.WindowExpr(partition_by=pc.Seq((self._expr,)))
            .call(sql.fns.row_number())
            .__eq__(sql.lit(1))
        )

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return self.__class__(
            sql.WindowExpr(
                partition_by=pc.Seq((self._expr,)),
                order_by=pc.Seq((self._expr,)),
                descending=pc.Some(value=True),
                nulls_last=pc.Some(value=True),
            )
            .call(sql.fns.row_number())
            .__eq__(sql.lit(1))
        )


@dataclass(slots=True)
class ExprStringNameSpace:
    """String operations namespace (equivalent to pl.Expr.str)."""

    _expr: sql.SqlExpr

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return Expr(sql.fns.upper(self._expr))

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return Expr(sql.fns.lower(self._expr))

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return Expr(sql.fns.length(self._expr))

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        """Check if string contains a pattern."""
        match literal:
            case True:
                return Expr(sql.fns.contains(self._expr, sql.lit(pattern)))
            case False:
                return Expr(sql.fns.regexp_matches(self._expr, sql.lit(pattern)))

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return Expr(sql.fns.starts_with(self._expr, sql.lit(prefix)))

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return Expr(sql.fns.ends_with(self._expr, sql.lit(suffix)))

    def replace(
        self,
        pattern: str,
        value: str | IntoExpr,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> Expr:
        """Replace first matching substring with a new string value."""
        value_expr = sql.from_value(value)
        pattern_expr = sql.lit(re.escape(pattern) if literal else pattern)

        def _replace_once(expr: sql.SqlExpr) -> sql.SqlExpr:
            return sql.fns.regexp_replace(expr, pattern_expr, value_expr)

        match n:
            case 0:
                return Expr(self._expr)
            case n_val if n_val < 0:
                return Expr(
                    sql.fns.regexp_replace(
                        self._expr,
                        pattern_expr,
                        value_expr,
                        sql.lit("g"),
                    )
                )
            case _:
                return Expr(
                    pc.Iter(range(n)).fold(
                        self._expr, lambda acc, _: _replace_once(acc)
                    )
                )

    def strip_chars(self, characters: str | None = None) -> Expr:
        """Strip leading and trailing characters."""
        match characters:
            case None:
                return Expr(sql.fns.trim(self._expr))
            case _:
                return Expr(sql.fns.trim(self._expr, sql.lit(characters)))

    def strip_chars_start(self, characters: IntoExpr = None) -> Expr:
        """Strip leading characters."""
        match characters:
            case None:
                return Expr(sql.fns.ltrim(self._expr))
            case _:
                characters_expr = sql.from_value(characters)
                return Expr(
                    sql.when(characters_expr.isnull(), sql.lit(None)).otherwise(
                        sql.fns.ltrim(self._expr, characters_expr)
                    )
                )

    def strip_chars_end(self, characters: IntoExpr = None) -> Expr:
        """Strip trailing characters."""
        match characters:
            case None:
                return Expr(sql.fns.rtrim(self._expr))
            case _:
                characters_expr = sql.from_value(characters)
                return Expr(
                    sql.when(characters_expr.isnull(), sql.lit(None)).otherwise(
                        sql.fns.rtrim(self._expr, characters_expr)
                    )
                )

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        match length:
            case None:
                args = (self._expr, sql.lit(offset + 1))
            case _:
                args = (
                    self._expr,
                    sql.lit(offset + 1),
                    sql.lit(length),
                )
        return Expr(sql.fns.substring(*args))

    def len_bytes(self) -> Expr:
        """Get the length in bytes."""
        return Expr(sql.fns.octet_length(sql.fns.encode(self._expr)))

    def split(self, by: str) -> Expr:
        """Split string by separator."""
        return Expr(sql.fns.string_split(self._expr, sql.lit(by)))

    def extract_all(self, pattern: str | Expr) -> Expr:
        """Extract all regex matches."""
        return Expr(sql.fns.regexp_extract_all(self._expr, sql.from_value(pattern)))

    def count_matches(self, pattern: str | Expr, *, literal: bool = False) -> Expr:
        """Count pattern matches."""
        pattern_expr = sql.from_value(pattern)
        match literal:
            case False:
                return Expr(
                    sql.fns.length(
                        sql.fns.regexp_extract_all(
                            self._expr,
                            pattern_expr,
                        ),
                    )
                )
            case True:
                return Expr(
                    sql.fns.length(self._expr)
                    .__sub__(
                        sql.fns.length(
                            sql.fns.replace(
                                self._expr,
                                pattern_expr,
                                sql.lit(""),
                            ),
                        )
                    )
                    .__truediv__(sql.fns.length(pattern_expr))
                )

    def to_date(self, format: str | None = None) -> Expr:  # noqa: A002
        """Convert string to date."""
        match format:
            case None:
                return Expr(self._expr.cast(datatypes.Date))
            case _:
                return Expr(
                    sql.fns.strptime(self._expr, sql.lit(format)).cast(datatypes.Date)
                )

    def to_datetime(self, format: str | None = None, *, time_unit: str = "us") -> Expr:  # noqa: A002
        """Convert string to datetime."""
        precision_map = {"ns": "TIMESTAMP_NS", "us": "TIMESTAMP", "ms": "TIMESTAMP_MS"}
        match format:
            case None:
                base_expr = self._expr.cast(precision_map.get(time_unit, "TIMESTAMP"))
            case _:
                base_expr = sql.fns.strptime(self._expr, sql.lit(format))
        return Expr(base_expr)

    def to_time(self, format: str | None = None) -> Expr:  # noqa: A002
        """Convert string to time."""
        match format:
            case None:
                return Expr(self._expr.cast(datatypes.Time))
            case _:
                return Expr(
                    sql.fns.strptime(self._expr, sql.lit(format)).cast(datatypes.Time)
                )

    def to_decimal(self, *, scale: int = 38) -> Expr:
        """Convert string to decimal."""
        precision = min(scale, 38)
        return Expr(self._expr.cast(f"DECIMAL({precision}, {precision // 2})"))

    def strip_prefix(self, prefix: IntoExpr) -> Expr:
        """Strip prefix from string."""
        match prefix:
            case str() as prefix_str:
                return Expr(
                    sql.fns.regexp_replace(
                        self._expr,
                        sql.lit(f"^{re.escape(prefix_str)}"),
                        sql.lit(""),
                    )
                )
            case _:
                prefix_expr = sql.from_value(prefix)
                return Expr(
                    sql.when(prefix_expr.isnull(), sql.lit(None)).otherwise(
                        sql.when(
                            sql.fns.starts_with(self._expr, prefix_expr),
                            sql.fns.substring(
                                self._expr,
                                sql.fns.length(prefix_expr).__add__(sql.lit(1)),
                            ),
                        ).otherwise(self._expr)
                    )
                )

    def strip_suffix(self, suffix: IntoExpr) -> Expr:
        """Strip suffix from string."""
        match suffix:
            case str() as suffix_str:
                return Expr(
                    sql.fns.regexp_replace(
                        self._expr,
                        sql.lit(f"{re.escape(suffix_str)}$"),
                        sql.lit(""),
                    )
                )
            case _:
                suffix_expr = sql.from_value(suffix)
                return Expr(
                    sql.when(suffix_expr.isnull(), sql.lit(None)).otherwise(
                        sql.when(
                            sql.fns.ends_with(self._expr, suffix_expr),
                            sql.fns.substring(
                                self._expr,
                                sql.lit(1),
                                sql.fns.length(self._expr).__sub__(
                                    sql.fns.length(suffix_expr)
                                ),
                            ),
                        ).otherwise(self._expr)
                    )
                )

    def head(self, n: int) -> Expr:
        """Get first n characters."""
        return Expr(sql.fns.left(self._expr, sql.lit(n)))

    def tail(self, n: int) -> Expr:
        """Get last n characters."""
        return Expr(sql.fns.right(self._expr, sql.lit(n)))

    def reverse(self) -> Expr:
        """Reverse the string."""
        return Expr(sql.fns.reverse(self._expr))

    def replace_all(
        self, pattern: str, value: IntoExpr, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences."""
        value_expr = sql.from_value(value)
        match literal:
            case True:
                return Expr(
                    sql.fns.replace(
                        self._expr,
                        sql.lit(pattern),
                        value_expr,
                    )
                )
            case False:
                return Expr(
                    sql.fns.regexp_replace(
                        self._expr,
                        sql.lit(pattern),
                        value_expr,
                        sql.lit("g"),
                    )
                )

    def to_titlecase(self) -> Expr:
        """Convert to title case."""
        elem = sql.col("_")
        return Expr(
            sql.fns.list_aggregate(
                sql.fns.list_transform(
                    sql.fns.regexp_extract_all(
                        sql.fns.lower(self._expr),
                        sql.lit(r"[a-z]*[^a-z]*"),
                    ),
                    sql.fn_once(
                        "_",
                        sql.fns.concat(
                            sql.fns.upper(sql.fns.array_extract(elem, sql.lit(1))),
                            sql.fns.substring(elem, sql.lit(2)),
                        ),
                    ),
                ),
                sql.lit("string_agg"),
                sql.lit(""),
            )
        )


@dataclass(slots=True)
class ExprListNameSpace(SqlExprHandler):
    """List operations namespace (equivalent to pl.Expr.list)."""

    def len(self) -> Expr:
        """Return the number of elements in each list."""
        return Expr(sql.fns.len(self._expr))

    def unique(self) -> Expr:
        """Return unique values in each list."""
        distinct_expr = sql.fns.list_distinct(self._expr)
        return Expr(
            sql.when(
                sql.fns.array_position(
                    self._expr,
                    sql.lit(None),
                ).isnotnull(),
                sql.fns.list_append(
                    distinct_expr,
                    sql.lit(None),
                ),
            ).otherwise(distinct_expr)
        )

    def contains(self, item: IntoExpr, *, nulls_equal: bool = True) -> Expr:
        """Check if sublists contain the given item."""
        item_expr = sql.from_value(item)
        contains_expr = sql.fns.list_contains(self._expr, item_expr)
        if nulls_equal:
            false_expr = duckdb.SQLExpression("false")
            return Expr(
                sql.when(
                    item_expr.isnull(),
                    sql.coalesce(
                        sql.fns.array_position(
                            self._expr,
                            sql.lit(None),
                        ).isnotnull(),
                        false_expr,
                    ),
                ).otherwise(sql.coalesce(contains_expr, false_expr))
            )
        return Expr(contains_expr)

    def get(self, index: int) -> Expr:
        """Return the value by index in each list."""
        return Expr(
            sql.fns.list_extract(
                self._expr,
                sql.lit(index + 1 if index >= 0 else index),
            )
        )

    def min(self) -> Expr:
        """Compute the min value of the lists in the array."""
        return Expr(sql.fns.list_min(self._expr))

    def max(self) -> Expr:
        """Compute the max value of the lists in the array."""
        return Expr(sql.fns.list_max(self._expr))

    def mean(self) -> Expr:
        """Compute the mean value of the lists in the array."""
        return Expr(sql.fns.list_avg(self._expr))

    def median(self) -> Expr:
        """Compute the median value of the lists in the array."""
        return Expr(sql.fns.list_median(self._expr))

    def sum(self) -> Expr:
        """Compute the sum value of the lists in the array."""
        expr_no_nulls = sql.fns.list_filter(
            self._expr, sql.fn_once("_", sql.col("_").isnotnull())
        )
        return Expr(
            sql.when(
                sql.fns.array_length(
                    expr_no_nulls,
                ).__eq__(sql.lit(0)),
                sql.lit(0),
            ).otherwise(sql.fns.list_sum(expr_no_nulls))
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the expression."""
        sort_direction = "DESC" if descending else "ASC"
        nulls_position = "NULLS LAST" if nulls_last else "NULLS FIRST"
        return Expr(
            sql.fns.list_sort(
                self._expr,
                sql.lit(sort_direction),
                sql.lit(nulls_position),
            )
        )


@dataclass(slots=True)
class ExprStructNameSpace(SqlExprHandler):
    """Struct operations namespace (equivalent to pl.Expr.struct)."""

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name."""
        return Expr(sql.fns.struct_extract(self._expr, sql.lit(name)))
