"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Concatenate, Self

import duckdb
import pyochain as pc

from . import datatypes


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


def all() -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    return Expr(duckdb.StarExpression())


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

    def __add__(self, other: object) -> Self:
        return self.add(other)

    def __radd__(self, other: object) -> Self:
        return self.__class__(_to_value(other) + self._expr)

    def __sub__(self, other: object) -> Self:
        return self.sub(other)

    def __rsub__(self, other: object) -> Self:
        return self.__class__(_to_value(other) - self._expr)

    def __mul__(self, other: object) -> Self:
        return self.mul(other)

    def __rmul__(self, other: object) -> Self:
        return self.__class__(_to_value(other) * self._expr)

    def __truediv__(self, other: object) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: object) -> Self:
        return self.__class__(_to_value(other) / self._expr)

    def __floordiv__(self, other: object) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: object) -> Self:
        return self.__class__(_to_value(other) // self._expr)

    def __mod__(self, other: object) -> Self:
        return self.mod(other)

    def __rmod__(self, other: object) -> Self:
        return self.__class__(_to_value(other) % self._expr)

    def __pow__(self, other: object) -> Self:
        return self.pow(other)

    def __rpow__(self, other: object) -> Self:
        return self.__class__(_to_value(other) ** self._expr)

    def __neg__(self) -> Self:
        return self.neg()

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self.eq(other)

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self.ne(other)

    def __lt__(self, other: object) -> Self:
        return self.lt(other)

    def __le__(self, other: object) -> Self:
        return self.le(other)

    def __gt__(self, other: object) -> Self:
        return self.gt(other)

    def __ge__(self, other: object) -> Self:
        return self.ge(other)

    def __and__(self, other: object) -> Self:
        return self.and_(other)

    def __rand__(self, other: object) -> Self:
        return self.__class__(_to_value(other) & self._expr)

    def __or__(self, other: object) -> Self:
        return self.or_(other)

    def __ror__(self, other: object) -> Self:
        return self.__class__(_to_value(other) | self._expr)

    def __invert__(self) -> Self:
        return self.not_()

    def __hash__(self) -> int:
        return hash(str(self._expr))

    def add(self, other: object) -> Self:
        """Add another expression or value."""
        return self.__class__(self._expr + _to_value(other))

    def sub(self, other: object) -> Self:
        return self.__class__(self._expr - _to_value(other))

    def mul(self, other: object) -> Self:
        return self.__class__(self._expr * _to_value(other))

    def truediv(self, other: object) -> Self:
        return self.__class__(self._expr / _to_value(other))

    def floordiv(self, other: object) -> Self:
        return self.__class__(self._expr // _to_value(other))

    def mod(self, other: object) -> Self:
        return self.__class__(self._expr % _to_value(other))

    def pow(self, other: object) -> Self:
        return self.__class__(self._expr ** _to_value(other))

    def neg(self) -> Self:
        return self.__class__(-self._expr)

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

    def or_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr | _to_value(others))

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

    def is_in(self, values: Iterable[object]) -> Self:
        """Check if value is in an iterable of values."""
        return self.__class__(self._expr.isin(*pc.Iter(values).map(_to_value)))

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
        return self.__class__(duckdb.FunctionExpression("floor", self._expr))

    def ceil(self) -> Self:
        """Round up to the nearest integer."""
        return self.__class__(duckdb.FunctionExpression("ceil", self._expr))

    def round(self, decimals: int = 0, *, mode: str = "half_to_even") -> Self:
        """Round to given number of decimal places."""
        func = "round_even" if mode == "half_to_even" else "round"
        return self.__class__(
            duckdb.FunctionExpression(
                func, self._expr, duckdb.ConstantExpression(decimals)
            )
        )

    def sqrt(self) -> Self:
        """Compute the square root."""
        return self.__class__(duckdb.FunctionExpression("sqrt", self._expr))

    def cbrt(self) -> Self:
        """Compute the cube root."""
        return self.__class__(duckdb.FunctionExpression("cbrt", self._expr))

    def log(self, base: float = 2.718281828459045) -> Self:
        """Compute the logarithm."""
        return self.__class__(
            duckdb.FunctionExpression(
                "log", duckdb.ConstantExpression(str(base)), self._expr
            )
        )

    def log10(self) -> Self:
        """Compute the base 10 logarithm."""
        return self.__class__(duckdb.FunctionExpression("log10", self._expr))

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x."""
        return self.__class__(
            duckdb.FunctionExpression("ln", self._expr + duckdb.ConstantExpression(1))
        )

    def exp(self) -> Self:
        """Compute the exponential."""
        return self.__class__(duckdb.FunctionExpression("exp", self._expr))

    def sin(self) -> Self:
        """Compute the sine."""
        return self.__class__(duckdb.FunctionExpression("sin", self._expr))

    def cos(self) -> Self:
        """Compute the cosine."""
        return self.__class__(duckdb.FunctionExpression("cos", self._expr))

    def tan(self) -> Self:
        """Compute the tangent."""
        return self.__class__(duckdb.FunctionExpression("tan", self._expr))

    def arcsin(self) -> Self:
        """Compute the arc sine."""
        return self.__class__(duckdb.FunctionExpression("asin", self._expr))

    def arccos(self) -> Self:
        """Compute the arc cosine."""
        return self.__class__(duckdb.FunctionExpression("acos", self._expr))

    def arctan(self) -> Self:
        """Compute the arc tangent."""
        return self.__class__(duckdb.FunctionExpression("atan", self._expr))

    def sinh(self) -> Self:
        """Compute the hyperbolic sine."""
        return self.__class__(
            duckdb.FunctionExpression(
                "/",
                duckdb.FunctionExpression("exp", self._expr)
                - duckdb.FunctionExpression("exp", -self._expr),
                duckdb.ConstantExpression("2"),
            )
        )

    def cosh(self) -> Self:
        """Compute the hyperbolic cosine."""
        return self.__class__(
            duckdb.FunctionExpression(
                "/",
                duckdb.FunctionExpression("exp", self._expr)
                + duckdb.FunctionExpression("exp", -self._expr),
                duckdb.ConstantExpression("2"),
            )
        )

    def tanh(self) -> Self:
        """Compute the hyperbolic tangent."""
        exp_x = duckdb.FunctionExpression("exp", self._expr)
        exp_neg_x = duckdb.FunctionExpression("exp", -self._expr)
        return self.__class__((exp_x - exp_neg_x) / (exp_x + exp_neg_x))

    def degrees(self) -> Self:
        """Convert radians to degrees."""
        return self.__class__(duckdb.FunctionExpression("degrees", self._expr))

    def radians(self) -> Self:
        """Convert degrees to radians."""
        return self.__class__(duckdb.FunctionExpression("radians", self._expr))

    def sign(self) -> Self:
        """Get the sign of the value."""
        return self.__class__(duckdb.FunctionExpression("sign", self._expr))

    def cum_sum(self) -> Self:
        """Compute cumulative sum."""
        return self.__class__(
            duckdb.SQLExpression(f"SUM({self._expr}) OVER (ROWS UNBOUNDED PRECEDING)")
        )

    def cum_max(self) -> Self:
        """Compute cumulative max."""
        return self.__class__(
            duckdb.SQLExpression(f"MAX({self._expr}) OVER (ROWS UNBOUNDED PRECEDING)")
        )

    def cum_min(self) -> Self:
        """Compute cumulative min."""
        return self.__class__(
            duckdb.SQLExpression(f"MIN({self._expr}) OVER (ROWS UNBOUNDED PRECEDING)")
        )

    def cum_count(self, *, reverse: bool = False) -> Self:
        """Compute cumulative count."""
        window = (
            "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
            if reverse
            else "ROWS UNBOUNDED PRECEDING"
        )
        return self.__class__(
            duckdb.SQLExpression(f"COUNT({self._expr}) OVER ({window})")
        )

    def cum_prod(self) -> Self:
        """Compute cumulative product."""
        return self.__class__(
            duckdb.SQLExpression(
                f"PRODUCT({self._expr}) OVER (ROWS UNBOUNDED PRECEDING)"
            )
        )

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value."""
        return self.__class__(
            duckdb.SQLExpression(
                f"LAST_VALUE({self._expr} IGNORE NULLS) OVER (ROWS UNBOUNDED PRECEDING)"
            )
        )

    def backward_fill(self) -> Self:
        """Fill null values with the next non-null value."""
        return self.__class__(
            duckdb.SQLExpression(
                f"FIRST_VALUE({self._expr} IGNORE NULLS) OVER (ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)"
            )
        )

    def interpolate(self) -> Self:
        """Interpolate null values using linear interpolation."""
        return self.__class__(
            duckdb.SQLExpression(
                f"COALESCE({self._expr}, "
                f"LAST_VALUE({self._expr} IGNORE NULLS) OVER (ROWS UNBOUNDED PRECEDING) + "
                f"(FIRST_VALUE({self._expr} IGNORE NULLS) OVER (ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) - "
                f"LAST_VALUE({self._expr} IGNORE NULLS) OVER (ROWS UNBOUNDED PRECEDING)) / 2)"
            )
        )

    def clip(
        self,
        lower_bound: object | None = None,
        upper_bound: object | None = None,
    ) -> Self:
        """Clip values to bounds."""
        expr = self._expr
        if lower_bound is not None:
            expr = duckdb.FunctionExpression("greatest", expr, _to_value(lower_bound))
        if upper_bound is not None:
            expr = duckdb.FunctionExpression("least", expr, _to_value(upper_bound))
        return self.__class__(expr)

    def is_nan(self) -> Self:
        """Check if value is NaN."""
        return self.__class__(duckdb.FunctionExpression("isnan", self._expr))

    def is_not_nan(self) -> Self:
        """Check if value is not NaN."""
        return self.__class__(~duckdb.FunctionExpression("isnan", self._expr))

    def is_finite(self) -> Self:
        """Check if value is finite."""
        return self.__class__(duckdb.FunctionExpression("isfinite", self._expr))

    def is_infinite(self) -> Self:
        """Check if value is infinite."""
        return self.__class__(duckdb.FunctionExpression("isinf", self._expr))

    def fill_nan(self, value: object) -> Self:
        """Fill NaN values."""
        return self.__class__(
            duckdb.CaseExpression(
                duckdb.FunctionExpression("isnan", self._expr), _to_value(value)
            ).otherwise(self._expr)
        )

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash."""
        return self.__class__(
            duckdb.FunctionExpression(
                "hash", self._expr, duckdb.ConstantExpression(str(seed))
            )
        )

    def shuffle(self) -> Self:
        """Shuffle values randomly."""
        return self.__class__(duckdb.SQLExpression(f"{self._expr} ORDER BY RANDOM()"))

    def explode(self) -> Self:
        """Explode list values."""
        return self.__class__(duckdb.FunctionExpression("unnest", self._expr))

    def flatten(self) -> Self:
        """Flatten nested lists."""
        return self.__class__(duckdb.FunctionExpression("flatten", self._expr))

    def replace(
        self,
        old: object | Iterable[object],
        new: object | Iterable[object],
    ) -> Self:
        """Replace values."""
        old_val = _to_value(old)
        new_val = _to_value(new)
        return self.__class__(
            duckdb.CaseExpression(self._expr == old_val, new_val).otherwise(self._expr)
        )

    def repeat_by(self, by: Expr | int) -> Self:
        """Repeat values by count, returning a list."""
        by_expr = by.expr if isinstance(by, Expr) else duckdb.ConstantExpression(by)
        return self.__class__(
            duckdb.SQLExpression(f"list_transform(range({by_expr}), _ -> {self._expr})")
        )

    def peak_max(self) -> Self:
        """Check if value is a local maximum."""
        return self.__class__(
            duckdb.SQLExpression(
                f"{self._expr} > LAG({self._expr}) OVER () AND "
                f"{self._expr} > LEAD({self._expr}) OVER ()"
            )
        )

    def peak_min(self) -> Self:
        """Check if value is a local minimum."""
        return self.__class__(
            duckdb.SQLExpression(
                f"{self._expr} < LAG({self._expr}) OVER () AND "
                f"{self._expr} < LEAD({self._expr}) OVER ()"
            )
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        return self.__class__(
            duckdb.SQLExpression(f"COUNT(*) OVER (PARTITION BY {self._expr}) > 1")
        )

    def is_unique(self) -> Self:
        """Check if value is unique."""
        return self.__class__(
            duckdb.SQLExpression(f"COUNT(*) OVER (PARTITION BY {self._expr}) = 1")
        )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self.__class__(
            duckdb.SQLExpression(f"ROW_NUMBER() OVER (PARTITION BY {self._expr}) = 1")
        )

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return self.__class__(
            duckdb.SQLExpression(
                f"ROW_NUMBER() OVER (PARTITION BY {self._expr} ORDER BY (SELECT COUNT(*) FROM (SELECT 1)) DESC) = 1"
            )
        )

    @property
    def str(self) -> ExprStringNameSpace:
        """Access string operations."""
        return ExprStringNameSpace(self._expr)


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

    def len_bytes(self) -> Expr:
        """Get the length in bytes."""
        return Expr(
            duckdb.FunctionExpression(
                "octet_length", duckdb.FunctionExpression("encode", self._expr)
            )
        )

    def pad_start(self, length: int, fill_char: str = " ") -> Expr:
        """Pad string from the start."""
        return Expr(
            duckdb.FunctionExpression(
                "lpad",
                self._expr,
                duckdb.ConstantExpression(length),
                duckdb.ConstantExpression(fill_char),
            )
        )

    def pad_end(self, length: int, fill_char: str = " ") -> Expr:
        """Pad string from the end."""
        return Expr(
            duckdb.FunctionExpression(
                "rpad",
                self._expr,
                duckdb.ConstantExpression(length),
                duckdb.ConstantExpression(fill_char),
            )
        )

    def zfill(self, length: int) -> Expr:
        """Pad string with zeros from the start."""
        return self.pad_start(length, "0")

    def split(self, by: str, *, inclusive: bool = False) -> Expr:
        """Split string by separator."""
        func = "string_split_regex" if inclusive else "string_split"
        return Expr(
            duckdb.FunctionExpression(func, self._expr, duckdb.ConstantExpression(by))
        )

    def extract(self, pattern: str, group_index: int = 1) -> Expr:
        """Extract first regex match."""
        return Expr(
            duckdb.FunctionExpression(
                "regexp_extract",
                self._expr,
                duckdb.ConstantExpression(pattern),
                duckdb.ConstantExpression(group_index),
            )
        )

    def extract_all(self, pattern: str) -> Expr:
        """Extract all regex matches."""
        return Expr(
            duckdb.FunctionExpression(
                "regexp_extract_all", self._expr, duckdb.ConstantExpression(pattern)
            )
        )

    def count_matches(self, pattern: str, *, literal: bool = False) -> Expr:
        """Count pattern matches."""
        return (
            Expr(
                duckdb.FunctionExpression(
                    "length",
                    duckdb.FunctionExpression(
                        "regexp_extract_all",
                        self._expr,
                        duckdb.ConstantExpression(pattern),
                    ),
                )
            )
            if not literal
            else Expr(
                (
                    duckdb.FunctionExpression("length", self._expr)
                    - duckdb.FunctionExpression(
                        "length",
                        duckdb.FunctionExpression(
                            "replace",
                            self._expr,
                            duckdb.ConstantExpression(pattern),
                            duckdb.ConstantExpression(""),
                        ),
                    )
                )
                / duckdb.ConstantExpression(len(pattern))
            )
        )

    def to_date(self, fmt: str | None = None) -> Expr:
        """Convert string to date."""
        return (
            Expr(self._expr.cast(datatypes.Date))
            if fmt is None
            else Expr(
                duckdb.FunctionExpression(
                    "strptime", self._expr, duckdb.ConstantExpression(fmt)
                ).cast(datatypes.Date)
            )
        )

    def to_datetime(
        self,
        fmt: str | None = None,
        *,
        time_unit: str = "us",
        time_zone: str | None = None,
    ) -> Expr:
        """Convert string to datetime."""
        precision_map = {"ns": "TIMESTAMP_NS", "us": "TIMESTAMP", "ms": "TIMESTAMP_MS"}
        ts_type = precision_map.get(time_unit, "TIMESTAMP")

        base_expr = (
            self._expr.cast(ts_type)
            if fmt is None
            else duckdb.FunctionExpression(
                "strptime", self._expr, duckdb.ConstantExpression(fmt)
            )
        )

        return (
            Expr(
                duckdb.FunctionExpression(
                    "timezone", duckdb.ConstantExpression(time_zone), base_expr
                )
            )
            if time_zone
            else Expr(base_expr)
        )

    def to_time(self, fmt: str | None = None) -> Expr:
        """Convert string to time."""
        return (
            Expr(self._expr.cast(datatypes.Time))
            if fmt is None
            else Expr(
                duckdb.FunctionExpression(
                    "strptime", self._expr, duckdb.ConstantExpression(fmt)
                ).cast(datatypes.Time)
            )
        )

    def to_decimal(self, *, scale: int = 38) -> Expr:
        """Convert string to decimal."""
        precision = min(scale, 38)
        return Expr(self._expr.cast(f"DECIMAL({precision}, {precision // 2})"))

    def strip_prefix(self, prefix: str) -> Expr:
        """Strip prefix from string."""
        return Expr(
            duckdb.FunctionExpression(
                "ltrim", self._expr, duckdb.ConstantExpression(prefix)
            )
        )

    def strip_suffix(self, suffix: str) -> Expr:
        """Strip suffix from string."""
        return Expr(
            duckdb.FunctionExpression(
                "rtrim", self._expr, duckdb.ConstantExpression(suffix)
            )
        )

    def head(self, n: int) -> Expr:
        """Get first n characters."""
        return Expr(
            duckdb.FunctionExpression("left", self._expr, duckdb.ConstantExpression(n))
        )

    def tail(self, n: int) -> Expr:
        """Get last n characters."""
        return Expr(
            duckdb.FunctionExpression("right", self._expr, duckdb.ConstantExpression(n))
        )

    def reverse(self) -> Expr:
        """Reverse the string."""
        return Expr(duckdb.FunctionExpression("reverse", self._expr))

    def json_decode(self, dtype: datatypes.DataType) -> Expr:
        """Parse JSON string."""
        return Expr(duckdb.FunctionExpression("json", self._expr).cast(dtype))

    def json_path_match(self, path: str) -> Expr:
        """Extract value from JSON by path."""
        return Expr(
            duckdb.FunctionExpression(
                "json_extract", self._expr, duckdb.ConstantExpression(path)
            )
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool = False) -> Expr:
        """Replace all occurrences."""
        return (
            Expr(
                duckdb.FunctionExpression(
                    "replace",
                    self._expr,
                    duckdb.ConstantExpression(pattern),
                    duckdb.ConstantExpression(value),
                )
            )
            if literal
            else Expr(
                duckdb.FunctionExpression(
                    "regexp_replace",
                    self._expr,
                    duckdb.ConstantExpression(pattern),
                    duckdb.ConstantExpression(value),
                    duckdb.ConstantExpression("g"),
                )
            )
        )

    def replace_many(
        self, patterns: Iterable[str], replace_with: str | Iterable[str]
    ) -> Expr:
        """Replace multiple patterns."""
        return self.replace(patterns.__iter__().__next__(), str(replace_with))

    def to_titlecase(self) -> Expr:
        """Convert to title case."""
        return Expr(
            duckdb.SQLExpression(
                f"list_aggr(list_transform(string_split({self._expr}, ' '), x -> concat(upper(left(x, 1)), substring(x, 2))), 'string_agg', ' ')"
            )
        )

    def find(self, pattern: str, *, literal: bool = False) -> Expr:
        """Find position of pattern (0-indexed, returns -1 if not found)."""
        if literal:
            return Expr(
                duckdb.FunctionExpression(
                    "instr", self._expr, duckdb.ConstantExpression(pattern)
                )
                - duckdb.ConstantExpression(1)
            )
        return Expr(
            duckdb.SQLExpression(
                f"CASE WHEN regexp_matches({self._expr}, '{pattern}') "
                f"THEN instr({self._expr}, regexp_extract({self._expr}, '{pattern}')) - 1 "
                f"ELSE -1 END"
            )
        )
