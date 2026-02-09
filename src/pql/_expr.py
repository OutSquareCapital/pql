"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

import pyochain as pc

from . import sql
from .sql import ExprHandler, SqlExpr

if TYPE_CHECKING:
    from .sql import IntoExpr

RoundMode = Literal["half_to_even", "half_away_from_zero"]


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
class Expr(ExprHandler[SqlExpr]):
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
    def expr(self) -> SqlExpr:
        """Get the underlying DuckDB expression."""
        return self._expr

    def __add__(self, other: IntoExpr) -> Self:
        return self.add(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) + self._expr)

    def __sub__(self, other: IntoExpr) -> Self:
        return self.sub(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) - self._expr)

    def __mul__(self, other: IntoExpr) -> Self:
        return self.mul(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) * self._expr)

    def __truediv__(self, other: IntoExpr) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) / self._expr)

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) // self._expr)

    def __mod__(self, other: IntoExpr) -> Self:
        return self.mod(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) % self._expr)

    def __pow__(self, other: IntoExpr) -> Self:
        return self.pow(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) ** self._expr)

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
        return self.__class__(SqlExpr.from_value(other) & self._expr)

    def __or__(self, other: IntoExpr) -> Self:
        return self.or_(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self.__class__(SqlExpr.from_value(other) | self._expr)

    def __invert__(self) -> Self:
        return self.not_()

    def __hash__(self) -> int:
        return hash(str(self._expr))

    def add(self, other: Any) -> Self:  # noqa: ANN401
        """Add another expression or value."""
        return self.__class__(self._expr + SqlExpr.from_value(other))

    def sub(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr - SqlExpr.from_value(other))

    def mul(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr * SqlExpr.from_value(other))

    def truediv(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr / SqlExpr.from_value(other))

    def floordiv(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr // SqlExpr.from_value(other))

    def mod(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr % SqlExpr.from_value(other))

    def pow(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr ** SqlExpr.from_value(other))

    def neg(self) -> Self:
        return self.__class__(-self._expr)

    def abs(self) -> Self:
        return self.__class__(self._expr.abs())

    def eq(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr == SqlExpr.from_value(other))

    def ne(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr != SqlExpr.from_value(other))

    def lt(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr < SqlExpr.from_value(other))

    def le(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr <= SqlExpr.from_value(other))

    def gt(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr > SqlExpr.from_value(other))

    def ge(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr >= SqlExpr.from_value(other))

    def and_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr & SqlExpr.from_value(others))

    def or_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr | SqlExpr.from_value(others))

    def not_(self) -> Self:
        return self.__class__(~self._expr)

    def alias(self, name: str) -> Self:
        """Rename the expression."""
        return self.__class__(self._expr.alias(name))

    def is_null(self) -> Self:
        """Check if the expression is NULL."""
        return self.__class__(self._expr.is_null())

    def is_not_null(self) -> Self:
        """Check if the expression is not NULL."""
        return self.__class__(self._expr.is_not_null())

    def cast(self, dtype: sql.datatypes.DataType) -> Self:
        """Cast to a different data type."""
        return self.__class__(self._expr.cast(dtype))

    def is_in(self, other: Collection[IntoExpr] | IntoExpr) -> Self:
        """Check if value is in an iterable of values."""
        return self.__class__(self._expr.is_in(*SqlExpr.from_iter(other)))

    def floor(self) -> Self:
        """Round down to the nearest integer."""
        return self.__class__(self._expr.floor())

    def ceil(self) -> Self:
        """Round up to the nearest integer."""
        return self.__class__(self._expr.ceil())

    def round(self, decimals: int = 0, *, mode: RoundMode = "half_to_even") -> Self:
        """Round to given number of decimal places."""
        match mode:
            case "half_to_even":
                rounded = self._expr.round_even(sql.lit(decimals))
            case "half_away_from_zero":
                rounded = self._expr.round(sql.lit(decimals))
        return self.__class__(rounded)

    def sqrt(self) -> Self:
        """Compute the square root."""
        return self.__class__(self._expr.sqrt())

    def cbrt(self) -> Self:
        """Compute the cube root."""
        return self.__class__(self._expr.cbrt())

    def log(self, base: float = 2.718281828459045) -> Self:
        """Compute the logarithm."""
        return self.__class__(self._expr.log(sql.lit(base)))

    def log10(self) -> Self:
        """Compute the base 10 logarithm."""
        return self.__class__(self._expr.log10())

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x."""
        return self.__class__(self._expr.add(sql.lit(1)).ln())

    def exp(self) -> Self:
        """Compute the exponential."""
        return self.__class__(self._expr.exp())

    def sin(self) -> Self:
        """Compute the sine."""
        return self.__class__(self._expr.sin())

    def cos(self) -> Self:
        """Compute the cosine."""
        return self.__class__(self._expr.cos())

    def tan(self) -> Self:
        """Compute the tangent."""
        return self.__class__(self._expr.tan())

    def arctan(self) -> Self:
        """Compute the arc tangent."""
        return self.__class__(self._expr.atan())

    def sinh(self) -> Self:
        """Compute the hyperbolic sine."""
        return self.__class__(self._expr.sinh())

    def cosh(self) -> Self:
        """Compute the hyperbolic cosine."""
        return self.__class__(self._expr.cosh())

    def tanh(self) -> Self:
        """Compute the hyperbolic tangent."""
        exp_x = self._expr.exp()
        exp_neg_x = (-self._expr).exp()
        return self.__class__(exp_x.sub(exp_neg_x)).truediv(exp_x.add(exp_neg_x))

    def degrees(self) -> Self:
        """Convert radians to degrees."""
        return self.__class__(self._expr.degrees())

    def radians(self) -> Self:
        """Convert degrees to radians."""
        return self.__class__(self._expr.radians())

    def sign(self) -> Self:
        """Get the sign of the value."""
        return self.__class__(self._expr.sign())

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value."""
        return self.__class__(
            self._expr.last_value().over(rows_end=0, ignore_nulls=True)
        )

    def backward_fill(self) -> Self:
        """Fill null values with the next non-null value."""
        return self.__class__(self._expr.any_value().over(rows_start=0))

    def is_nan(self) -> Self:
        """Check if value is NaN."""
        return self.__class__(self._expr.isnan())

    def is_not_nan(self) -> Self:
        """Check if value is not NaN."""
        return self.__class__(~self._expr.isnan())

    def is_finite(self) -> Self:
        """Check if value is finite."""
        return self.__class__(self._expr.isfinite())

    def is_infinite(self) -> Self:
        """Check if value is infinite."""
        return self.__class__(self._expr.isinf())

    def fill_nan(self, value: int | float | Expr | None) -> Self:  # noqa: PYI041
        """Fill NaN values."""
        return self.__class__(
            sql.when(self._expr.isnan(), SqlExpr.from_value(value)).otherwise(
                self._expr
            )
        )

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash."""
        return self.__class__(self._expr.str.hash(sql.lit(seed)))

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values."""
        return self.__class__(
            sql.when(
                self._expr.eq(SqlExpr.from_value(old)), SqlExpr.from_value(new)
            ).otherwise(self._expr)
        )

    def repeat_by(self, by: Expr | int) -> Self:
        """Repeat values by count, returning a list."""
        return self.__class__(
            SqlExpr.from_value(by)
            .list.range()
            .list.transform(sql.fn_once("_", self._expr))
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        return self.__class__(
            sql.all().count().over(partition_by=pc.Seq((self._expr,))).gt(sql.lit(1))
        )

    def is_unique(self) -> Self:
        """Check if value is unique."""
        return self.__class__(
            sql.all().count().over(partition_by=pc.Seq((self._expr,))).eq(sql.lit(1))
        )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self.__class__(
            self._expr.row_number()
            .over(partition_by=pc.Seq((self._expr,)))
            .eq(sql.lit(value=1))
        )

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return self.__class__(
            self._expr.row_number()
            .over(
                partition_by=pc.Seq((self._expr,)),
                order_by=pc.Seq((self._expr,)),
                descending=True,
                nulls_last=True,
            )
            .eq(sql.lit(1))
        )


@dataclass(slots=True)
class ExprStringNameSpace:
    """String operations namespace (equivalent to pl.Expr.str)."""

    _expr: sql.SqlExpr

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return Expr(self._expr.str.upper())

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return Expr(self._expr.str.lower())

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return Expr(self._expr.str.length())

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        """Check if string contains a pattern."""
        match literal:
            case True:
                return Expr(self._expr.str.contains(sql.lit(pattern)))
            case False:
                return Expr(self._expr.re.matches(sql.lit(pattern)))

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return Expr(self._expr.str.starts_with(sql.lit(prefix)))

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return Expr(self._expr.str.ends_with(sql.lit(suffix)))

    def replace(
        self, pattern: str, value: str | IntoExpr, *, literal: bool = False, n: int = 1
    ) -> Expr:
        """Replace first matching substring with a new string value."""
        value_expr = SqlExpr.from_value(value)
        pattern_expr = sql.lit(re.escape(pattern) if literal else pattern)

        def _replace_once(expr: SqlExpr) -> SqlExpr:
            return expr.str.replace(pattern_expr, value_expr)

        match n:
            case 0:
                return Expr(self._expr)
            case n_val if n_val < 0:
                return Expr(
                    self._expr.re.replace(pattern_expr, value_expr, sql.lit("g"))
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
                return Expr(self._expr.str.trim())
            case _:
                return Expr(self._expr.str.trim(sql.lit(characters)))

    def strip_chars_start(self, characters: IntoExpr = None) -> Expr:
        """Strip leading characters."""
        match characters:
            case None:
                return Expr(self._expr.str.ltrim())
            case _:
                characters_expr = SqlExpr.from_value(characters)
                return Expr(
                    sql.when(characters_expr.is_null(), sql.lit(None)).otherwise(
                        self._expr.str.ltrim(characters_expr)
                    )
                )

    def strip_chars_end(self, characters: IntoExpr = None) -> Expr:
        """Strip trailing characters."""
        match characters:
            case None:
                return Expr(self._expr.str.rtrim())
            case _:
                characters_expr = SqlExpr.from_value(characters)
                return Expr(
                    sql.when(characters_expr.is_null(), sql.lit(None)).otherwise(
                        self._expr.str.rtrim(characters_expr)
                    )
                )

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        match length:
            case None:
                return Expr(self._expr.str.substring(sql.lit(offset + 1)))
            case _:
                return Expr(
                    self._expr.str.substring(sql.lit(offset + 1), sql.lit(length))
                )

    def len_bytes(self) -> Expr:
        """Get the length in bytes."""
        return Expr(self._expr.encode().octet_length())

    def split(self, by: str) -> Expr:
        """Split string by separator."""
        return Expr(self._expr.str.split(sql.lit(by)))

    def extract_all(self, pattern: str | Expr) -> Expr:
        """Extract all regex matches."""
        return Expr(self._expr.re.extract_all(SqlExpr.from_value(pattern)))

    def count_matches(self, pattern: str | Expr, *, literal: bool = False) -> Expr:
        """Count pattern matches."""
        pattern_expr = SqlExpr.from_value(pattern)
        match literal:
            case False:
                return Expr(
                    self._expr.re.extract_all(pattern_expr).list.len(),
                )
            case True:
                return Expr(
                    self._expr.str.length()
                    .sub(self._expr.str.replace(pattern_expr, sql.lit("")).str.length())
                    .truediv(pattern_expr.str.length())
                )

    def to_date(self, format: str | None = None) -> Expr:  # noqa: A002
        """Convert string to date."""
        match format:
            case None:
                return Expr(self._expr.cast(sql.datatypes.Date))
            case _:
                return Expr(self._expr.str.strptime(sql.lit(format)))

    def to_datetime(
        self,
        format: str | None = None,  # noqa: A002
        *,
        time_unit: sql.datatypes.TimeUnit = "us",
    ) -> Expr:
        """Convert string to datetime."""
        match format:
            case None:
                return Expr(self._expr.cast(sql.datatypes.PRECISION_MAP[time_unit]))
            case _:
                return Expr(self._expr.str.strptime(sql.lit(format)))

    def to_time(self, format: str | None = None) -> Expr:  # noqa: A002
        """Convert string to time."""
        match format:
            case None:
                return Expr(self._expr.cast(sql.datatypes.Time))
            case _:
                return Expr(
                    self._expr.str.strptime(sql.lit(format)).cast(sql.datatypes.Time)
                )

    def to_decimal(self, *, scale: int = 38) -> Expr:
        """Convert string to decimal."""
        precision = min(scale, 38)

        return Expr(self._expr.cast(f"DECIMAL({precision}, {precision // 2})"))  # pyright: ignore[reportArgumentType]

    def strip_prefix(self, prefix: IntoExpr) -> Expr:
        """Strip prefix from string."""
        match prefix:
            case str() as prefix_str:
                return Expr(
                    self._expr.re.replace(
                        sql.lit(f"^{re.escape(prefix_str)}"), sql.lit("")
                    )
                )
            case _:
                prefix_expr = SqlExpr.from_value(prefix)
                return Expr(
                    sql.when(prefix_expr.is_null(), sql.lit(None)).otherwise(
                        sql.when(
                            self._expr.str.starts_with(prefix_expr),
                            self._expr.str.substring(
                                prefix_expr.str.length().add(sql.lit(1))
                            ),
                        ).otherwise(self._expr)
                    )
                )

    def strip_suffix(self, suffix: IntoExpr) -> Expr:
        """Strip suffix from string."""
        match suffix:
            case str() as suffix_str:
                return Expr(
                    self._expr.re.replace(
                        sql.lit(f"{re.escape(suffix_str)}$"), sql.lit("")
                    )
                )
            case _:
                suffix_expr = SqlExpr.from_value(suffix)
                return Expr(
                    sql.when(suffix_expr.is_null(), sql.lit(None)).otherwise(
                        sql.when(
                            self._expr.str.ends_with(suffix_expr),
                            self._expr.str.substring(
                                sql.lit(1),
                                self._expr.str.length().sub(suffix_expr.str.length()),
                            ),
                        ).otherwise(self._expr)
                    )
                )

    def head(self, n: int) -> Expr:
        """Get first n characters."""
        return Expr(self._expr.str.left(sql.lit(n)))

    def tail(self, n: int) -> Expr:
        """Get last n characters."""
        return Expr(self._expr.str.right(sql.lit(n)))

    def reverse(self) -> Expr:
        """Reverse the string."""
        return Expr(self._expr.str.reverse())

    def replace_all(
        self, pattern: str, value: IntoExpr, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences."""
        value_expr = SqlExpr.from_value(value)
        match literal:
            case True:
                return Expr(self._expr.str.replace(sql.lit(pattern), value_expr))
            case False:
                return Expr(
                    self._expr.re.replace(sql.lit(pattern), value_expr, sql.lit("g"))
                )

    def to_titlecase(self) -> Expr:
        """Convert to title case."""
        elem = sql.col("_")
        lambda_expr = sql.fn_once(
            "_",
            elem.list.extract(sql.lit(1))
            .str.upper()
            .list.concat(elem.str.substring(sql.lit(2))),
        )
        return Expr(
            self._expr.str.lower()
            .re.extract_all(sql.lit(r"[a-z]*[^a-z]*"))
            .list.transform(lambda_expr)
            .list.aggregate(sql.lit("string_agg"), sql.lit(""))
        )


@dataclass(slots=True)
class ExprListNameSpace(ExprHandler[sql.SqlExpr]):
    """List operations namespace (equivalent to pl.Expr.list)."""

    def len(self) -> Expr:
        """Return the number of elements in each list."""
        return Expr(self._expr.list.length())

    def unique(self) -> Expr:
        """Return unique values in each list."""
        distinct_expr = self._expr.list.distinct()
        return Expr(
            sql.when(
                self._expr.list.position(sql.lit(None)).is_not_null(),
                distinct_expr.list.append(sql.lit(None)),
            ).otherwise(distinct_expr)
        )

    def contains(self, item: IntoExpr, *, nulls_equal: bool = True) -> Expr:
        """Check if sublists contain the given item."""
        item_expr = SqlExpr.from_value(item)
        contains_expr = self._expr.list.contains(item_expr)
        if nulls_equal:
            return Expr(
                sql.when(
                    item_expr.is_null(),
                    sql.coalesce(
                        self._expr.list.position(sql.lit(None)).is_not_null(),
                        sql.lit(value=False),
                    ),
                ).otherwise(sql.coalesce(contains_expr, sql.lit(value=False)))
            )
        return Expr(contains_expr)

    def get(self, index: int) -> Expr:
        """Return the value by index in each list."""
        return Expr(
            self._expr.list.extract(
                sql.lit(index + 1 if index >= 0 else index),
            )
        )

    def min(self) -> Expr:
        """Compute the min value of the lists in the array."""
        return Expr(self._expr.list.min())

    def max(self) -> Expr:
        """Compute the max value of the lists in the array."""
        return Expr(self._expr.list.max())

    def mean(self) -> Expr:
        """Compute the mean value of the lists in the array."""
        return Expr(self._expr.list.avg())

    def median(self) -> Expr:
        """Compute the median value of the lists in the array."""
        return Expr(self._expr.list.median())

    def sum(self) -> Expr:
        """Compute the sum value of the lists in the array."""
        expr_no_nulls = self._expr.list.filter(
            sql.fn_once("_", sql.col("_").is_not_null())
        )
        return Expr(
            sql.when(expr_no_nulls.list.length().eq(sql.lit(0)), sql.lit(0)).otherwise(
                expr_no_nulls.list.sum()
            )
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the expression."""
        sort_direction = "DESC" if descending else "ASC"
        nulls_position = "NULLS LAST" if nulls_last else "NULLS FIRST"
        return Expr(
            self._expr.list.sort(sql.lit(sort_direction), sql.lit(nulls_position))
        )


@dataclass(slots=True)
class ExprStructNameSpace(ExprHandler[sql.SqlExpr]):
    """Struct operations namespace (equivalent to pl.Expr.struct)."""

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name."""
        return Expr(self._expr.struct.extract(sql.lit(name)))
