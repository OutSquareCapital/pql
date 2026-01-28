"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self

import duckdb
import pyochain as pc

from . import datatypes
from ._ast import WindowExpr, iter_to_exprs, to_value

if TYPE_CHECKING:
    from ._ast import IntoExpr


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(duckdb.ColumnExpression(name))

    def __getattr__(self, name: str) -> Expr:
        return self.__call__(name)


col: Col = Col()


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
    def expr(self) -> duckdb.Expression:
        """Get the underlying DuckDB expression."""
        return self._expr

    def __add__(self, other: IntoExpr) -> Self:
        return self.add(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) + self._expr)

    def __sub__(self, other: IntoExpr) -> Self:
        return self.sub(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) - self._expr)

    def __mul__(self, other: IntoExpr) -> Self:
        return self.mul(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) * self._expr)

    def __truediv__(self, other: IntoExpr) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) / self._expr)

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) // self._expr)

    def __mod__(self, other: IntoExpr) -> Self:
        return self.mod(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) % self._expr)

    def __pow__(self, other: IntoExpr) -> Self:
        return self.pow(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) ** self._expr)

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
        return self.__class__(to_value(other) & self._expr)

    def __or__(self, other: IntoExpr) -> Self:
        return self.or_(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self.__class__(to_value(other) | self._expr)

    def __invert__(self) -> Self:
        return self.not_()

    def __hash__(self) -> int:
        return hash(str(self._expr))

    def add(self, other: Any) -> Self:  # noqa: ANN401
        """Add another expression or value."""
        return self.__class__(self._expr + to_value(other))

    def sub(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr - to_value(other))

    def mul(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr * to_value(other))

    def truediv(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr / to_value(other))

    def floordiv(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr // to_value(other))

    def mod(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr % to_value(other))

    def pow(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr ** to_value(other))

    def neg(self) -> Self:
        return self.__class__(-self._expr)

    def abs(self) -> Self:
        return self.__class__(duckdb.FunctionExpression("abs", self._expr))

    def eq(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr == to_value(other))

    def ne(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr != to_value(other))

    def lt(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr < to_value(other))

    def le(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr <= to_value(other))

    def gt(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr > to_value(other))

    def ge(self, other: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr >= to_value(other))

    def and_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr & to_value(others))

    def or_(self, others: Any) -> Self:  # noqa: ANN401
        return self.__class__(self._expr | to_value(others))

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
        return self.__class__(self._expr.isin(*iter_to_exprs(other)))

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

    def round(
        self,
        decimals: int = 0,
        *,
        mode: Literal["half_to_even", "half_away_from_zero"] = "half_to_even",
    ) -> Self:
        """Round to given number of decimal places."""
        match mode:
            case "half_to_even":
                func = "round_even"
            case "half_away_from_zero":
                func = "round"
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
                "log", duckdb.ConstantExpression(base), self._expr
            )
        )

    def log10(self) -> Self:
        """Compute the base 10 logarithm."""
        return self.__class__(duckdb.FunctionExpression("log10", self._expr))

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x."""
        return self.__class__(
            duckdb.FunctionExpression(
                "ln", self._expr.__add__(duckdb.ConstantExpression(1))
            )
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

    def arctan(self) -> Self:
        """Compute the arc tangent."""
        return self.__class__(duckdb.FunctionExpression("atan", self._expr))

    def sinh(self) -> Self:
        """Compute the hyperbolic sine."""
        return self.__class__(
            duckdb.FunctionExpression(
                "/",
                duckdb.FunctionExpression("exp", self._expr).__sub__(
                    duckdb.FunctionExpression("exp", -self._expr)
                ),
                duckdb.ConstantExpression("2"),
            )
        )

    def cosh(self) -> Self:
        """Compute the hyperbolic cosine."""
        return self.__class__(
            duckdb.FunctionExpression(
                "/",
                duckdb.FunctionExpression("exp", self._expr).__add__(
                    duckdb.FunctionExpression("exp", -self._expr)
                ),
                duckdb.ConstantExpression("2"),
            )
        )

    def tanh(self) -> Self:
        """Compute the hyperbolic tangent."""
        exp_x = duckdb.FunctionExpression("exp", self._expr)
        exp_neg_x = duckdb.FunctionExpression("exp", -self._expr)
        return self.__class__(
            (exp_x.__sub__(exp_neg_x)).__truediv__(exp_x.__add__(exp_neg_x))
        )

    def degrees(self) -> Self:
        """Convert radians to degrees."""
        return self.__class__(duckdb.FunctionExpression("degrees", self._expr))

    def radians(self) -> Self:
        """Convert degrees to radians."""
        return self.__class__(duckdb.FunctionExpression("radians", self._expr))

    def sign(self) -> Self:
        """Get the sign of the value."""
        return self.__class__(duckdb.FunctionExpression("sign", self._expr))

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value."""
        return self.__class__(
            WindowExpr(rows_end=pc.Some(0), ignore_nulls=True).call(
                duckdb.FunctionExpression("last_value", self._expr),
            )
        )

    def backward_fill(self) -> Self:
        """Fill null values with the next non-null value."""
        return self.__class__(
            WindowExpr(rows_start=pc.Some(0), ignore_nulls=True).call(
                duckdb.FunctionExpression("first_value", self._expr),
            )
        )

    def interpolate(self) -> Self:
        """Interpolate null values using linear interpolation."""
        last_value = WindowExpr(rows_end=pc.Some(0), ignore_nulls=True).call(
            duckdb.FunctionExpression("last_value", self._expr),
        )
        return self.__class__(
            duckdb.CoalesceOperator(
                self._expr,
                last_value.__add__(
                    WindowExpr(rows_start=pc.Some(0), ignore_nulls=True)
                    .call(duckdb.FunctionExpression("first_value", self._expr))
                    .__sub__(last_value)
                ).__truediv__(duckdb.ConstantExpression(2)),
            )
        )

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

    def fill_nan(self, value: IntoExpr) -> Self:
        """Fill NaN values."""
        return self.__class__(
            duckdb.CaseExpression(
                duckdb.FunctionExpression("isnan", self._expr), to_value(value)
            ).otherwise(self._expr)
        )

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash."""
        return self.__class__(
            duckdb.FunctionExpression(
                "hash", self._expr, duckdb.ConstantExpression(seed)
            )
        )

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values."""
        return self.__class__(
            duckdb.CaseExpression(
                self._expr.__eq__(to_value(old)), to_value(new)
            ).otherwise(self._expr)
        )

    def repeat_by(self, by: Expr | int) -> Self:
        """Repeat values by count, returning a list."""
        return self.__class__(
            duckdb.FunctionExpression(
                "list_transform",
                duckdb.FunctionExpression("range", to_value(by)),
                duckdb.LambdaExpression("_", self._expr),
            )
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        return self.__class__(
            WindowExpr(partition_by=pc.Seq((self._expr,)))
            .call(duckdb.FunctionExpression("count", duckdb.StarExpression()))
            .__gt__(duckdb.ConstantExpression(1)),
        )

    def is_unique(self) -> Self:
        """Check if value is unique."""
        return self.__class__(
            WindowExpr(partition_by=pc.Seq((self._expr,)))
            .call(duckdb.FunctionExpression("count", duckdb.StarExpression()))
            .__eq__(duckdb.ConstantExpression(1)),
        )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self.__class__(
            WindowExpr(partition_by=pc.Seq((self._expr,)))
            .call(duckdb.FunctionExpression("row_number"))
            .__eq__(duckdb.ConstantExpression(1))
        )

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return self.__class__(
            WindowExpr(
                partition_by=pc.Seq((self._expr,)),
                order_by=pc.Seq((self._expr,)),
                descending=pc.Some(value=True),
                nulls_last=pc.Some(value=True),
            )
            .call(duckdb.FunctionExpression("row_number"))
            .__eq__(duckdb.ConstantExpression(1))
        )


@dataclass(slots=True)
class ExprStringNameSpace:
    """String operations namespace (equivalent to pl.Expr.str)."""

    _expr: duckdb.Expression

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return Expr(duckdb.FunctionExpression("upper", self._expr))

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return Expr(duckdb.FunctionExpression("lower", self._expr))

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return Expr(duckdb.FunctionExpression("length", self._expr))

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        """Check if string contains a pattern."""
        match literal:
            case True:
                return Expr(
                    duckdb.FunctionExpression(
                        "contains", self._expr, duckdb.ConstantExpression(pattern)
                    )
                )
            case False:
                return Expr(
                    duckdb.FunctionExpression(
                        "regexp_matches", self._expr, duckdb.ConstantExpression(pattern)
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

    def replace(
        self,
        pattern: str,
        value: str | IntoExpr,
        *,
        literal: bool = False,
        n: int = 1,
    ) -> Expr:
        """Replace first matching substring with a new string value."""
        value_expr = to_value(value)
        pattern_expr = duckdb.ConstantExpression(
            re.escape(pattern) if literal else pattern
        )

        def _replace_once(expr: duckdb.Expression) -> duckdb.Expression:
            return duckdb.FunctionExpression(
                "regexp_replace", expr, pattern_expr, value_expr
            )

        match n:
            case 0:
                return Expr(self._expr)
            case n_val if n_val < 0:
                return Expr(
                    duckdb.FunctionExpression(
                        "regexp_replace",
                        self._expr,
                        pattern_expr,
                        value_expr,
                        duckdb.ConstantExpression("g"),
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
                return Expr(duckdb.FunctionExpression("trim", self._expr))
            case _:
                return Expr(
                    duckdb.FunctionExpression(
                        "trim", self._expr, duckdb.ConstantExpression(characters)
                    )
                )

    def strip_chars_start(self) -> Expr:
        """Strip leading whitespace."""
        return Expr(duckdb.FunctionExpression("ltrim", self._expr))

    def strip_chars_end(self) -> Expr:
        """Strip trailing whitespace."""
        return Expr(duckdb.FunctionExpression("rtrim", self._expr))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        match length:
            case None:
                args = (self._expr, duckdb.ConstantExpression(offset + 1))
            case _:
                args = (
                    self._expr,
                    duckdb.ConstantExpression(offset + 1),
                    duckdb.ConstantExpression(length),
                )
        return Expr(duckdb.FunctionExpression("substring", *args))

    def len_bytes(self) -> Expr:
        """Get the length in bytes."""
        return Expr(
            duckdb.FunctionExpression(
                "octet_length", duckdb.FunctionExpression("encode", self._expr)
            )
        )

    def split(self, by: str) -> Expr:
        """Split string by separator."""
        return Expr(
            duckdb.FunctionExpression(
                "string_split", self._expr, duckdb.ConstantExpression(by)
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
        match literal:
            case False:
                return Expr(
                    duckdb.FunctionExpression(
                        "length",
                        duckdb.FunctionExpression(
                            "regexp_extract_all",
                            self._expr,
                            duckdb.ConstantExpression(pattern),
                        ),
                    )
                )
            case True:
                return Expr(
                    duckdb.FunctionExpression("length", self._expr)
                    .__sub__(
                        duckdb.FunctionExpression(
                            "length",
                            duckdb.FunctionExpression(
                                "replace",
                                self._expr,
                                duckdb.ConstantExpression(pattern),
                                duckdb.ConstantExpression(""),
                            ),
                        )
                    )
                    .__truediv__(duckdb.ConstantExpression(len(pattern)))
                )

    def to_date(self, format: str | None = None) -> Expr:  # noqa: A002
        """Convert string to date."""
        match format:
            case None:
                return Expr(self._expr.cast(datatypes.Date))
            case _:
                return Expr(
                    duckdb.FunctionExpression(
                        "strptime", self._expr, duckdb.ConstantExpression(format)
                    ).cast(datatypes.Date)
                )

    def to_datetime(self, format: str | None = None, *, time_unit: str = "us") -> Expr:  # noqa: A002
        """Convert string to datetime."""
        precision_map = {"ns": "TIMESTAMP_NS", "us": "TIMESTAMP", "ms": "TIMESTAMP_MS"}
        match format:
            case None:
                base_expr = self._expr.cast(precision_map.get(time_unit, "TIMESTAMP"))
            case _:
                base_expr = duckdb.FunctionExpression(
                    "strptime", self._expr, duckdb.ConstantExpression(format)
                )
        return Expr(base_expr)

    def to_time(self, format: str | None = None) -> Expr:  # noqa: A002
        """Convert string to time."""
        match format:
            case None:
                return Expr(self._expr.cast(datatypes.Time))
            case _:
                return Expr(
                    duckdb.FunctionExpression(
                        "strptime", self._expr, duckdb.ConstantExpression(format)
                    ).cast(datatypes.Time)
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

    def replace_all(
        self, pattern: str, value: IntoExpr, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences."""
        value_expr = to_value(value)
        match literal:
            case True:
                return Expr(
                    duckdb.FunctionExpression(
                        "replace",
                        self._expr,
                        duckdb.ConstantExpression(pattern),
                        value_expr,
                    )
                )
            case False:
                return Expr(
                    duckdb.FunctionExpression(
                        "regexp_replace",
                        self._expr,
                        duckdb.ConstantExpression(pattern),
                        value_expr,
                        duckdb.ConstantExpression("g"),
                    )
                )

    def to_titlecase(self) -> Expr:
        """Convert to title case."""
        elem = duckdb.ColumnExpression("_")
        return Expr(
            duckdb.FunctionExpression(
                "list_aggregate",
                duckdb.FunctionExpression(
                    "list_transform",
                    duckdb.FunctionExpression(
                        "regexp_extract_all",
                        duckdb.FunctionExpression("lower", self._expr),
                        duckdb.ConstantExpression(r"[a-z]*[^a-z]*"),
                    ),
                    duckdb.LambdaExpression(
                        "_",
                        duckdb.FunctionExpression(
                            "concat",
                            duckdb.FunctionExpression(
                                "upper",
                                duckdb.FunctionExpression(
                                    "array_extract", elem, duckdb.ConstantExpression(1)
                                ),
                            ),
                            duckdb.FunctionExpression(
                                "substring", elem, duckdb.ConstantExpression(2)
                            ),
                        ),
                    ),
                ),
                duckdb.ConstantExpression("string_agg"),
                duckdb.ConstantExpression(""),
            )
        )


@dataclass(slots=True)
class ExprListNameSpace:
    """List operations namespace (equivalent to pl.Expr.list)."""

    _expr: duckdb.Expression

    def len(self) -> Expr:
        """Return the number of elements in each list."""
        return Expr(duckdb.FunctionExpression("len", self._expr))

    def unique(self) -> Expr:
        """Return unique values in each list."""
        distinct_expr = duckdb.FunctionExpression("list_distinct", self._expr)
        return Expr(
            duckdb.CaseExpression(
                duckdb.FunctionExpression(
                    "array_position",
                    self._expr,
                    duckdb.ConstantExpression(None),
                ).isnotnull(),
                duckdb.FunctionExpression(
                    "list_append",
                    distinct_expr,
                    duckdb.ConstantExpression(None),
                ),
            ).otherwise(distinct_expr)
        )

    def contains(self, item: IntoExpr, *, nulls_equal: bool = True) -> Expr:
        """Check if sublists contain the given item."""
        item_expr = to_value(item)
        contains_expr = duckdb.FunctionExpression(
            "list_contains", self._expr, item_expr
        )
        if nulls_equal:
            false_expr = duckdb.SQLExpression("false")
            null_in_list = duckdb.FunctionExpression(
                "array_position",
                self._expr,
                duckdb.ConstantExpression(None),
            ).isnotnull()
            return Expr(
                duckdb.CaseExpression(
                    item_expr.isnull(),
                    duckdb.CoalesceOperator(null_in_list, false_expr),
                ).otherwise(duckdb.CoalesceOperator(contains_expr, false_expr))
            )
        return Expr(contains_expr)

    def get(self, index: int) -> Expr:
        """Return the value by index in each list."""
        return Expr(
            duckdb.FunctionExpression(
                "list_extract",
                self._expr,
                duckdb.ConstantExpression(index + 1 if index >= 0 else index),
            )
        )

    def min(self) -> Expr:
        """Compute the min value of the lists in the array."""
        return Expr(duckdb.FunctionExpression("list_min", self._expr))

    def max(self) -> Expr:
        """Compute the max value of the lists in the array."""
        return Expr(duckdb.FunctionExpression("list_max", self._expr))

    def mean(self) -> Expr:
        """Compute the mean value of the lists in the array."""
        return Expr(duckdb.FunctionExpression("list_avg", self._expr))

    def median(self) -> Expr:
        """Compute the median value of the lists in the array."""
        return Expr(duckdb.FunctionExpression("list_median", self._expr))

    def sum(self) -> Expr:
        """Compute the sum value of the lists in the array."""
        elem = duckdb.ColumnExpression("_")
        expr_no_nulls = duckdb.FunctionExpression(
            "list_filter",
            self._expr,
            duckdb.LambdaExpression("_", elem.isnotnull()),
        )
        expr_sum = duckdb.FunctionExpression("list_sum", expr_no_nulls)
        return Expr(
            duckdb.CaseExpression(
                duckdb.FunctionExpression(
                    "array_length",
                    expr_no_nulls,
                ).__eq__(duckdb.ConstantExpression(0)),
                duckdb.ConstantExpression(0),
            ).otherwise(expr_sum)
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the expression."""
        sort_direction = "DESC" if descending else "ASC"
        nulls_position = "NULLS LAST" if nulls_last else "NULLS FIRST"
        return Expr(
            duckdb.FunctionExpression(
                "list_sort",
                self._expr,
                duckdb.ConstantExpression(sort_direction),
                duckdb.ConstantExpression(nulls_position),
            )
        )


@dataclass(slots=True)
class ExprStructNameSpace:
    """Struct operations namespace (equivalent to pl.Expr.struct)."""

    _expr: duckdb.Expression

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name."""
        return Expr(
            duckdb.FunctionExpression(
                "struct_extract", self._expr, duckdb.ConstantExpression(name)
            )
        )
