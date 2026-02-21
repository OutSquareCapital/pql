"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

import pyochain as pc

from . import sql
from .sql import CoreHandler, SqlExpr, into_expr, try_flatten, try_iter

if TYPE_CHECKING:
    from ._datatypes import DataType
    from .sql import IntoExpr

RoundMode = Literal["half_to_even", "half_away_from_zero"]
ClosedInterval = Literal["both", "left", "right", "none"]
RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
FillNullStrategy = Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
RollingInterpolationMethod = Literal["nearest", "higher", "lower", "midpoint", "linear"]


class Col:
    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name))

    def __getattr__(self, name: str) -> Expr:
        return self.__call__(name)


def lit(value: IntoExpr) -> Expr:
    """Create a literal expression."""
    return Expr(sql.lit(value))


col: Col = Col()


def all() -> Expr:
    """Create an expression representing all columns (equivalent to pl.all())."""
    return Expr(sql.all())


def element() -> Expr:
    """Alias for an element being evaluated in a list context."""
    return Expr(sql.col("_"))


@dataclass(slots=True)
class Expr(CoreHandler[SqlExpr]):
    """Expression wrapper providing Polars-like API over DuckDB expressions."""

    _is_scalar_like: bool = False
    _has_window: bool = False
    _is_unique_projection: bool = False

    def _new(
        self,
        value: SqlExpr,
        *,
        is_scalar_like: bool | None = None,
        is_unique_projection: bool | None = None,
    ) -> Self:
        return self.__class__(
            value,
            pc.Option(is_scalar_like).unwrap_or(self._is_scalar_like),
            _has_window=self._has_window,
            _is_unique_projection=pc.Option(is_unique_projection).unwrap_or(
                self._is_unique_projection
            ),
        )

    def _new_window(self, expr: SqlExpr, *, is_scalar_like: bool | None = None) -> Self:
        return self.__class__(
            expr,
            pc.Option(is_scalar_like).unwrap_or(self._is_scalar_like),
            _has_window=True,
            _is_unique_projection=self._is_unique_projection,
        )

    def _reversed(self, expr: SqlExpr, *, reverse: bool = False) -> Self:
        match reverse:
            case True:
                return self._new_window(expr.over(rows_start=0))
            case False:
                return self._new_window(expr.over(rows_end=0))

    def _rolling_agg(
        self,
        agg: Callable[[SqlExpr], SqlExpr],
        window_size: int,
        min_samples: int | None,
        *,
        center: bool,
    ) -> Self:
        def _rolling_bounds() -> tuple[int, int]:
            match center:
                case True:
                    left = window_size // 2
                    right = window_size - left - 1
                    return (-left, right)
                case False:
                    return (-(window_size - 1), 0)

        rows_start, rows_end = _rolling_bounds()
        return self._new_window(
            sql.when(
                self.inner()
                .count()
                .over(rows_start=rows_start, rows_end=rows_end)
                .ge(sql.lit(pc.Option(min_samples).unwrap_or(window_size))),
                agg(self.inner()).over(rows_start=rows_start, rows_end=rows_end),
            ).otherwise(sql.lit(None))
        )

    def __repr__(self) -> str:
        return f"Expr({self.inner()})"

    @property
    def str(self) -> ExprStringNameSpace:
        """Access string operations."""
        return ExprStringNameSpace(self.inner())

    @property
    def list(self) -> ExprListNameSpace:
        """Access list operations."""
        return ExprListNameSpace(self.inner())

    @property
    def struct(self) -> ExprStructNameSpace:
        """Access struct operations."""
        return ExprStructNameSpace(self.inner())

    @property
    def expr(self) -> SqlExpr:  # pragma: no cover
        """Get the underlying DuckDB expression."""
        return self.inner()

    def __add__(self, other: IntoExpr) -> Self:
        return self.add(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self._new(into_expr(other).radd(self.inner()))

    def __sub__(self, other: IntoExpr) -> Self:
        return self.sub(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rsub(into_expr(other)))

    def __mul__(self, other: IntoExpr) -> Self:
        return self.mul(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self._new(into_expr(other).rmul(self.inner()))

    def __truediv__(self, other: IntoExpr) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rtruediv(into_expr(other)))

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rfloordiv(into_expr(other)))

    def __mod__(self, other: IntoExpr) -> Self:
        return self.mod(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rmod(into_expr(other)))

    def __pow__(self, other: IntoExpr) -> Self:
        return self.pow(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rpow(into_expr(other)))

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
        return self._new(into_expr(other).rand(self.inner()))

    def __or__(self, other: IntoExpr) -> Self:
        return self.or_(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self._new(into_expr(other).ror(self.inner()))

    def __invert__(self) -> Self:
        return self.not_()

    def __hash__(self) -> int:
        return hash(str(self.inner()))

    def add(self, other: Any) -> Self:  # noqa: ANN401
        """Add another expression or value."""
        return self._new(self.inner().add(into_expr(other)))

    def sub(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().sub(into_expr(other)))

    def mul(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().mul(into_expr(other)))

    def truediv(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().truediv(into_expr(other)))

    def floordiv(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().floordiv(into_expr(other)))

    def mod(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().mod(into_expr(other)))

    def pow(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().pow(into_expr(other)))

    def neg(self) -> Self:
        return self._new(self.inner().neg())

    def abs(self) -> Self:
        return self._new(self.inner().abs())

    def eq(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().eq(into_expr(other)))

    def ne(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().ne(into_expr(other)))

    def lt(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().lt(into_expr(other)))

    def le(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().le(into_expr(other)))

    def gt(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().gt(into_expr(other)))

    def ge(self, other: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().ge(into_expr(other)))

    def and_(self, others: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().and_(into_expr(others)))

    def or_(self, others: Any) -> Self:  # noqa: ANN401
        return self._new(self.inner().or_(into_expr(others)))

    def not_(self) -> Self:
        return self._new(self.inner().not_())

    def alias(self, name: str) -> Self:
        """Rename the expression."""
        return self._new(self.inner().alias(name))

    def is_null(self) -> Self:
        """Check if the expression is NULL."""
        return self._new(self.inner().is_null())

    def is_not_null(self) -> Self:
        """Check if the expression is not NULL."""
        return self._new(self.inner().is_not_null())

    def cast(self, dtype: DataType) -> Self:
        """Cast to a different data type."""
        return self._new(self.inner().cast(dtype.raw.to_duckdb()))

    def is_in(self, other: Collection[IntoExpr] | IntoExpr) -> Self:
        """Check if value is in an iterable of values."""
        return self._new(self.inner().is_in(*try_iter(other).map(into_expr)))

    def shift(self, n: int = 1) -> Self:
        match n:
            case 0:
                return self
            case n_val if n_val > 0:
                expr = self.inner().lag(sql.lit(n_val))
            case _:
                expr = self.inner().lead(sql.lit(-n))
        return self._new_window(expr.over())

    def diff(self) -> Self:
        return self.sub(self.shift())

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        lower_expr = into_expr(lower_bound)
        upper_expr = into_expr(upper_bound)
        match closed:
            case "both":
                return self.ge(lower_expr).and_(self.le(upper_expr))
            case "left":
                return self.ge(lower_expr).and_(self.lt(upper_expr))
            case "right":
                return self.gt(lower_expr).and_(self.le(upper_expr))
            case "none":
                return self.gt(lower_expr).and_(self.lt(upper_expr))

    def clip(
        self, lower_bound: IntoExpr | None = None, upper_bound: IntoExpr | None = None
    ) -> Self:
        match (lower_bound, upper_bound):
            case (None, None):
                return self
            case (None, upper):
                return self._new(self.inner().least(into_expr(upper)))
            case (lower, None):
                return self._new(self.inner().greatest(into_expr(lower)))
            case (lower, upper):
                return self._new(
                    self.inner().greatest(into_expr(lower)).least(into_expr(upper))
                )

    def count(self) -> Self:
        """Count the number of values."""
        return self._new(self.inner().count(), is_scalar_like=True)

    def len(self) -> Self:
        """Get the number of rows in context (including nulls)."""
        return self._new(self.inner().is_null().count(), is_scalar_like=True)

    def sum(self) -> Self:
        """Compute the sum."""
        return self._new(self.inner().sum(), is_scalar_like=True)

    def mean(self) -> Self:
        """Compute the mean."""
        return self._new(self.inner().mean(), is_scalar_like=True)

    def median(self) -> Self:
        """Compute the median."""
        return self._new(self.inner().median(), is_scalar_like=True)

    def min(self) -> Self:
        """Compute the minimum."""
        return self._new(self.inner().min(), is_scalar_like=True)

    def max(self) -> Self:
        """Compute the maximum."""
        return self._new(self.inner().max(), is_scalar_like=True)

    def first(self, *, ignore_nulls: bool = False) -> Self:
        """Get first value."""
        match ignore_nulls:
            case True:
                return self._new(self.inner().any_value(), is_scalar_like=True)
            case False:
                return self._new(self.inner().first(), is_scalar_like=True)

    def last(self, *, ignore_nulls: bool = False) -> Self:
        """Get last value."""
        match ignore_nulls:
            case True:
                return self._new(
                    self.filter(self.is_not_null()).inner().last(), is_scalar_like=True
                )
            case False:
                return self._new(self.inner().last(), is_scalar_like=True)

    def mode(self) -> Self:
        """Compute mode."""
        return self._new(self.inner().mode(), is_scalar_like=True)

    def unique(self) -> Self:
        """Get unique values."""
        return self._new(self.inner(), is_unique_projection=True)

    def is_close(
        self,
        other: IntoExpr,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-5,
        *,
        nans_equal: bool = False,
    ) -> Self:
        """Check if two floating point values are close."""
        other_expr = into_expr(other)
        threshold = sql.lit(abs_tol).add(sql.lit(rel_tol).mul(other_expr.abs()))
        close = self.inner().sub(other_expr).abs().le(threshold)
        match nans_equal:
            case False:
                return self._new(close)
            case True:
                return self._new(
                    sql.when(
                        self.inner().isnan().and_(other_expr.isnan()),
                        sql.lit(value=True),
                    ).otherwise(close)
                )

    def rolling_mean(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling mean."""
        return self._rolling_agg(
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            agg=lambda expr: expr.mean(),
        )

    def rolling_sum(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling sum."""
        return self._rolling_agg(
            lambda expr: expr.sum(), window_size, min_samples, center=center
        )

    def rolling_std(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Compute rolling std."""
        return self._rolling_agg(
            lambda expr: expr.stddev_pop() if ddof == 0 else expr.stddev_samp(),
            window_size,
            min_samples,
            center=center,
        )

    def rolling_var(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Compute rolling variance."""
        return self._rolling_agg(
            lambda expr: expr.var_pop() if ddof == 0 else expr.var_samp(),
            window_size,
            min_samples,
            center=center,
        )

    def std(self, ddof: int = 1) -> Self:
        """Compute the standard deviation."""
        match ddof:
            case 0:
                expr = self.inner().stddev_pop()
            case _:
                expr = self.inner().stddev_samp()
        return self._new(expr, is_scalar_like=True)

    def var(self, ddof: int = 1) -> Self:
        """Compute the variance."""
        match ddof:
            case 0:
                expr = self.inner().var_pop()
            case _:
                expr = self.inner().var_samp()
        return self._new(expr, is_scalar_like=True)

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
        base = self.inner().kurtosis_pop() if bias else self.inner().kurtosis()
        match fisher:
            case True:
                return self._new(base, is_scalar_like=True)
            case False:
                return self._new(base.add(sql.lit(3)), is_scalar_like=True)

    def skew(self, *, bias: bool = True) -> Self:
        adjusted = self.inner().skewness()
        match bias:
            case False:
                return self._new(adjusted, is_scalar_like=True)
            case True:
                n = self.inner().count()
                factor = n.sub(sql.lit(2)).truediv(n.mul(n.sub(sql.lit(1))).sqrt())
                return self._new(adjusted.mul(factor), is_scalar_like=True)

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> Self:
        match interpolation:
            case "linear" | "midpoint":
                expr = self.inner().quantile_cont(sql.lit(quantile))
            case _:
                expr = self.inner().quantile(sql.lit(quantile))
        return self._new(expr, is_scalar_like=True)

    def all(self) -> Self:
        """Return whether all values are true."""
        return self._new(self.inner().bool_and(), is_scalar_like=True)

    def any(self) -> Self:
        """Return whether any value is true."""
        return self._new(self.inner().bool_or(), is_scalar_like=True)

    def n_unique(self) -> Self:
        """Count distinct values."""
        return self._new(
            self.inner().implode().list.distinct().list.length(), is_scalar_like=True
        )

    def null_count(self) -> Self:
        """Count null values."""
        return self.len().sub(self.count())

    def rank(self, method: RankMethod = "average", *, descending: bool = False) -> Self:
        """Compute rank values."""
        base_rank = (
            self.inner().rank().over(order_by=self.inner(), descending=descending)
        )
        peer_count = sql.all().count().over(self.inner())
        match method:
            case "average":
                max_rank = base_rank.add(peer_count).sub(sql.lit(1))
                expr = base_rank.add(max_rank).truediv(sql.lit(2))
            case "min":
                expr = base_rank
            case "max":
                expr = base_rank.add(peer_count).sub(sql.lit(1))
            case "dense":
                expr = (
                    self.inner()
                    .dense_rank()
                    .over(order_by=self.inner(), descending=descending)
                )
            case "ordinal":
                expr = (
                    self.inner()
                    .row_number()
                    .over(order_by=self.inner(), descending=descending)
                )
        return self._new_window(expr)

    def cum_count(self, *, reverse: bool = False) -> Self:
        """Cumulative non-null count."""
        return self._reversed(self.inner().count(), reverse=reverse)

    def cum_sum(self, *, reverse: bool = False) -> Self:
        """Cumulative sum."""
        return self._reversed(self.inner().sum(), reverse=reverse)

    def cum_prod(self, *, reverse: bool = False) -> Self:
        """Cumulative product."""
        return self._reversed(self.inner().product(), reverse=reverse)

    def cum_min(self, *, reverse: bool = False) -> Self:
        """Cumulative minimum."""
        return self._reversed(self.inner().min(), reverse=reverse)

    def cum_max(self, *, reverse: bool = False) -> Self:
        """Cumulative maximum."""
        return self._reversed(self.inner().max(), reverse=reverse)

    def over(
        self,
        partition_by: IntoExpr | Iterable[IntoExpr] | None,
        *more_exprs: IntoExpr,
        order_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> Self:
        partition_exprs = (
            try_iter(partition_by)
            .chain(more_exprs)
            .map(lambda x: into_expr(x, as_col=True))
        )
        expr = (
            pc.Option(order_by)
            .map(
                lambda value: (
                    try_iter(value).map(lambda x: into_expr(x, as_col=True)).collect()
                )
            )
            .map(lambda x: self.inner().over(partition_exprs, x))
            .unwrap_or(self.inner().over(partition_exprs))
        )
        return self._new_window(expr)

    def filter(self, *predicates: Any) -> Self:  # noqa: ANN401
        cond = (
            try_flatten(predicates)
            .map(lambda v: into_expr(v, as_col=True))
            .fold(sql.lit(value=True), lambda acc, pred: acc.and_(pred))
        )
        return self._new(sql.when(cond, self.inner()).otherwise(sql.lit(None)))

    def drop_nulls(self) -> Self:
        return self.filter(self.is_not_null())

    def floor(self) -> Self:
        """Round down to the nearest integer."""
        return self._new(self.inner().floor())

    def ceil(self) -> Self:
        """Round up to the nearest integer."""
        return self._new(self.inner().ceil())

    def round(self, decimals: int = 0, *, mode: RoundMode = "half_to_even") -> Self:
        """Round to given number of decimal places."""
        match mode:
            case "half_to_even":
                rounded = self.inner().round_even(sql.lit(decimals))
            case "half_away_from_zero":
                rounded = self.inner().round(sql.lit(decimals))
        return self._new(rounded)

    def sqrt(self) -> Self:
        """Compute the square root."""
        return self._new(self.inner().sqrt())

    def cbrt(self) -> Self:
        """Compute the cube root."""
        return self._new(self.inner().cbrt())

    def log(self, base: float = 2.718281828459045) -> Self:
        """Compute the logarithm."""
        return self._new(self.inner().log(sql.lit(base)))

    def log10(self) -> Self:
        """Compute the base 10 logarithm."""
        return self._new(self.inner().log10())

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x."""
        return self._new(self.inner().add(sql.lit(1)).ln())

    def exp(self) -> Self:
        """Compute the exponential."""
        return self._new(self.inner().exp())

    def sin(self) -> Self:
        """Compute the sine."""
        return self._new(self.inner().sin())

    def cos(self) -> Self:
        """Compute the cosine."""
        return self._new(self.inner().cos())

    def tan(self) -> Self:
        """Compute the tangent."""
        return self._new(self.inner().tan())

    def arctan(self) -> Self:
        """Compute the arc tangent."""
        return self._new(self.inner().atan())

    def sinh(self) -> Self:
        """Compute the hyperbolic sine."""
        return self._new(self.inner().sinh())

    def cosh(self) -> Self:
        """Compute the hyperbolic cosine."""
        return self._new(self.inner().cosh())

    def tanh(self) -> Self:
        """Compute the hyperbolic tangent."""
        return self._new(self.inner().tanh())

    def degrees(self) -> Self:
        """Convert radians to degrees."""
        return self._new(self.inner().degrees())

    def radians(self) -> Self:
        """Convert degrees to radians."""
        return self._new(self.inner().radians())

    def sign(self) -> Self:
        """Get the sign of the value."""
        return self._new(self.inner().sign())

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value."""
        return self._new(self.inner().last_value().over(rows_end=0, ignore_nulls=True))

    def backward_fill(self) -> Self:
        """Fill null values with the next non-null value."""
        return self._new(self.inner().any_value().over(rows_start=0))

    def is_nan(self) -> Self:
        """Check if value is NaN."""
        return self._new(self.inner().isnan())

    def is_not_nan(self) -> Self:
        """Check if value is not NaN."""
        return self._new(self.inner().isnan().not_())

    def is_finite(self) -> Self:
        """Check if value is finite."""
        return self._new(self.inner().isfinite())

    def is_infinite(self) -> Self:
        """Check if value is infinite."""
        return self._new(self.inner().isinf())

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values."""
        return self._new(sql.when(self.inner().isnan(), value).otherwise(self.inner()))

    def fill_null(  # noqa: PLR0911,PLR0912,C901
        self,
        value: IntoExpr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        match (pc.Option(value), pc.Option(strategy)):
            case (pc.Some(val), pc.NONE):
                return self._new(sql.coalesce(self.inner(), val))
            case (_, pc.Some("forward") | pc.Some("backward") as strat):
                match pc.Option(limit):
                    case pc.Some(lim) if lim <= 0:
                        return self
                    case pc.Some(lim):
                        return self._new(
                            pc.Iter(range(1, lim + 1))
                            .map(
                                lambda offset: (
                                    self.shift(offset).inner()
                                    if strategy == "forward"
                                    else self.shift(-offset).inner()
                                )
                            )
                            .insert(self.inner())
                            .reduce(sql.coalesce)
                        )
                    case _:
                        match strat:
                            case pc.Some("forward"):
                                return self.forward_fill()
                            case _:
                                return self.backward_fill()
            case (_, pc.Some("min")):
                return self._new(sql.coalesce(self.inner(), self.inner().min().over()))
            case (_, pc.Some("max")):
                return self._new(sql.coalesce(self.inner(), self.inner().max().over()))
            case (_, pc.Some("mean")):
                return self._new(sql.coalesce(self.inner(), self.inner().mean().over()))
            case (_, pc.Some("zero")):
                return self._new(sql.coalesce(self.inner(), sql.lit(0)))
            case (_, pc.Some("one")):
                return self._new(sql.coalesce(self.inner(), sql.lit(1)))
            case _:
                msg = "must specify either a fill `value` or `strategy`"
                raise ValueError(msg)

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash."""
        return self._new(self.inner().str.hash(sql.lit(seed)))

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values."""
        return self._new(
            sql.when(self.inner().eq(into_expr(old)), new).otherwise(self.inner())
        )

    def repeat_by(self, by: Expr | int) -> Self:
        """Repeat values by count, returning a list."""
        return self._new(
            into_expr(by).list.range().list.transform(sql.fn_once("_", self.inner()))
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        return self._new(sql.all().count().over(self.inner()).gt(sql.lit(1)))

    def is_unique(self) -> Self:
        """Check if value is unique."""
        return self._new(sql.all().count().over(self.inner()).eq(sql.lit(1)))

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self._new(self.inner().row_number().over(self.inner()).eq(sql.lit(1)))

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return self._new(
            self.inner()
            .row_number()
            .over(self.inner(), self.inner(), descending=True, nulls_last=True)
            .eq(sql.lit(1))
        )


@dataclass(slots=True)
class ExprStringNameSpace(CoreHandler[SqlExpr]):
    """String operations namespace (equivalent to pl.Expr.str)."""

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return Expr(self.inner().str.upper())

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return Expr(self.inner().str.lower())

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return Expr(self.inner().str.length())

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        """Check if string contains a pattern."""
        match literal:
            case True:
                return Expr(self.inner().str.contains(sql.lit(pattern)))
            case False:
                return Expr(self.inner().re.matches(sql.lit(pattern)))

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return Expr(self.inner().str.starts_with(sql.lit(prefix)))

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return Expr(self.inner().str.ends_with(sql.lit(suffix)))

    def replace(
        self, pattern: str, value: str | IntoExpr, *, literal: bool = False, n: int = 1
    ) -> Expr:
        """Replace first matching substring with a new string value."""
        value_expr = into_expr(value)
        pattern_expr = sql.lit(re.escape(pattern) if literal else pattern)

        def _replace_once(expr: SqlExpr) -> SqlExpr:
            return expr.str.replace(pattern_expr, value_expr)

        match n:
            case 0:
                return Expr(self.inner())
            case n_val if n_val < 0:
                return Expr(
                    self.inner().re.replace(pattern_expr, value_expr, sql.lit("g"))
                )
            case _:
                return Expr(
                    pc.Iter(range(n)).fold(
                        self.inner(), lambda acc, _: _replace_once(acc)
                    )
                )

    def strip_chars(self, characters: str | None = None) -> Expr:
        """Strip leading and trailing characters."""
        match characters:
            case None:
                return Expr(self.inner().str.trim())
            case _:
                return Expr(self.inner().str.trim(sql.lit(characters)))

    def strip_chars_start(self, characters: IntoExpr = None) -> Expr:
        """Strip leading characters."""
        match characters:
            case None:
                return Expr(self.inner().str.ltrim())
            case _:
                return Expr(self.inner().str.ltrim(into_expr(characters)))

    def strip_chars_end(self, characters: IntoExpr = None) -> Expr:
        """Strip trailing characters."""
        match characters:
            case None:
                return Expr(self.inner().str.rtrim())
            case _:
                return Expr(self.inner().str.rtrim(into_expr(characters)))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        return Expr(self.inner().str.substring(sql.lit(offset + 1), length))

    def len_bytes(self) -> Expr:
        """Get the length in bytes."""
        return Expr(self.inner().encode().octet_length())

    def split(self, by: str) -> Expr:
        """Split string by separator."""
        return Expr(self.inner().str.split(sql.lit(by)))

    def extract_all(self, pattern: str | Expr) -> Expr:
        """Extract all regex matches."""
        return Expr(self.inner().re.extract_all(into_expr(pattern)))

    def count_matches(self, pattern: str | Expr, *, literal: bool = False) -> Expr:
        """Count pattern matches."""
        pattern_expr = into_expr(pattern)
        match literal:
            case False:
                return Expr(
                    self.inner().re.extract_all(pattern_expr).list.len(),
                )
            case True:
                return Expr(
                    self.inner()
                    .str.length()
                    .sub(
                        self.inner().str.replace(pattern_expr, sql.lit("")).str.length()
                    )
                    .truediv(pattern_expr.str.length())
                )

    def strip_prefix(self, prefix: IntoExpr) -> Expr:
        """Strip prefix from string."""
        match prefix:
            case str() as prefix_str:
                return Expr(
                    self.inner().re.replace(
                        sql.lit(f"^{re.escape(prefix_str)}"), sql.lit("")
                    )
                )
            case _:
                prefix_expr = into_expr(prefix)
                return Expr(
                    sql.when(
                        self.inner().str.starts_with(prefix_expr),
                        self.inner().str.substring(
                            prefix_expr.str.length().add(sql.lit(1))
                        ),
                    ).otherwise(self.inner())
                )

    def strip_suffix(self, suffix: IntoExpr) -> Expr:
        """Strip suffix from string."""
        match suffix:
            case str() as suffix_str:
                return Expr(
                    self.inner().re.replace(
                        sql.lit(f"{re.escape(suffix_str)}$"), sql.lit("")
                    )
                )
            case _:
                suffix_expr = into_expr(suffix)
                return Expr(
                    sql.when(
                        self.inner().str.ends_with(suffix_expr),
                        self.inner().str.substring(
                            sql.lit(1),
                            self.inner().str.length().sub(suffix_expr.str.length()),
                        ),
                    ).otherwise(self.inner())
                )

    def head(self, n: int) -> Expr:
        """Get first n characters."""
        return Expr(self.inner().str.left(sql.lit(n)))

    def tail(self, n: int) -> Expr:
        """Get last n characters."""
        return Expr(self.inner().str.right(sql.lit(n)))

    def reverse(self) -> Expr:
        """Reverse the string."""
        return Expr(self.inner().str.reverse())

    def pad_start(self, length: int, fill_char: str = " ") -> Expr:
        return Expr(self.inner().str.lpad(sql.lit(length), sql.lit(fill_char)))

    def pad_end(self, length: int, fill_char: str = " ") -> Expr:
        return Expr(self.inner().str.rpad(sql.lit(length), sql.lit(fill_char)))

    def zfill(self, width: int) -> Expr:
        return Expr(self.inner().str.lpad(sql.lit(width), sql.lit("0")))

    def replace_all(
        self, pattern: str, value: IntoExpr, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences."""
        value_expr = into_expr(value)
        match literal:
            case True:
                return Expr(self.inner().str.replace(sql.lit(pattern), value_expr))
            case False:
                return Expr(
                    self.inner().re.replace(sql.lit(pattern), value_expr, sql.lit("g"))
                )

    def to_titlecase(self) -> Expr:
        """Convert to title case."""
        elem = sql.col("_")
        lambda_expr = sql.fn_once(
            "_",
            elem.list.extract(sql.lit(1))
            .str.upper()
            .str.concat(elem.str.substring(sql.lit(2))),
        )
        return Expr(
            self.inner()
            .str.lower()
            .re.extract_all(sql.lit(r"[a-z]*[^a-z]*"))
            .list.transform(lambda_expr)
            .list.aggregate(sql.lit("string_agg"), sql.lit(""))
        )


@dataclass(slots=True)
class ExprListNameSpace(CoreHandler[sql.SqlExpr]):
    """List operations namespace (equivalent to pl.Expr.list)."""

    # TODO: reduce, agg, filter

    def eval(self, expr: Expr) -> Expr:
        """Run an expression against each list element."""
        return Expr(self.inner().list.transform(sql.fn_once("_", expr.inner())))

    def len(self) -> Expr:
        """Return the number of elements in each list."""
        return Expr(self.inner().list.length())

    def unique(self) -> Expr:
        """Return unique values in each list."""
        return Expr(self.inner().list.distinct())

    def contains(self, item: IntoExpr, *, nulls_equal: bool = True) -> Expr:
        """Check if sublists contain the given item."""
        item_expr = into_expr(item)
        contains_expr = self.inner().list.contains(item_expr)
        if nulls_equal:
            return Expr(
                sql.when(
                    item_expr.is_null(),
                    sql.coalesce(
                        self.inner().list.position(sql.lit(None)).is_not_null(),
                        sql.lit(value=False),
                    ),
                ).otherwise(sql.coalesce(contains_expr, sql.lit(value=False)))
            )
        return Expr(contains_expr)

    def get(self, index: int) -> Expr:
        """Return the value by index in each list."""
        return Expr(
            self.inner().list.extract(
                sql.lit(index + 1 if index >= 0 else index),
            )
        )

    def min(self) -> Expr:
        """Compute the min value of the lists in the array."""
        return Expr(self.inner().list.min())

    def max(self) -> Expr:
        """Compute the max value of the lists in the array."""
        return Expr(self.inner().list.max())

    def mean(self) -> Expr:
        """Compute the mean value of the lists in the array."""
        return Expr(self.inner().list.avg())

    def median(self) -> Expr:
        """Compute the median value of the lists in the array."""
        return Expr(self.inner().list.median())

    def sum(self) -> Expr:
        """Compute the sum value of the lists in the array."""
        return Expr(self.inner().list.sum())

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the expression."""
        sort_direction = "DESC" if descending else "ASC"
        nulls_position = "NULLS LAST" if nulls_last else "NULLS FIRST"
        return Expr(
            self.inner().list.sort(sql.lit(sort_direction), sql.lit(nulls_position))
        )


@dataclass(slots=True)
class ExprStructNameSpace(CoreHandler[sql.SqlExpr]):
    """Struct operations namespace (equivalent to pl.Expr.struct)."""

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name."""
        return Expr(self.inner().struct.extract(sql.lit(name)))
