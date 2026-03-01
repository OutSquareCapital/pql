"""Expression wrapper providing Polars-like API over DuckDB native expressions."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Self

import pyochain as pc

from . import _datatypes as dt, sql  # pyright: ignore[reportPrivateUsage]
from ._args_iter import try_chain, try_iter
from ._computations import fill_nulls, round, shift

if TYPE_CHECKING:
    from ._datatypes import DataType
    from ._typing import (
        ClosedInterval,
        FillNullStrategy,
        RankMethod,
        RollingInterpolationMethod,
        RoundMode,
        TransferEncoding,
    )
    from .sql.typing import IntoExpr, IntoExprColumn

_NONE = sql.lit(None)
_EMPTY_STR = sql.lit("")


class RollingBounds(NamedTuple):
    start: int
    end: int

    @classmethod
    def new(cls, window_size: int, *, center: bool) -> Self:
        match center:
            case True:
                left = window_size // 2
                right = window_size - left - 1
                return cls(-left, right)
            case False:
                return cls(-(window_size - 1), 0)


@dataclass(slots=True)
class ExprMeta:
    """Metadata for expressions, used for tracking properties that affect query generation."""

    root_name: str
    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)
    is_scalar_like: bool = False
    has_window: bool = False
    is_unique_projection: bool = False
    is_multi: bool = False

    @classmethod
    def __from_expr__(cls, expr: sql.SqlExpr) -> Self:
        return cls(expr.inner().get_name())

    @property
    def output_name(self) -> str:
        match self.alias_name:
            case pc.Some(alias_fn):
                return alias_fn(self.root_name)
            case _:
                return self.root_name

    @property
    def is_scalar_select(self) -> bool:
        return self.is_scalar_like and not self.has_window

    def from_projection(self, output_name: str) -> Self:
        return replace(self, root_name=output_name, alias_name=pc.NONE, is_multi=False)

    def resolve_output_names(
        self, base_names: pc.Seq[str], forced_name: pc.Option[str]
    ) -> pc.Seq[str]:
        match forced_name:
            case pc.Some(name):
                return pc.Seq((name,))
            case _:
                match self.alias_name:
                    case pc.Some(alias_fn):
                        return base_names.iter().map(alias_fn).collect()
                    case _:
                        return base_names

    def with_alias_mapper(self, mapper: Callable[[str], str]) -> Self:
        match self.alias_name:
            case pc.Some(current):

                def _composed(name: str) -> str:
                    return mapper(current(name))

                composed = _composed
            case _:

                def _composed(name: str) -> str:
                    return mapper(name)

                composed = _composed
        return replace(self, alias_name=pc.Some(composed))

    def clear_alias(self) -> Self:
        return replace(self, alias_name=pc.NONE)


@dataclass(slots=True)
class ExprProjection:
    expr: sql.SqlExpr
    meta: ExprMeta

    def as_aliased(self) -> sql.SqlExpr:
        return self.expr.alias(self.meta.output_name)

    def as_unique(self) -> str:
        base_sql = str(self.expr)
        return (
            base_sql
            if self.meta.output_name == base_sql
            else f"{base_sql} AS {self.meta.output_name}"
        )


@dataclass(slots=True)
class ExprPlan:
    projections: pc.Seq[ExprProjection]

    @classmethod
    def from_inputs(
        cls,
        columns: pc.Seq[str],
        exprs: pc.Iter[IntoExpr],
        named_exprs: dict[str, IntoExpr] | None = None,
    ) -> Self:
        expr_map = (
            pc.Option(named_exprs)
            .map(
                lambda mapping: (
                    pc.Dict.from_ref(mapping)
                    .items()
                    .iter()
                    .map_star(
                        lambda k, v: _resolve_projection(
                            columns, v, alias_override=pc.Some(k)
                        )
                    )
                    .flatten()
                )
            )
            .unwrap_or(pc.Iter[ExprProjection].new())
        )
        return cls(
            exprs.flat_map(lambda value: _resolve_projection(columns, value))
            .chain(expr_map)
            .collect()
        )

    def into_iter(self) -> pc.Iter[ExprProjection]:
        return self.projections.iter()

    def aliased_sql(self) -> pc.Iter[sql.SqlExpr]:
        return self.into_iter().map(lambda p: p.as_aliased())

    def unique(self) -> pc.Iter[str]:
        return self.into_iter().map(lambda p: p.as_unique())

    def to_updates(self) -> pc.Dict[str, sql.SqlExpr]:
        return (
            self.into_iter()
            .map(lambda p: (p.meta.output_name, p.expr))
            .collect(pc.Dict)
        )

    def is_scalar_select(self) -> bool:
        return self.into_iter().all(lambda p: p.meta.is_scalar_select)

    def can_use_unique(self) -> bool:
        return self.into_iter().all(lambda p: p.meta.is_unique_projection)


def _resolve_projection(
    columns: pc.Seq[str], value: IntoExpr, *, alias_override: pc.Option[str] = pc.NONE
) -> pc.Iter[ExprProjection]:
    into_proj = pc.Iter[ExprProjection].once
    match value:
        case Expr() as expr:
            base_names = (
                columns if expr.meta.is_multi else pc.Seq((expr.meta.root_name,))
            )
            output_names = expr.meta.resolve_output_names(base_names, alias_override)
            match expr.meta.is_multi and alias_override.is_none():
                case True:
                    return (
                        columns.iter()
                        .zip(output_names)
                        .map_star(
                            lambda column_name, output_name: ExprProjection(
                                sql.col(column_name),
                                expr.meta.from_projection(output_name),
                            )
                        )
                    )
                case False:
                    return into_proj(
                        ExprProjection(
                            expr.inner(),
                            expr.meta.from_projection(output_names.first()),
                        )
                    )
        case _:
            resolved = sql.into_expr(value, as_col=True)
            resolved_meta = ExprMeta.__from_expr__(resolved)
            return into_proj(
                ExprProjection(
                    resolved,
                    resolved_meta.from_projection(
                        alias_override.unwrap_or(resolved_meta.root_name)
                    ),
                )
            )


@dataclass(slots=True, init=False)
class Expr(sql.CoreHandler[sql.SqlExpr]):
    """Expression wrapper providing Polars-like API over DuckDB expressions."""

    meta: ExprMeta

    def __init__(self, inner: sql.SqlExpr, meta: pc.Option[ExprMeta] = pc.NONE) -> None:
        self._inner = inner
        self.meta = meta.map(replace).unwrap_or_else(
            lambda: ExprMeta.__from_expr__(inner)
        )

    def _new(self, value: sql.SqlExpr, meta: pc.Option[ExprMeta] = pc.NONE) -> Self:
        return self.__class__(
            value,
            pc.Some(meta.unwrap_or_else(lambda: replace(self.meta, is_multi=False))),
        )

    def _with_meta(
        self,
        value: sql.SqlExpr,
        **changes: str | bool | pc.Option[Callable[[str], str]],
    ) -> Self:
        return self._new(value, pc.Some(replace(self.meta, **changes)))

    def _as_window(
        self, expr: sql.SqlExpr, *, is_scalar_like: bool | None = None
    ) -> Self:
        return self._with_meta(
            expr,
            has_window=True,
            is_scalar_like=pc.Option(is_scalar_like).unwrap_or(
                self.meta.is_scalar_like
            ),
        )

    def _as_scalar(self, expr: sql.SqlExpr) -> Self:
        return self._with_meta(expr, is_scalar_like=True)

    def _reversed(self, expr: sql.SqlExpr, *, reverse: bool = False) -> Self:
        match reverse:
            case True:
                return self._as_window(expr.over(rows_start=0))
            case False:
                return self._as_window(expr.over(rows_end=0))

    def _clear_alias_name(self) -> Expr:
        return self._new(self.inner(), pc.Some(self.meta.clear_alias()))

    def _rolling_agg(
        self,
        agg: Callable[[sql.SqlExpr], sql.SqlExpr],
        window_size: int,
        min_samples: int | None,
        *,
        center: bool,
    ) -> Self:
        bounds = RollingBounds.new(window_size, center=center)
        return self._as_window(
            sql.when(
                self.inner()
                .count()
                .over(rows_start=bounds.start, rows_end=bounds.end)
                .ge(pc.Option(min_samples).unwrap_or(window_size))
            )
            .then(agg(self.inner()).over(rows_start=bounds.start, rows_end=bounds.end))
            .otherwise(_NONE)
        )

    @property
    def str(self) -> ExprStringNameSpace:
        """Access string operations."""
        return ExprStringNameSpace(self)

    @property
    def list(self) -> ExprListNameSpace:
        """Access list operations."""
        return ExprListNameSpace(self)

    @property
    def arr(self) -> ExprArrayNameSpace:
        """Access array operations."""
        return ExprArrayNameSpace(self)

    @property
    def struct(self) -> ExprStructNameSpace:
        """Access struct operations."""
        return ExprStructNameSpace(self)

    @property
    def name(self) -> ExprNameNameSpace:
        """Access name operations."""
        return ExprNameNameSpace(self)

    def __add__(self, other: IntoExpr) -> Self:
        return self.add(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self._new(sql.into_expr(other).radd(self.inner()))

    def __sub__(self, other: IntoExpr) -> Self:
        return self.sub(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rsub(other))

    def __mul__(self, other: IntoExpr) -> Self:
        return self.mul(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self._new(sql.into_expr(other).rmul(self.inner()))

    def __truediv__(self, other: IntoExpr) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rtruediv(other))

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rfloordiv(other))

    def __mod__(self, other: IntoExpr) -> Self:
        return self.mod(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rmod(other))

    def __pow__(self, other: IntoExpr) -> Self:
        return self.pow(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self._new(self.inner().rpow(other))

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
        return self._new(sql.into_expr(other).rand(self.inner()))

    def __or__(self, other: IntoExpr) -> Self:
        return self.or_(other)

    def __xor__(self, other: IntoExpr) -> Self:
        return self.xor(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self._new(sql.into_expr(other).ror(self.inner()))

    def __invert__(self) -> Self:
        return self.not_()

    def __hash__(self) -> int:
        return hash(str(self.inner()))

    def add(self, other: IntoExpr) -> Self:
        """Add another expression or value."""
        return self._new(self.inner().add(other))

    def sub(self, other: IntoExpr) -> Self:
        return self._new(self.inner().sub(other))

    def mul(self, other: IntoExpr) -> Self:
        return self._new(self.inner().mul(other))

    def truediv(self, other: IntoExpr) -> Self:
        return self._new(self.inner().truediv(other))

    def floordiv(self, other: IntoExpr) -> Self:
        return self._new(self.inner().floordiv(other))

    def mod(self, other: IntoExpr) -> Self:
        return self._new(self.inner().mod(other))

    def pow(self, other: IntoExpr) -> Self:
        return self._new(self.inner().pow(other))

    def neg(self) -> Self:
        return self._new(self.inner().neg())

    def abs(self) -> Self:
        return self._new(self.inner().abs())

    def eq(self, other: IntoExpr) -> Self:
        return self._new(self.inner().eq(other))

    def ne(self, other: IntoExpr) -> Self:
        return self._new(self.inner().ne(other))

    def lt(self, other: IntoExpr) -> Self:
        return self._new(self.inner().lt(other))

    def le(self, other: IntoExpr) -> Self:
        return self._new(self.inner().le(other))

    def gt(self, other: IntoExpr) -> Self:
        return self._new(self.inner().gt(other))

    def ge(self, other: IntoExpr) -> Self:
        return self._new(self.inner().ge(other))

    def and_(self, others: IntoExpr) -> Self:
        return self._new(self.inner().and_(others))

    def or_(self, others: IntoExpr) -> Self:
        return self._new(self.inner().or_(others))

    def not_(self) -> Self:
        return self._new(self.inner().not_())

    def bitwise_and(self) -> Self:
        return self._as_scalar(self.inner().bit_and())

    def bitwise_or(self) -> Self:
        return self._as_scalar(self.inner().bit_or())

    def bitwise_xor(self) -> Self:
        return self._as_scalar(self.inner().bit_xor())

    def xor(self, other: IntoExpr) -> Self:
        return self._new(self.inner().xor(sql.into_expr(other)))

    def alias(self, name: str) -> Self:
        """Rename the expression."""
        return self._with_meta(self.inner(), alias_name=pc.Some(lambda _: name))

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
        return self._new(self.inner().is_in(*try_iter(other)))

    def shift(self, n: int = 1) -> Self:
        return self._as_window(expr=self.inner().pipe(shift, n))

    def diff(self) -> Self:
        return self.sub(self.shift())

    def pct_change(self, n: int = 1) -> Self:
        return self.truediv(self.shift(n)).sub(1)

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        match closed:
            case "both":
                return self.ge(lower_bound).and_(self.le(upper_bound))
            case "left":
                return self.ge(lower_bound).and_(self.lt(upper_bound))
            case "right":
                return self.gt(lower_bound).and_(self.le(upper_bound))
            case "none":
                return self.gt(lower_bound).and_(self.lt(upper_bound))

    def clip(
        self, lower_bound: IntoExpr | None = None, upper_bound: IntoExpr | None = None
    ) -> Self:
        match (lower_bound, upper_bound):
            case (None, None):
                return self
            case (None, upper):
                return self._new(self.inner().least(sql.into_expr(upper)))
            case (lower, None):
                return self._new(self.inner().greatest(sql.into_expr(lower)))
            case (lower, upper):
                return self._new(
                    self.inner()
                    .greatest(sql.into_expr(lower))
                    .least(sql.into_expr(upper))
                )

    def count(self) -> Self:
        """Count the number of values."""
        return self._as_scalar(self.inner().count())

    def len(self) -> Self:
        """Get the number of rows in context (including nulls)."""
        return self._as_scalar(self.inner().is_null().count())

    def sum(self) -> Self:
        """Compute the sum."""
        return self._as_scalar(self.inner().sum())

    def mean(self) -> Self:
        """Compute the mean."""
        return self._as_scalar(self.inner().mean())

    def median(self) -> Self:
        """Compute the median."""
        return self._as_scalar(self.inner().median())

    def min(self) -> Self:
        """Compute the minimum."""
        return self._as_scalar(self.inner().min())

    def max(self) -> Self:
        """Compute the maximum."""
        return self._as_scalar(self.inner().max())

    def first(self, *, ignore_nulls: bool = False) -> Self:
        """Get first value."""
        match ignore_nulls:
            case True:
                return self._as_scalar(self.inner().any_value())
            case False:
                return self._as_scalar(self.inner().first())

    def last(self) -> Self:
        """Get last value."""
        return self._as_scalar(self.inner().last())

    def mode(self) -> Self:
        """Compute mode."""
        return self._as_scalar(self.inner().mode())

    def approx_n_unique(self) -> Self:
        """Approximate the number of unique values."""
        return self._as_scalar(self.inner().approx_count_distinct())

    def product(self) -> Self:
        """Compute the product."""
        return self._as_scalar(self.inner().product())

    def max_by(self, by: IntoExpr) -> Self:
        """Return the value corresponding to the maximum of another expression."""
        return self._as_scalar(self.inner().max_by(sql.into_expr(by, as_col=True)))

    def min_by(self, by: IntoExpr) -> Self:
        """Return the value corresponding to the minimum of another expression."""
        return self._as_scalar(self.inner().min_by(sql.into_expr(by, as_col=True)))

    def implode(self) -> Self:
        """Aggregate values into a list."""
        return self._as_scalar(self.inner().implode())

    def unique(self) -> Self:
        """Get unique values."""
        return self._with_meta(self.inner(), is_unique_projection=True)

    def is_close(
        self,
        other: IntoExpr,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-5,
        *,
        nans_equal: bool = False,
    ) -> Self:
        """Check if two floating point values are close."""
        other_expr = sql.into_expr(other)
        threshold = sql.lit(abs_tol).add(sql.lit(rel_tol).mul(other_expr.abs()))
        close = self.inner().sub(other_expr).abs().le(threshold)
        match nans_equal:
            case False:
                return self._new(close)
            case True:
                return self._new(
                    sql.when(self.inner().isnan().and_(other_expr.isnan()))
                    .then(value=True)
                    .otherwise(close)
                )

    def rolling_max(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling mean."""
        return self._rolling_agg(
            lambda expr: expr.max(), window_size, min_samples, center=center
        )

    def rolling_min(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling mean."""
        return self._rolling_agg(
            lambda expr: expr.min(), window_size, min_samples, center=center
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
            lambda expr: expr.mean(), window_size, min_samples, center=center
        )

    def rolling_median(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling mean."""
        return self._rolling_agg(
            lambda expr: expr.median(), window_size, min_samples, center=center
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
            lambda expr: expr.std(ddof), window_size, min_samples, center=center
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
            lambda expr: expr.var(ddof), window_size, min_samples, center=center
        )

    def std(self, ddof: int = 1) -> Self:
        """Compute the standard deviation."""
        return self._as_scalar(self.inner().std(ddof))

    def var(self, ddof: int = 1) -> Self:
        """Compute the variance."""
        return self._as_scalar(self.inner().var(ddof))

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
        base = self.inner().kurtosis(bias=bias)
        match fisher:
            case True:
                return self._as_scalar(base)
            case False:
                return self._as_scalar(base.add(3))

    def skew(self, *, bias: bool = True) -> Self:
        adjusted = self.inner().skewness()
        match bias:
            case False:
                return self._as_scalar(adjusted)
            case True:
                n = self.inner().count()
                factor = n.sub(2).truediv(n.mul(n.sub(1)).sqrt())
                return self._as_scalar(adjusted.mul(factor))

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> Self:
        match interpolation:
            case "linear" | "midpoint":
                expr = self.inner().quantile_cont(quantile)
            case _:
                expr = self.inner().quantile(quantile)
        return self._as_scalar(expr)

    def all(self) -> Self:
        """Return whether all values are true."""
        return self._as_scalar(self.inner().bool_and())

    def any(self) -> Self:
        """Return whether any value is true."""
        return self._as_scalar(self.inner().bool_or())

    def n_unique(self) -> Self:
        """Count distinct values."""
        return self._as_scalar(self.inner().implode().list.distinct().list.length())

    def null_count(self) -> Self:
        """Count null values."""
        return self.len().sub(self.count())

    def has_nulls(self) -> Self:
        """Return whether the expression contains nulls."""
        return self._as_scalar(self.inner().is_null().bool_or())

    def rank(self, method: RankMethod = "average", *, descending: bool = False) -> Self:
        """Compute rank values."""
        base_rank = (
            self.inner().rank().over(order_by=self.inner(), descending=descending)
        )
        peer_count = sql.all().count().over(self.inner())
        match method:
            case "average":
                max_rank = base_rank.add(peer_count).sub(1)
                expr = base_rank.add(max_rank).truediv(2)
            case "min":
                expr = base_rank
            case "max":
                expr = base_rank.add(peer_count).sub(1)
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
        return self._as_window(expr)

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
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        expr = partial(self.inner().over, descending=descending, nulls_last=nulls_last)
        return (
            try_chain(partition_by, more_exprs)
            .map(lambda x: sql.into_expr(x, as_col=True))
            .into(
                lambda partition_exprs: (
                    pc.Option(order_by)
                    .map(
                        lambda value: (
                            try_iter(value)
                            .map(lambda x: sql.into_expr(x, as_col=True))
                            .collect()
                        )
                    )
                    .map(lambda order_exprs: expr(partition_exprs, order_exprs))
                    .unwrap_or_else(lambda: expr(partition_exprs))
                )
            )
            .pipe(self._as_window)
        )

    def floor(self) -> Self:
        """Round down to the nearest integer."""
        return self._new(self.inner().floor())

    def ceil(self) -> Self:
        """Round up to the nearest integer."""
        return self._new(self.inner().ceil())

    def round(self, decimals: int = 0, *, mode: RoundMode = "half_to_even") -> Self:
        """Round to given number of decimal places."""
        return self._new(round(self.inner(), decimals, mode=mode))

    def sqrt(self) -> Self:
        """Compute the square root."""
        return self._new(self.inner().sqrt())

    def cbrt(self) -> Self:
        """Compute the cube root."""
        return self._new(self.inner().cbrt())

    def log(self, base: float = 2.718281828459045) -> Self:
        """Compute the logarithm."""
        return self._new(self.inner().log(base))

    def log10(self) -> Self:
        """Compute the base 10 logarithm."""
        return self._new(self.inner().log10())

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x."""
        return self._new(self.inner().add(1).ln())

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

    def arccos(self) -> Self:
        """Compute the arc cosine."""
        return self._new(self.inner().acos())

    def arccosh(self) -> Self:
        """Compute the inverse hyperbolic cosine."""
        return self._new(self.inner().acosh())

    def arcsin(self) -> Self:
        """Compute the arc sine."""
        return self._new(self.inner().asin())

    def arcsinh(self) -> Self:
        """Compute the inverse hyperbolic sine."""
        return self._new(self.inner().asinh())

    def arctanh(self) -> Self:
        """Compute the inverse hyperbolic tangent."""
        return self._new(self.inner().atanh())

    def cot(self) -> Self:
        """Compute the cotangent."""
        return self._new(self.inner().cot())

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

    def backward_fill(self, limit: int | None = None) -> Self:
        """Fill null values with the next non-null value."""
        expr = self.inner().any_value()
        return (
            pc.Option(limit)
            .map(lambda lmt: expr.over(rows_start=0, rows_end=lmt))
            .unwrap_or_else(lambda: expr.over(rows_start=0))
            .pipe(self._as_window)
        )

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
        return self._new(
            sql.when(self.inner().isnan()).then(value).otherwise(self.inner())
        )

    def fill_null(
        self,
        value: IntoExpr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        return (
            self.inner()
            .pipe(fill_nulls, value, strategy, limit)
            .map(self._new)
            .unwrap()
        )

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash."""
        return self._new(self.inner().str.hash(sql.lit(seed)))

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values."""
        return self._new(
            sql.when(self.inner().eq(old)).then(new).otherwise(self.inner())
        )

    def repeat_by(self, by: Expr | int) -> Self:
        """Repeat values by count, returning a list."""
        return self._new(
            sql.into_expr(by).list.range().list.transform(sql.fn_once(self.inner()))
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated."""
        return self._new(sql.all().count().over(self.inner()).gt(1))

    def is_unique(self) -> Self:
        """Check if value is unique."""
        return self._new(sql.all().count().over(self.inner()).eq(1))

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence."""
        return self._new(self.inner().row_number().over(self.inner()).eq(1))

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence."""
        return self._new(
            self.inner()
            .row_number()
            .over(self.inner(), self.inner(), descending=True, nulls_last=True)
            .eq(1)
        )


@dataclass(slots=True)
class ExprNameSpaceBase:
    _parent: Expr

    def inner(self) -> sql.SqlExpr:
        return self._parent.inner()

    def _new(self, expr: sql.SqlExpr) -> Expr:
        return self._parent._new(expr)  # pyright: ignore[reportPrivateUsage]

    def _with_alias_mapper(self, mapper: Callable[[str], str]) -> Expr:
        return self._parent._new(  # pyright: ignore[reportPrivateUsage]
            self._parent.inner(),
            pc.Some(self._parent.meta.with_alias_mapper(mapper)),
        )


@dataclass(slots=True)
class ExprStringNameSpace(ExprNameSpaceBase):
    """String operations namespace (equivalent to pl.Expr.str)."""

    def join(self, delimiter: str = "", *, ignore_nulls: bool = True) -> Expr:
        """Vertically concatenate string values into a single string."""
        aggregated = self.inner().str.agg(sql.lit(delimiter))
        match ignore_nulls:
            case True:
                return self._parent._as_scalar(aggregated)  # pyright: ignore[reportPrivateUsage]
            case False:
                return self._parent._as_scalar(  # pyright: ignore[reportPrivateUsage]
                    sql.when(self.inner().is_null().bool_or())
                    .then(_NONE)
                    .otherwise(aggregated)
                )

    def escape_regex(self) -> Expr:
        """Escape all regex meta characters in the string."""
        return self._new(
            self.inner().re.replace(
                sql.lit(r"([.^$*+?{}\[\]\\|()])"), sql.lit(r"\\\1"), sql.lit("g")
            )
        )

    def to_uppercase(self) -> Expr:
        """Convert to uppercase."""
        return self._new(self.inner().str.upper())

    def to_lowercase(self) -> Expr:
        """Convert to lowercase."""
        return self._new(self.inner().str.lower())

    def len_chars(self) -> Expr:
        """Get the length in characters."""
        return self._new(self.inner().str.length())

    def contains(self, pattern: str, *, literal: bool = False) -> Expr:
        """Check if string contains a pattern."""
        match literal:
            case True:
                return self._new(self.inner().str.contains(sql.lit(pattern)))
            case False:
                return self._new(self.inner().re.matches(sql.lit(pattern)))

    def starts_with(self, prefix: str) -> Expr:
        """Check if string starts with prefix."""
        return self._new(self.inner().str.starts_with(sql.lit(prefix)))

    def ends_with(self, suffix: str) -> Expr:
        """Check if string ends with suffix."""
        return self._new(self.inner().str.ends_with(sql.lit(suffix)))

    def replace(
        self, pattern: str, value: str | Expr, *, literal: bool = False, n: int = 1
    ) -> Expr:
        """Replace first matching substring with a new string value."""
        value_expr = sql.into_expr(value)
        pattern_expr = sql.lit(re.escape(pattern) if literal else pattern)

        def _replace_once(expr: sql.SqlExpr) -> sql.SqlExpr:
            return expr.str.replace(pattern_expr, value_expr)

        match n:
            case 0:
                return self._new(self.inner())
            case n_val if n_val < 0:
                return self._new(
                    self.inner().re.replace(pattern_expr, value_expr, sql.lit("g"))
                )
            case _:
                return (
                    pc.Iter(range(n))
                    .fold(self.inner(), lambda acc, _: _replace_once(acc))
                    .pipe(self._new)
                )

    def strip_chars(self, characters: str | None = None) -> Expr:
        """Strip leading and trailing characters."""
        match characters:
            case None:
                return self._new(self.inner().str.trim())
            case _:
                return self._new(self.inner().str.trim(sql.lit(characters)))

    def strip_chars_start(self, characters: IntoExpr = None) -> Expr:
        """Strip leading characters."""
        match characters:
            case None:
                return self._new(self.inner().str.ltrim())
            case _:
                return self._new(self.inner().str.ltrim(sql.into_expr(characters)))

    def strip_chars_end(self, characters: IntoExpr = None) -> Expr:
        """Strip trailing characters."""
        match characters:
            case None:
                return self._new(self.inner().str.rtrim())
            case _:
                return self._new(self.inner().str.rtrim(sql.into_expr(characters)))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring."""
        return self._new(self.inner().str.substring(offset + 1, length))

    def len_bytes(self) -> Expr:
        """Get the length in bytes."""
        return self._new(self.inner().encode().octet_length())

    def split(self, by: str) -> Expr:
        """Split string by separator."""
        return self._new(self.inner().str.split(sql.lit(by)))

    def extract_all(self, pattern: str | Expr) -> Expr:
        """Extract all regex matches."""
        return self._new(self.inner().re.extract_all(sql.into_expr(pattern)))

    def extract(self, pattern: IntoExprColumn, group_index: int = 1) -> Expr:
        """Extract a regex capture group."""
        return self._new(self.inner().re.extract(sql.into_expr(pattern), group_index))

    def find(self, pattern: str | Expr, *, literal: bool = False) -> Expr:
        """Return the first match offset as a zero-based index."""
        pattern_expr = sql.into_expr(pattern)
        match literal:
            case True:
                return (
                    self.inner()
                    .str.strpos(pattern_expr)
                    .pipe(
                        lambda pos: (
                            sql.when(pos.eq(sql.lit(0)))
                            .then(_NONE)
                            .otherwise(pos.sub(sql.lit(1)))
                        )
                    )
                    .pipe(self._new)
                )
            case False:
                return (
                    self.inner()
                    .re.extract(pattern_expr, 0)
                    .pipe(
                        lambda matched: (
                            sql.when(matched.eq(_EMPTY_STR))
                            .then(_NONE)
                            .otherwise(self.inner().str.strpos(matched).sub(sql.lit(1)))
                        )
                    )
                    .pipe(self._new)
                )

    def json_path_match(self, json_path: IntoExprColumn) -> Expr:
        """Extract first JSONPath match from string JSON values."""
        return self._new(self.inner().json.extract_string(sql.into_expr(json_path)))

    def to_date(self, format: str | None = None) -> Expr:  # noqa: A002
        """Parse string values as date."""
        match format:
            case None:
                return self._parent.cast(dt.Date())
            case _:
                return self._new(self.inner().str.strptime(sql.lit(format))).cast(
                    dt.Date()
                )

    def to_datetime(self, format: str | None = None) -> Expr:  # noqa: A002
        """Parse string values as datetime."""
        match format:
            case None:
                return self._parent.cast(dt.Datetime())
            case _:
                return self._new(self.inner().str.strptime(sql.lit(format))).cast(
                    dt.Datetime()
                )

    def to_time(self, format: str | None = None) -> Expr:  # noqa: A002
        """Parse string values as time."""
        match format:
            case None:
                return self._parent.cast(dt.Time())
            case _:
                return self._new(self.inner().str.strptime(sql.lit(format))).cast(
                    dt.Time()
                )

    def strptime(self, format: str | Expr) -> Expr:  # noqa: A002
        """Parse string values into datetime using one or more formats."""
        return self._new(self.inner().str.strptime(sql.into_expr(format)))

    def encode(self, encoding: TransferEncoding = "base64") -> Expr:
        """Encode UTF-8 strings as binary values."""
        match encoding:
            case "base64":
                return self._new(self.inner().encode().str.to_base64())
            case "hex":
                return self._new(self.inner().encode().str.to_hex().str.lower())

    def normalize(self) -> Expr:
        """Normalize strings using NFC normalization."""
        return self._new(self.inner().str.nfc_normalize())

    def to_decimal(self, scale: int) -> Expr:
        """Parse string values as decimal with the requested scale."""
        return self._parent.cast(dt.Decimal(scale=scale))

    def count_matches(self, pattern: str | Expr, *, literal: bool = False) -> Expr:
        """Count pattern matches."""
        pattern_expr = sql.into_expr(pattern)
        match literal:
            case False:
                return self._new(self.inner().re.extract_all(pattern_expr).list.len())
            case True:
                return (
                    self.inner()
                    .str.length()
                    .sub(
                        self.inner().str.replace(pattern_expr, _EMPTY_STR).str.length()
                    )
                    .truediv(pattern_expr.str.length())
                    .pipe(self._new)
                )

    def strip_prefix(self, prefix: IntoExpr) -> Expr:
        """Strip prefix from string."""
        match prefix:
            case str() as prefix_str:
                return (
                    self.inner()
                    .re.replace(sql.lit(f"^{re.escape(prefix_str)}"), _EMPTY_STR)
                    .pipe(self._new)
                )
            case _:
                return (
                    sql.into_expr(prefix)
                    .pipe(
                        lambda prefix: sql.when(
                            self.inner().str.starts_with(prefix),
                        ).then(
                            self.inner().str.substring(
                                prefix.str.length().add(sql.lit(1))
                            )
                        )
                    )
                    .otherwise(self.inner())
                    .pipe(self._new)
                )

    def strip_suffix(self, suffix: IntoExpr) -> Expr:
        """Strip suffix from string."""
        match suffix:
            case str() as suffix_str:
                return self._new(
                    self.inner().re.replace(
                        sql.lit(f"{re.escape(suffix_str)}$"), _EMPTY_STR
                    )
                )
            case _:
                suffix_expr = sql.into_expr(suffix)
                return self._new(
                    sql.when(
                        self.inner().str.ends_with(suffix_expr),
                    )
                    .then(
                        self.inner().str.substring(
                            1, self.inner().str.length().sub(suffix_expr.str.length())
                        )
                    )
                    .otherwise(self.inner())
                )

    def head(self, n: int) -> Expr:
        """Get first n characters."""
        return self._new(self.inner().str.left(n))

    def tail(self, n: int) -> Expr:
        """Get last n characters."""
        return self._new(self.inner().str.right(n))

    def reverse(self) -> Expr:
        """Reverse the string."""
        return self._new(self.inner().str.reverse())

    def pad_start(self, length: int, fill_char: str = " ") -> Expr:
        return self._new(self.inner().str.lpad(length, sql.lit(fill_char)))

    def pad_end(self, length: int, fill_char: str = " ") -> Expr:
        return self._new(self.inner().str.rpad(length, sql.lit(fill_char)))

    def zfill(self, width: int) -> Expr:
        return self._new(self.inner().str.lpad(width, sql.lit("0")))

    def replace_all(
        self, pattern: str, value: IntoExpr, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences."""
        value_expr = sql.into_expr(value)
        match literal:
            case True:
                return self._new(self.inner().str.replace(sql.lit(pattern), value_expr))
            case False:
                return self._new(
                    self.inner().re.replace(sql.lit(pattern), value_expr, sql.lit("g"))
                )

    def to_titlecase(self) -> Expr:
        """Convert to title case."""
        return self._new(
            self.inner()
            .str.lower()
            .re.extract_all(sql.lit(r"[a-z]*[^a-z]*"))
            .list.transform(
                sql.fn_once(
                    sql.element()
                    .list.extract(1)
                    .str.upper()
                    .str.concat(sql.element().str.substring(2))
                )
            )
            .list.aggregate(sql.lit("string_agg"), _EMPTY_STR)
        )


@dataclass(slots=True)
class ExprArrayNameSpace(ExprNameSpaceBase):
    """Array operations namespace (equivalent to pl.Expr.array)."""

    def all(self) -> Expr:
        """Return whether all values in the array are true."""
        return self._new(self.inner().list.bool_and())

    def any(self) -> Expr:
        """Return whether any value in the array is true."""
        return self._new(self.inner().list.bool_or())

    def eval(self, expr: Expr) -> Expr:
        """Run an expression against each array element."""
        return self._new(self.inner().arr.transform(sql.fn_once(expr.inner())))

    def filter(self, predicate: Expr) -> Expr:
        return self._new(self.inner().arr.filter(sql.fn_once(predicate.inner())))

    def len(self) -> Expr:
        """Return the number of elements in each array."""
        return self._new(self.inner().arr.length())

    def unique(self) -> Expr:
        """Return unique values in each array."""
        return self._new(self.inner().arr.distinct())

    def n_unique(self) -> Expr:
        """Return the number of unique values in each array."""
        return self._new(self.inner().arr.distinct().arr.length())

    def contains(self, item: IntoExpr) -> Expr:
        """Check if subarrays contain the given item."""
        return self._new(self.inner().arr.contains(sql.into_expr(item)))

    def count_matches(self, element: IntoExpr) -> Expr:
        """Count matches in each array."""
        return self._new(
            self.inner()
            .arr.filter(sql.fn_once(sql.element().eq(sql.into_expr(element))))
            .arr.length()
        )

    def drop_nulls(self) -> Expr:
        """Drop null values in each array."""
        return self._new(
            self.inner().arr.filter(sql.fn_once(sql.element().is_not_null()))
        )

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> Expr:
        """Join string values in each array with a separator."""
        joined = self.inner().arr.aggregate(
            sql.lit("string_agg"), sql.into_expr(separator)
        )
        match ignore_nulls:
            case True:
                return self._new(sql.coalesce(joined, _EMPTY_STR))
            case False:
                return self._new(
                    sql.when(
                        self.inner()
                        .arr.filter(sql.fn_once(sql.element().is_null()))
                        .arr.length()
                        .gt(sql.lit(0))
                    )
                    .then(_NONE)
                    .otherwise(sql.coalesce(joined, _EMPTY_STR))
                )

    def get(self, index: int) -> Expr:
        """Return the value by index in each array."""
        return self._new(self.inner().arr.extract(index + 1 if index >= 0 else index))

    def first(self) -> Expr:
        """Get the first element of each array."""
        return self._new(self.inner().list.first())

    def last(self) -> Expr:
        """Get the last element of each array."""
        return self._new(self.inner().list.last())

    def min(self) -> Expr:
        """Compute the min value of the arrays in the column."""
        return self._new(self.inner().list.min())

    def max(self) -> Expr:
        """Compute the max value of the arrays in the column."""
        return self._new(self.inner().list.max())

    def mean(self) -> Expr:
        """Compute the mean value of the arrays in the column."""
        return self._new(self.inner().list.avg())

    def median(self) -> Expr:
        """Compute the median value of the arrays in the column."""
        return self._new(self.inner().list.median())

    def sum(self) -> Expr:
        """Compute the sum value of the arrays in the column."""
        return self._new(self.inner().list.sum())

    def std(self, ddof: int = 1) -> Expr:
        """Compute the standard deviation of the arrays in the column."""
        return self._new(self.inner().list.std(ddof))

    def var(self, ddof: int = 1) -> Expr:
        """Compute the variance of the arrays in the column."""
        return self._new(self.inner().list.var(ddof))

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the column."""
        return self._new(
            self.inner().arr.sort(
                sql.lit(sql.Kword.sort_order(desc=descending)),
                sql.lit(sql.Kword.null_order(last=nulls_last)),
            )
        )

    def reverse(self) -> Expr:
        """Reverse the arrays of the expression."""
        return self._new(self.inner().arr.reverse())


@dataclass(slots=True)
class ExprListNameSpace(ExprNameSpaceBase):
    """List operations namespace (equivalent to pl.Expr.list)."""

    def all(self) -> Expr:
        """Return whether all values in the list are true."""
        return self._new(self.inner().list.bool_and())

    def any(self) -> Expr:
        """Return whether any value in the listis true."""
        return self._new(self.inner().list.bool_or())

    def eval(self, expr: Expr) -> Expr:
        """Run an expression against each list element."""
        return self._new(self.inner().list.transform(sql.fn_once(expr.inner())))

    def filter(self, predicate: Expr) -> Expr:
        return self._new(self.inner().list.filter(sql.fn_once(predicate.inner())))

    def len(self) -> Expr:
        """Return the number of elements in each list."""
        return self._new(self.inner().list.length())

    def unique(self) -> Expr:
        """Return unique values in each list."""
        return self._new(self.inner().list.distinct())

    def n_unique(self) -> Expr:
        """Return the number of unique values in each list."""
        return self._new(self.inner().list.distinct().list.length())

    def contains(self, item: IntoExpr) -> Expr:
        """Check if sublists contain the given item."""
        return self._new(self.inner().list.contains(sql.into_expr(item)))

    def count_matches(self, element: IntoExpr) -> Expr:
        """Count matches in each list."""
        return self._new(
            self.inner()
            .list.filter(sql.fn_once(sql.element().eq(sql.into_expr(element))))
            .list.length()
        )

    def drop_nulls(self) -> Expr:
        """Drop null values in each list."""
        return self._new(
            self.inner().list.filter(sql.fn_once(sql.element().is_not_null()))
        )

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> Expr:
        """Join string values in each list with a separator."""
        joined = self.inner().list.aggregate(
            sql.lit("string_agg"), sql.into_expr(separator)
        )
        match ignore_nulls:
            case True:
                return self._new(sql.coalesce(joined, _EMPTY_STR))
            case False:
                return self._new(
                    sql.when(
                        self.inner()
                        .list.filter(sql.fn_once(sql.element().is_null()))
                        .list.length()
                        .gt(sql.lit(0))
                    )
                    .then(_NONE)
                    .otherwise(sql.coalesce(joined, _EMPTY_STR))
                )

    def get(self, index: int) -> Expr:
        """Return the value by index in each list."""
        return self._new(self.inner().list.extract(index + 1 if index >= 0 else index))

    def first(self) -> Expr:
        """Get the first element of each list."""
        return self._new(self.inner().list.first())

    def last(self) -> Expr:
        """Get the last element of each list."""
        return self._new(self.inner().list.last())

    def min(self) -> Expr:
        """Compute the min value of the lists in the column."""
        return self._new(self.inner().list.min())

    def max(self) -> Expr:
        """Compute the max value of the lists in the column."""
        return self._new(self.inner().list.max())

    def mean(self) -> Expr:
        """Compute the mean value of the lists in the column."""
        return self._new(self.inner().list.avg())

    def median(self) -> Expr:
        """Compute the median value of the lists in the column."""
        return self._new(self.inner().list.median())

    def sum(self) -> Expr:
        """Compute the sum value of the lists in the column."""
        return self._new(self.inner().list.sum())

    def std(self, ddof: int = 1) -> Expr:
        """Compute the standard deviation of the lists in the column."""
        return self._new(self.inner().list.std(ddof))

    def var(self, ddof: int = 1) -> Expr:
        """Compute the variance of the lists in the column."""
        return self._new(self.inner().list.var(ddof))

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the column."""
        return self._new(
            self.inner().list.sort(
                sql.lit(sql.Kword.sort_order(desc=descending)),
                sql.lit(sql.Kword.null_order(last=nulls_last)),
            )
        )

    def reverse(self) -> Expr:
        """Reverse the lists of the expression."""
        return self._new(self.inner().list.reverse())


@dataclass(slots=True)
class ExprStructNameSpace(ExprNameSpaceBase):
    """Struct operations namespace (equivalent to pl.Expr.struct)."""

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name."""
        return self._new(self.inner().struct.extract(sql.lit(name)))

    def json_encode(self) -> Expr:
        """Encode struct values as JSON strings."""
        return self._new(self.inner().to_json())

    def with_fields(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr,
        **named_exprs: IntoExpr,
    ) -> Expr:
        """Return a new struct with updated or additional fields."""
        return (
            ExprPlan.from_inputs(
                pc.Seq[str].new(), try_chain(expr, more_exprs), named_exprs
            )
            .aliased_sql()
            .into(lambda args: self._new(self.inner().struct.insert(*args)))
        )


@dataclass(slots=True)
class ExprNameNameSpace(ExprNameSpaceBase):
    """Name operations namespace (equivalent to pl.Expr.name)."""

    def keep(self) -> Expr:
        return self._parent._clear_alias_name()  # pyright: ignore[reportPrivateUsage]

    def map(self, function: Callable[[str], str]) -> Expr:
        return self._with_alias_mapper(function)

    def prefix(self, prefix: str) -> Expr:
        return self._with_alias_mapper(lambda name: f"{prefix}{name}")

    def suffix(self, suffix: str) -> Expr:
        return self._with_alias_mapper(lambda name: f"{name}{suffix}")

    def to_lowercase(self) -> Expr:
        return self._with_alias_mapper(str.lower)

    def to_uppercase(self) -> Expr:
        return self._with_alias_mapper(str.upper)

    def replace(self, pattern: str, value: str, *, literal: bool = False) -> Expr:
        match literal:
            case True:
                return self._with_alias_mapper(
                    lambda name: name.replace(pattern, value)
                )
            case False:
                regex = re.compile(pattern)
                return self._with_alias_mapper(lambda name: regex.sub(value, name))
