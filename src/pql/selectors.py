"""Column selectors for PQL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Self, overload

import duckdb
import pyochain as pc

from . import _datatypes as dt, sql  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr, ExprMeta

if TYPE_CHECKING:
    from .sql.typing import IntoExpr
type Schema = pc.Dict[str, dt.DataType]
type ColumnResolver = Callable[[Schema], pc.traits.PyoCollection[str]]

_SENTINEL = "__pql_selector__"
_SENTINEL_COL = sql.col(_SENTINEL)


def all_columns_resolver(schema: Schema) -> pc.traits.PyoKeysView[str]:
    return schema.keys()


def exclude_resolver(excluded: pc.Set[str]) -> ColumnResolver:
    return lambda schema: (
        schema.keys().iter().filter(lambda n: n not in excluded).collect()
    )


def fixed_resolver(names: pc.Seq[str]) -> ColumnResolver:
    return lambda _schema: names


def _dtype_resolver(predicate: Callable[[dt.DataType], bool]) -> ColumnResolver:
    return lambda schema: (
        schema.items()
        .iter()
        .filter_star(lambda _, dtype: predicate(dtype))
        .map_star(lambda name, _: name)
        .collect()
    )


def _union_resolver(left: ColumnResolver, right: ColumnResolver) -> ColumnResolver:
    def _resolve(schema: Schema) -> pc.Seq[str]:
        selected = left(schema).iter().chain(right(schema)).collect(pc.Set)
        return schema.keys().iter().filter(lambda n: n in selected).collect()

    return _resolve


def _intersection_resolver(
    left: ColumnResolver, right: ColumnResolver
) -> ColumnResolver:
    def _resolve(schema: Schema) -> pc.Seq[str]:
        right_set = right(schema)
        return left(schema).iter().filter(lambda n: n in right_set).collect()

    return _resolve


def _difference_resolver(left: ColumnResolver, right: ColumnResolver) -> ColumnResolver:
    def _resolve(schema: Schema) -> pc.Seq[str]:
        right_set = right(schema)
        return left(schema).iter().filter(lambda n: n not in right_set).collect()

    return _resolve


def _complement_resolver(inner: ColumnResolver) -> ColumnResolver:
    def _resolve(schema: Schema) -> pc.Seq[str]:
        excluded = inner(schema)
        return schema.keys().iter().filter(lambda n: n not in excluded).collect()

    return _resolve


def _replay_transform(template_sql: str, column_name: str) -> sql.SqlExpr:
    """Replay a selector transform by substituting the sentinel with a real column name."""
    return sql.SqlExpr(
        duckdb.SQLExpression(template_sql.replace(_SENTINEL, column_name))
    )


class Selector(Expr):
    """Column selector based on dtype predicates."""

    @classmethod
    def __from_resolver__(cls, resolver: ColumnResolver) -> Self:
        meta = ExprMeta(
            _SENTINEL,
            column_resolver=pc.Some(resolver),
            multi_agg=pc.Some(
                lambda col: _replay_transform(str(_SENTINEL_COL), str(col))
            ),
        )
        return cls(_SENTINEL_COL, pc.Some(meta))

    def _new(self, value: sql.SqlExpr, meta: pc.Option[ExprMeta] = pc.NONE) -> Self:
        """Preserve column_resolver and build a transform from the sentinel-based SQL."""
        new_meta = replace(
            meta.unwrap_or(self.meta),
            column_resolver=self.meta.column_resolver,
            multi_agg=pc.Some(lambda col: _replay_transform(str(value), str(col))),  # type: ignore[arg-type]
        )
        return self.__class__(value, pc.Some(new_meta))

    @overload
    def union(self, other: Selector) -> Selector: ...
    @overload
    def union(self, other: IntoExpr) -> Expr: ...
    def union(self, other: IntoExpr) -> Selector | Expr:
        match other:
            case Selector() if self.meta.is_multi and other.meta.is_multi:
                return Selector.__from_resolver__(
                    _union_resolver(
                        self.meta.column_resolver.unwrap(),
                        other.meta.column_resolver.unwrap(),
                    )
                )
            case _:
                return Expr.__or__(self, other)

    @overload
    def __or__(self, other: Selector) -> Selector: ...
    @overload
    def __or__(self, other: IntoExpr) -> Expr: ...
    def __or__(self, other: IntoExpr) -> Selector | Expr:
        return self.union(other)

    @overload
    def intersection(self, other: Selector) -> Selector: ...
    @overload
    def intersection(self, other: IntoExpr) -> Expr: ...
    def intersection(self, other: IntoExpr) -> Selector | Expr:
        match other:
            case Selector() if self.meta.is_multi and other.meta.is_multi:
                return Selector.__from_resolver__(
                    _intersection_resolver(
                        self.meta.column_resolver.unwrap(),
                        other.meta.column_resolver.unwrap(),
                    )
                )
            case _:
                return Expr.__and__(self, other)

    @overload
    def __and__(self, other: Selector) -> Selector: ...
    @overload
    def __and__(self, other: IntoExpr) -> Expr: ...
    def __and__(self, other: IntoExpr) -> Selector | Expr:
        return self.intersection(other)

    @overload
    def difference(self, other: Selector) -> Selector: ...
    @overload
    def difference(self, other: IntoExpr) -> Expr: ...
    def difference(self, other: IntoExpr) -> Selector | Expr:
        match other:
            case Selector() if self.meta.is_multi and other.meta.is_multi:
                return Selector.__from_resolver__(
                    _difference_resolver(
                        self.meta.column_resolver.unwrap(),
                        other.meta.column_resolver.unwrap(),
                    )
                )
            case _:
                return Expr.__sub__(self, other)

    @overload
    def __sub__(self, other: Selector) -> Selector: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    def __sub__(self, other: IntoExpr) -> Selector | Expr:
        return self.difference(other)

    def complement(self) -> Selector:
        return Selector.__from_resolver__(
            _complement_resolver(self.meta.column_resolver.unwrap())
        )

    def __invert__(self) -> Selector:
        return self.complement()


def numeric() -> Selector:
    """Select all numeric columns."""
    return Selector.__from_resolver__(
        _dtype_resolver(lambda d: isinstance(d, dt.NumericType))
    )


def string() -> Selector:
    """Select all string columns."""
    return Selector.__from_resolver__(
        _dtype_resolver(lambda d: isinstance(d, dt.StringType))
    )


def boolean() -> Selector:
    """Select all boolean columns."""
    return Selector.__from_resolver__(
        _dtype_resolver(lambda d: isinstance(d, dt.Boolean))
    )


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes."""
    return Selector.__from_resolver__(_dtype_resolver(lambda d: isinstance(d, dtypes)))


__all__ = ["boolean", "by_dtype", "numeric", "string"]
