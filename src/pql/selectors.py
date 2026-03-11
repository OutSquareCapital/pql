"""Column selectors for PQL."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Self, overload, override

import duckdb
import pyochain as pc

from . import _datatypes as dt, sql  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr
from ._meta import ExprMeta, MultiExpansion

if TYPE_CHECKING:
    from ._schema import ColumnResolver, Schema
    from .sql.typing import IntoExpr

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


def ordered_name_resolver(names: pc.Seq[str]) -> ColumnResolver:
    return lambda schema: names.iter().filter(lambda name: name in schema).collect()


def _dtype_resolver(*on: type[dt.DataType]) -> ColumnResolver:
    return lambda schema: (
        schema.items()
        .iter()
        .filter_star(lambda _, dtype: isinstance(dtype, on))
        .map_star(lambda name, _: name)
        .collect()
    )


def _name_resolver(predicate: Callable[[str], bool]) -> ColumnResolver:
    return lambda schema: schema.keys().iter().filter(predicate).collect()


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
            expansion=pc.Some(
                MultiExpansion(
                    resolver,
                    lambda col: _replay_transform(str(_SENTINEL_COL), str(col)),
                )
            ),
        )
        return cls(_SENTINEL_COL, pc.Some(meta))

    @override
    def _new(self, value: sql.SqlExpr, meta: pc.Option[ExprMeta] = pc.NONE) -> Self:
        """Preserve expansion resolver and build a transform from the sentinel-based SQL."""
        new_expansion = self.meta.expansion.map(
            lambda exp: MultiExpansion(
                exp.resolver,
                lambda col: _replay_transform(str(value), str(col)),
            )
        )
        new_meta = replace(meta.unwrap_or(self.meta), expansion=new_expansion)
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
                        self.meta.expansion.unwrap().resolver,
                        other.meta.expansion.unwrap().resolver,
                    )
                )
            case _:
                return Expr.__or__(self, other)

    @overload
    def __or__(self, other: Selector) -> Selector: ...
    @overload
    def __or__(self, other: IntoExpr) -> Expr: ...
    @override
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
                        self.meta.expansion.unwrap().resolver,
                        other.meta.expansion.unwrap().resolver,
                    )
                )
            case _:
                return Expr.__and__(self, other)

    @overload
    def __and__(self, other: Selector) -> Selector: ...
    @overload
    def __and__(self, other: IntoExpr) -> Expr: ...
    @override
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
                        self.meta.expansion.unwrap().resolver,
                        other.meta.expansion.unwrap().resolver,
                    )
                )
            case _:
                return Expr.__sub__(self, other)

    @overload
    def __sub__(self, other: Selector) -> Selector: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    @override
    def __sub__(self, other: IntoExpr) -> Selector | Expr:
        return self.difference(other)

    def complement(self) -> Selector:
        return Selector.__from_resolver__(
            _complement_resolver(self.meta.expansion.unwrap().resolver)
        )

    @override
    def __invert__(self) -> Selector:
        return self.complement()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes."""
    return Selector.__from_resolver__(_dtype_resolver(*dtypes))


def numeric() -> Selector:
    """Select all numeric columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.NumericType))


def string() -> Selector:
    """Select all string columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.StringType))


def boolean() -> Selector:
    """Select all boolean columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Boolean))


def all() -> Selector:
    """Select all columns."""
    return Selector.__from_resolver__(all_columns_resolver)


def float() -> Selector:
    """Select all float columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.FloatType))


def integer() -> Selector:
    """Select all integer columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.IntegerType))


def signed_integer() -> Selector:
    """Select all signed integer columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.SignedIntegerType))


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.UnsignedIntegerType))


def temporal() -> Selector:
    """Select all temporal columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.TemporalType))


def date() -> Selector:
    """Select all date columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Date))


def time() -> Selector:
    """Select all time columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Time))


def duration() -> Selector:
    """Select all duration columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Duration))


def binary() -> Selector:
    """Select all binary columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Binary))


def enum() -> Selector:
    """Select all enum columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Enum))


def decimal() -> Selector:
    """Select all decimal columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Decimal))


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.NestedType))


def struct() -> Selector:
    """Select all struct columns."""
    return Selector.__from_resolver__(_dtype_resolver(dt.Struct))


# ──── name-based selectors ────


def matches(pattern: str) -> Selector:
    """Select columns whose names match the given regex pattern."""
    compiled = re.compile(pattern)
    return Selector.__from_resolver__(
        _name_resolver(lambda name: compiled.search(name) is not None)
    )


def by_name(*names: str) -> Selector:
    """Select columns by exact name."""
    return Selector.__from_resolver__(ordered_name_resolver(pc.Seq(names)))


def starts_with(*prefix: str) -> Selector:
    """Select columns whose names start with any of the given prefixes."""
    return Selector.__from_resolver__(
        _name_resolver(lambda name: name.startswith(prefix))
    )


def ends_with(*suffix: str) -> Selector:
    """Select columns whose names end with any of the given suffixes."""
    return Selector.__from_resolver__(
        _name_resolver(lambda name: name.endswith(suffix))
    )


def contains(*substring: str) -> Selector:
    """Select columns whose names contain any of the given substrings."""
    subs = pc.Seq(substring)
    return Selector.__from_resolver__(
        _name_resolver(lambda name: subs.iter().any(lambda s: s in name))
    )


__all__ = [
    "all",
    "binary",
    "boolean",
    "by_dtype",
    "by_name",
    "contains",
    "date",
    "decimal",
    "duration",
    "ends_with",
    "enum",
    "float",
    "integer",
    "matches",
    "nested",
    "numeric",
    "signed_integer",
    "starts_with",
    "string",
    "struct",
    "temporal",
    "time",
    "unsigned_integer",
]
