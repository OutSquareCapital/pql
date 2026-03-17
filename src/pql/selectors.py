"""Column selectors for PQL."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Self, overload, override

import pyochain as pc

from . import _datatypes as dt  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr
from ._meta import (
    SENTINEL_COL,
    ColumnResolver,
    MultiMeta,
    all_columns_resolver,
    complement_resolver,
    difference_resolver,
    dtype_resolver,
    intersection_resolver,
    name_resolver,
    ordered_name_resolver,
    union_resolver,
)

if TYPE_CHECKING:
    from .sql.typing import IntoExpr
SELECTOR = "__selector__"


class Selector(Expr):
    """Column selector based on dtype predicates."""

    meta: MultiMeta  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def _resolver(self) -> ColumnResolver:
        return self.meta.resolver

    @classmethod
    def __from_resolver__(cls, resolver: ColumnResolver) -> Self:
        return cls(SENTINEL_COL, MultiMeta(SELECTOR, resolver=resolver))

    @overload
    def union(self, other: Selector) -> Selector: ...
    @overload
    def union(self, other: IntoExpr) -> Expr: ...
    def union(self, other: IntoExpr) -> Selector | Expr:
        match other:
            case Selector():
                return Selector.__from_resolver__(
                    union_resolver(self._resolver, other._resolver)
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
            case Selector():
                return Selector.__from_resolver__(
                    intersection_resolver(self._resolver, other._resolver)
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
            case Selector():
                return Selector.__from_resolver__(
                    difference_resolver(self._resolver, other._resolver)
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
        return Selector.__from_resolver__(complement_resolver(self._resolver))

    @override
    def __invert__(self) -> Selector:
        return self.complement()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes."""
    return Selector.__from_resolver__(dtype_resolver(*dtypes))


def numeric() -> Selector:
    """Select all numeric columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.NumericType))


def string() -> Selector:
    """Select all string columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.StringType))


def boolean() -> Selector:
    """Select all boolean columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Boolean))


def all() -> Selector:
    """Select all columns."""
    return Selector.__from_resolver__(all_columns_resolver())


def float() -> Selector:
    """Select all float columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.FloatType))


def integer() -> Selector:
    """Select all integer columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.IntegerType))


def signed_integer() -> Selector:
    """Select all signed integer columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.SignedIntegerType))


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.UnsignedIntegerType))


def temporal() -> Selector:
    """Select all temporal columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.TemporalType))


def date() -> Selector:
    """Select all date columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Date))


def time() -> Selector:
    """Select all time columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Time))


def duration() -> Selector:
    """Select all duration columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Duration))


def binary() -> Selector:
    """Select all binary columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Binary))


def enum() -> Selector:
    """Select all enum columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Enum))


def decimal() -> Selector:
    """Select all decimal columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Decimal))


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.NestedType))


def struct() -> Selector:
    """Select all struct columns."""
    return Selector.__from_resolver__(dtype_resolver(dt.Struct))


# ──── name-based selectors ────


def matches(pattern: str) -> Selector:
    """Select columns whose names match the given regex pattern."""
    compiled = re.compile(pattern)
    return Selector.__from_resolver__(
        name_resolver(lambda name: compiled.search(name) is not None)
    )


def by_name(*names: str) -> Selector:
    """Select columns by exact name."""
    return Selector.__from_resolver__(ordered_name_resolver(pc.Seq(names)))


def starts_with(*prefix: str) -> Selector:
    """Select columns whose names start with any of the given prefixes."""
    return Selector.__from_resolver__(
        name_resolver(lambda name: name.startswith(prefix))
    )


def ends_with(*suffix: str) -> Selector:
    """Select columns whose names end with any of the given suffixes."""
    return Selector.__from_resolver__(name_resolver(lambda name: name.endswith(suffix)))


def contains(*substring: str) -> Selector:
    """Select columns whose names contain any of the given substrings."""
    subs = pc.Seq(substring)
    return Selector.__from_resolver__(
        name_resolver(lambda name: subs.iter().any(lambda s: s in name))
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
